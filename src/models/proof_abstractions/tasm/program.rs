use std::panic::catch_unwind;
use std::panic::RefUnwindSafe;

use itertools::Itertools;
use tasm_lib::library::Library;
use tasm_lib::triton_vm::error::InstructionError;
use tasm_lib::triton_vm::prelude::*;
use tasm_lib::twenty_first::math::b_field_element::BFieldElement;
use tasm_lib::Digest;
use tracing::debug;

use super::environment;
use super::prover_job::ProverJob;
use super::prover_job::ProverJobError;
use super::prover_job::ProverJobResult;
use super::prover_job::ProverJobSettings;
use crate::job_queue::triton_vm::TritonVmJobPriority;
use crate::job_queue::triton_vm::TritonVmJobQueue;

#[derive(Debug, Clone)]
pub enum ConsensusError {
    RustShadowPanic(String),
    TritonVMPanic(String, InstructionError),
}

/// A `ConsensusProgram` represents the logic subprogram for transaction or
/// block validity.
pub trait ConsensusProgram
where
    Self: RefUnwindSafe + std::fmt::Debug,
{
    /// The canonical reference source code for the consensus program, written in
    /// the subset of rust that the tasm-lang compiler understands. To run this
    /// program, call [`Self::run_rust`], which spawns a new thread, boots the
    /// environment, and executes the program.
    fn source(&self);

    /// Helps identify all imported Triton assembly snippets.
    /// You probably want to use [`Self::code`].
    // Implemented this way to ensure synchronicity between the library in use
    // and the actual code.
    #[doc(hidden)]
    fn library_and_code(&self) -> (Library, Vec<LabelledInstruction>);

    /// A derivative of source, in Triton-assembler (tasm) rather than rust. Either
    /// produced automatically or hand-optimized.
    ///
    /// See also [`Self::program`].
    fn code(&self) -> Vec<LabelledInstruction> {
        let (_, code) = self.library_and_code();
        code
    }

    /// The Triton VM [`Program`].
    fn program(&self) -> Program {
        Program::new(&self.code())
    }

    /// The [program](Self::program)'s hash [digest](Digest).
    //
    // note: we do not provide a default impl because implementors should cache
    // their Digest with OnceLock.
    fn hash(&self) -> Digest;

    /// Run the source program natively in rust, but with the emulated TritonVM
    /// environment for input, output, nondeterminism, and program digest.
    fn run_rust(
        &self,
        input: &PublicInput,
        nondeterminism: NonDeterminism,
    ) -> Result<Vec<BFieldElement>, ConsensusError> {
        debug!(
            "Running consensus program with input: {}",
            input.individual_tokens.iter().map(|b| b.value()).join(",")
        );
        let program_digest = catch_unwind(|| self.hash()).unwrap_or_default();
        let emulation_result = catch_unwind(|| {
            environment::init(program_digest, &input.individual_tokens, nondeterminism);
            self.source();
            environment::PUB_OUTPUT.take()
        });

        emulation_result.map_err(|e| ConsensusError::RustShadowPanic(format!("{e:?}")))
    }

    /// Use Triton VM to run the tasm code.
    ///
    /// Should only be called in tests. In production code, use [`Self::run_rust`]
    /// instead – it's faster.
    #[cfg(test)]
    fn run_tasm(
        &self,
        input: &PublicInput,
        nondeterminism: NonDeterminism,
    ) -> Result<Vec<BFieldElement>, ConsensusError> {
        let mut vm_state = VMState::new(self.program(), input.clone(), nondeterminism.clone());
        tasm_lib::maybe_write_debuggable_vm_state_to_disk(&vm_state);

        let init_stack = vm_state.op_stack.clone();
        if let Err(err) = vm_state.run() {
            let err_str = format!("Triton VM failed.\nError: {err}\nVMState:\n{vm_state}");
            debug!(err_str);
            return Err(ConsensusError::TritonVMPanic(err_str, err));
        }

        // Do some sanity checks that are likely to catch programming
        // errors in the consensus program. This doesn't catch
        // soundness errors, though, since a valid proof could still be
        // generated even though one of these checks fail.
        assert!(
            vm_state.secret_digests.is_empty(),
            "Secret digest list must be empty after executing consensus program"
        );
        assert_eq!(&init_stack, &vm_state.op_stack);

        Ok(vm_state.public_output)
    }

    /// `Ok(())` iff the given input & non-determinism triggers the failure of
    /// either the instruction `assert` or `assert_vector`, and if that
    /// instruction's error ID is one of the expected error IDs.
    #[cfg(test)]
    fn test_assertion_failure(
        &self,
        public_input: PublicInput,
        non_determinism: NonDeterminism,
        expected_error_ids: &[i128],
    ) -> proptest::test_runner::TestCaseResult {
        let fail = |reason: String| Err(proptest::test_runner::TestCaseError::Fail(reason.into()));

        let tasm_result = self.run_tasm(&public_input, non_determinism.clone());
        let Err(ConsensusError::TritonVMPanic(_, err)) = tasm_result else {
            return fail("expected a failure in Triton VM, but it halted gracefully".into());
        };

        let err = match err {
            InstructionError::AssertionFailed(err)
            | InstructionError::VectorAssertionFailed(_, err) => err,
            _ => return fail(format!("expected an assertion failure, but got: {err}")),
        };

        let ids_str = expected_error_ids.iter().join(", ");
        let expected_ids_str = format!("expected an error ID in {{{ids_str}}}");
        let Some(err_id) = err.id else {
            return fail(format!("{expected_ids_str}, but found none"));
        };

        proptest::prop_assert!(
            expected_error_ids.contains(&err_id),
            "{expected_ids_str}, but found {err_id}",
        );

        let rust_result = self.run_rust(&public_input, non_determinism.clone());
        proptest::prop_assert!(rust_result.is_err());

        Ok(())
    }

    /// Run the program and generate a proof for it, assuming running halts
    /// gracefully.
    ///
    /// If we are in the test environment, try reading it from disk. And if it
    /// not there, generate it and store it to disk.
    ///
    /// This method is a thin wrapper around [`prove_consensus_program`], which
    /// does the same but for arbitrary programs.
    #[allow(async_fn_in_trait)] // Trait must be public bc of benchmarks.
    #[allow(private_interfaces)] // Trait must be public bc of benchmarks.
    #[doc(hidden)]
    async fn prove(
        &self,
        claim: Claim,
        nondeterminism: NonDeterminism,
        triton_vm_job_queue: &TritonVmJobQueue,
        proof_job_options: TritonVmProofJobOptions,
    ) -> anyhow::Result<Proof> {
        prove_consensus_program(
            self.program(),
            claim,
            nondeterminism,
            triton_vm_job_queue,
            proof_job_options,
        )
        .await
    }
}

/// Run the program and generate a proof for it, assuming the Triton VM run
/// halts gracefully.
///
/// If we are in a test environment, try reading it from disk. If it is not
/// there, generate it and store it to disk.
///
/// This method works for arbitrary programs, including ones that do not
/// implement trait [`ConsensusProgram`].
///
/// The proof is executed as a triton-vm-job-queue job which ensures that
/// no two tasks run the prover simultaneously.
pub(crate) async fn prove_consensus_program(
    program: Program,
    claim: Claim,
    nondeterminism: NonDeterminism,
    triton_vm_job_queue: &TritonVmJobQueue,
    proof_job_options: TritonVmProofJobOptions,
) -> anyhow::Result<Proof> {
    // create a triton-vm-job-queue job for generating this proof.
    let job = ProverJob {
        program,
        claim,
        nondeterminism,
        job_settings: proof_job_options.job_settings,
    };

    // queue the job and await the result.
    // todo: perhaps the priority should (somehow) depend on type of Program?
    let result = triton_vm_job_queue
        .add_job(Box::new(job), proof_job_options.job_priority)
        .await?
        .result()
        .await?;

    // obtain resulting proof.
    let result: Result<Proof, ProverJobError> = result
        .into_any()
        .downcast::<ProverJobResult>()
        .expect("downcast should succeed, else bug")
        .into();

    Ok(result?)
}

#[derive(Clone, Debug, Default, Copy)]
pub(crate) struct TritonVmProofJobOptions {
    pub job_priority: TritonVmJobPriority,
    pub job_settings: ProverJobSettings,
}

impl From<(TritonVmJobPriority, Option<u8>)> for TritonVmProofJobOptions {
    fn from(v: (TritonVmJobPriority, Option<u8>)) -> Self {
        let (job_priority, max_log2_padded_height_for_proofs) = v;
        Self {
            job_priority,
            job_settings: ProverJobSettings {
                max_log2_padded_height_for_proofs,
            },
        }
    }
}

#[cfg(test)]
impl From<TritonVmJobPriority> for TritonVmProofJobOptions {
    fn from(job_priority: TritonVmJobPriority) -> Self {
        Self {
            job_priority,
            job_settings: Default::default(),
        }
    }
}

#[cfg(test)]
pub mod test {
    use std::fs::create_dir_all;
    use std::fs::File;
    use std::io::stdout;
    use std::io::Read;
    use std::io::Write;
    use std::path::Path;
    use std::path::PathBuf;
    use std::time::Duration;
    use std::time::SystemTime;

    use itertools::Itertools;
    use rand::seq::SliceRandom;
    use rand::thread_rng;
    use reqwest::header::HeaderMap;
    use reqwest::header::HeaderValue;
    use reqwest::Url;
    use tasm_lib::triton_vm;
    use tasm_lib::twenty_first::util_types::algebraic_hasher::AlgebraicHasher;
    use tracing::debug;
    use tracing::Span;

    use super::*;
    use crate::models::blockchain::shared::Hash;
    use crate::triton_vm::stark::Stark;

    const TEST_DATA_DIR: &str = "test_data";
    const TEST_NAME_HTTP_HEADER_KEY: &str = "Test-Name";

    pub(crate) fn consensus_program_negative_test<T: ConsensusProgram>(
        consensus_program: T,
        input: &PublicInput,
        nondeterminism: NonDeterminism,
        expected_error_ids: &[i128],
    ) {
        let rust_result = consensus_program.run_rust(input, nondeterminism.clone());
        assert2::assert!(let Err(ConsensusError::RustShadowPanic(_)) = rust_result);

        let program_name = core::any::type_name::<T>();
        let tasm_result = consensus_program.run_tasm(input, nondeterminism);
        let consensus_err = tasm_result.expect_err(&format!(
            "negative test failed to fail for consensus program {program_name}"
        ));
        let ConsensusError::TritonVMPanic(_, instruction_error) = consensus_err else {
            panic!("Triton VM must fail for consensus program {program_name}")
        };
        let assertion_err = match instruction_error {
            InstructionError::AssertionFailed(err)
            | InstructionError::VectorAssertionFailed(_, err) => err,
            _ => panic!("Triton VM must fail on assertion, but got: {instruction_error}"),
        };
        let err_id = assertion_err
            .id
            .expect("failed assertion must have an error ID, but does not");

        let expected_error_ids_str = expected_error_ids.iter().join(", ");
        assert!(
            expected_error_ids.contains(&err_id),
            "error ID {err_id} ∉ {{{expected_error_ids_str}}}\nTriton VM execution failed due to \
            unfulfilled assertion with error ID {err_id}, but expected one of the following IDs: \
            {{{expected_error_ids_str}}}",
        );
    }

    /// Derive a file name from the claim, includes the extension
    fn proof_filename(claim: &Claim) -> String {
        let base_name = Hash::hash(claim).to_hex();

        format!("{base_name}.proof")
    }

    fn proof_path(claim: &Claim) -> PathBuf {
        let name = proof_filename(claim);
        let mut path = PathBuf::new();
        path.push(TEST_DATA_DIR);
        path.push(Path::new(&name));

        path
    }

    /// First, attempt to load the proof from disk. If it does not exist,
    /// attempt to fetch it online. If that also fails, run the prover and
    /// save the proof before returning it.
    pub(crate) fn load_proof_or_produce_and_save(
        claim: &Claim,
        program: Program,
        nondeterminism: NonDeterminism,
    ) -> Proof {
        let name = proof_filename(claim);
        match try_load_proof_from_disk(claim) {
            Some(proof) => {
                debug!(" - Loaded proof from disk: {name}.");
                assert!(
                    triton_vm::verify(Stark::default(), claim, &proof),
                    "proof loaded from disk is invalid"
                );
                proof
            }
            None => {
                debug!("Proof not found on disk.");
                match try_fetch_and_verify_proof_from_server(claim) {
                    Some(proof) => proof,
                    None => {
                        debug!("Proof not found on proof servers - Proving locally ... ");
                        stdout().flush().expect("could not flush terminal");
                        let tick = SystemTime::now();
                        let proof = produce_and_save_proof(claim, program, nondeterminism);
                        let duration = SystemTime::now().duration_since(tick).unwrap();
                        debug!(
                            "success! Proof time: {:?}. Proof stored to disk: {name}",
                            duration
                        );
                        proof
                    }
                }
            }
        }
    }

    /// Tries to load a proof for the claim from the test data directory
    fn try_load_proof_from_disk(claim: &Claim) -> Option<Proof> {
        let path = proof_path(claim);
        let Ok(mut input_file) = File::open(path.clone()) else {
            debug!("cannot open file '{}' -- might not exist", path.display());
            return None;
        };
        let mut file_contents = vec![];
        if input_file.read_to_end(&mut file_contents).is_err() {
            debug!("cannot read file '{}'", path.display());
            return None;
        }
        let mut proof_data = vec![];
        for ch in file_contents.chunks(8) {
            if let Ok(eight_bytes) = TryInto::<[u8; 8]>::try_into(ch) {
                proof_data.push(BFieldElement::new(u64::from_be_bytes(eight_bytes)));
            } else {
                debug!("cannot cast chunk to eight bytes");
                return None;
            }
        }
        let proof = Proof(proof_data);
        Some(proof)
    }

    /// Load a list of proof-servers from test data directory
    fn load_servers() -> Vec<Url> {
        let mut server_list_path = PathBuf::new();
        server_list_path.push(TEST_DATA_DIR);
        server_list_path.push(Path::new("proof_servers").with_extension("txt"));
        let Ok(mut input_file) = File::open(server_list_path.clone()) else {
            debug!(
                "cannot proof-server list '{}' -- file might not exist",
                server_list_path.display()
            );
            return vec![];
        };
        let mut file_contents = vec![];
        if input_file.read_to_end(&mut file_contents).is_err() {
            debug!("cannot read file '{}'", server_list_path.display());
            return vec![];
        }
        let Ok(file_as_string) = String::from_utf8(file_contents) else {
            debug!(
                "cannot parse file '{}' -- is it valid utf8?",
                server_list_path.display()
            );
            return vec![];
        };
        file_as_string
            .lines()
            .map(|s| Url::parse(s).expect("Must be able to parse string '{s}' as URL"))
            .collect()
    }

    #[test]
    fn test_load_servers() {
        let servers = load_servers();
        for server in servers {
            println!("read server: {}", server);
        }
    }

    /// Queries known servers for proofs.
    ///
    /// The proof-servers file is located in `proof_servers.txt` test data
    /// directory. It should contain one line per URL, ending in a slash.
    fn try_fetch_and_verify_proof_from_server(claim: &Claim) -> Option<Proof> {
        let filename = proof_filename(claim);
        let (proof, server) = try_fetch_from_server_inner(filename.clone())?;

        if !triton_vm::verify(Stark::default(), claim, &proof) {
            eprintln!("Invalid proof served by {server}. Proof {filename} does not verify.");
            return None;
        }

        let path = proof_path(claim);
        save_proof(&path, &proof);

        Some(proof)
    }

    /// Tries to fetch a proof from a server, does not validate the proof
    ///
    /// If a proof was found, returns it along with the URL of the server
    /// serving the proof. The caller should validate the proof. Does
    /// not store the proof to disk.
    /// TODO: Consider making this async.
    fn try_fetch_from_server_inner(filename: String) -> Option<(Proof, Url)> {
        fn get_test_name_from_tracing() -> String {
            match Span::current().metadata().map(|x| x.name()) {
                Some(test_name) => test_name.to_owned(),
                None => "unknown".to_owned(),
            }
        }

        fn attempt_to_get_test_name() -> String {
            let thread = std::thread::current();
            match thread.name() {
                Some(test_name) => {
                    if test_name.eq("tokio-runtime-worker") {
                        get_test_name_from_tracing()
                    } else {
                        test_name.to_owned()
                    }
                }
                None => get_test_name_from_tracing(),
            }
        }

        let mut servers = load_servers();
        servers.shuffle(&mut thread_rng());

        // Add test name to request allow server to see which test requires a proof
        let mut headers = HeaderMap::new();
        headers.insert(
            TEST_NAME_HTTP_HEADER_KEY,
            HeaderValue::from_str(&attempt_to_get_test_name())
                .unwrap_or_else(|_| panic!("Must be to add test name to HTTP header")),
        );

        // TODO: Use regular (non-blocking) reqwest client if this function
        // is made `async`.
        for server in servers {
            let server_ = server.clone();
            let filename_ = filename.clone();
            let headers_ = headers.clone();
            let handle = std::thread::spawn(move || {
                let http_client = reqwest::blocking::Client::builder()
                    .timeout(Duration::from_secs(10))
                    .default_headers(headers_)
                    .build()
                    .expect("Must be able to build HTTP client instance");
                let url = server_.join(&filename_).unwrap_or_else(|_| {
                    panic!("Must be able to form URL. Got: '{server_}' and '{filename_}'.")
                });
                debug!("requesting: <{url}>");
                let Ok(response) = http_client.get(url).send() else {
                    println!(
                        "server '{}' failed for file '{}'; trying next ...",
                        server_.clone(),
                        filename_
                    );

                    return None;
                };

                Some((response.status(), response.bytes()))
            });

            let Some((status_code, body)) = handle.join().unwrap() else {
                eprintln!("Could not connect to server {server}.");
                continue;
            };

            if !status_code.is_success() {
                eprintln!("{server} responded with {status_code}");
                continue;
            }

            let Ok(file_contents) = body else {
                eprintln!(
                    "error reading file '{}' from server '{}'; trying next ...",
                    filename, server
                );

                continue;
            };

            let mut proof_data = vec![];
            for ch in file_contents.chunks(8) {
                if let Ok(eight_bytes) = TryInto::<[u8; 8]>::try_into(ch) {
                    proof_data.push(BFieldElement::new(u64::from_be_bytes(eight_bytes)));
                } else {
                    eprintln!("cannot cast chunk to eight bytes. Server was: {server}");
                    continue;
                }
            }

            let proof = Proof(proof_data);
            println!("got proof.");

            return Some((proof, server));
        }

        println!("No known servers serve file `{}`", filename);

        None
    }

    #[tokio::test]
    async fn test_query_proof() {
        // Ensure file exists on machine, in case this machine syncs automatically with proof server
        let program = triton_program!(halt);
        let claim = Claim::about_program(&program);
        prove_consensus_program(
            program,
            claim.clone(),
            NonDeterminism::default(),
            &TritonVmJobQueue::dummy(),
            TritonVmProofJobOptions::default(),
        )
        .await
        .unwrap();

        // Then verify that the proof server has this file
        let filename = proof_filename(&claim);
        let (proof, url) =
            try_fetch_from_server_inner(filename).expect("Expected this proof on the proof server");
        assert!(
            triton_vm::verify(Stark::default(), &claim, &proof),
            "Returned proof from {url} must be valid"
        );
    }

    /// Call Triton VM prover to produce a proof and save it to disk.
    fn produce_and_save_proof(
        claim: &Claim,
        program: Program,
        nondeterminism: NonDeterminism,
    ) -> Proof {
        let name = proof_filename(claim);
        let mut path = PathBuf::new();
        path.push(TEST_DATA_DIR);
        create_dir_all(&path)
            .unwrap_or_else(|_| panic!("cannot create '{TEST_DATA_DIR}' directory"));
        path.push(Path::new(&name));

        let proof = triton_vm::prove(Stark::default(), claim, program, nondeterminism)
            .expect("cannot produce proof");

        save_proof(&path, &proof);
        proof
    }

    /// Store a proof to the given file
    fn save_proof(path: &PathBuf, proof: &Proof) {
        let proof_data = proof
            .0
            .iter()
            .copied()
            .flat_map(|b| b.value().to_be_bytes())
            .collect_vec();
        let mut output_file = File::create(path).expect("cannot open file for writing");
        output_file
            .write_all(&proof_data)
            .expect("cannot write to file");
    }
}
