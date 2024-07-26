use std::collections::HashMap;

use crate::models::proof_abstractions::tasm::builtins as tasmlib;
use get_size::GetSize;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use tasm_lib::{
    memory::{encode_to_memory, FIRST_NON_DETERMINISTICALLY_INITIALIZED_MEMORY_ADDRESS},
    structure::tasm_object::TasmObject,
    triton_vm::{
        self,
        instruction::LabelledInstruction,
        prelude::{BFieldCodec, BFieldElement},
        program::{NonDeterminism, Program, PublicInput},
        proof::{Claim, Proof},
        stark::Stark,
    },
    twenty_first::util_types::algebraic_hasher::AlgebraicHasher,
    Digest,
};

use crate::models::proof_abstractions::SecretWitness;
use crate::models::{
    blockchain::shared::Hash, proof_abstractions::tasm::program::ConsensusProgram,
};
use crate::models::{
    blockchain::transaction::{
        primitive_witness::PrimitiveWitness,
        validity::{
            collect_lock_scripts::{CollectLockScripts, CollectLockScriptsWitness},
            collect_type_scripts::CollectTypeScripts,
            kernel_to_outputs::KernelToOutputs,
            removal_records_integrity::RemovalRecordsIntegrityWitness,
        },
    },
    proof_abstractions::mast_hash::MastHash,
};

use super::{
    collect_type_scripts::CollectTypeScriptsWitness, kernel_to_outputs::KernelToOutputsWitness,
    removal_records_integrity::RemovalRecordsIntegrity,
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, GetSize, BFieldCodec, TasmObject)]
pub struct ProofCollection {
    pub removal_records_integrity: Proof,
    pub collect_lock_scripts: Proof,
    pub lock_scripts_halt: Vec<Proof>,
    pub kernel_to_outputs: Proof,
    pub collect_type_scripts: Proof,
    pub type_scripts_halt: Vec<Proof>,
    pub lock_script_hashes: Vec<Digest>,
    pub type_script_hashes: Vec<Digest>,
    pub kernel_mast_hash: Digest,
    pub salted_inputs_hash: Digest,
    pub salted_outputs_hash: Digest,
}

impl ProofCollection {
    // The following const Digests should match with the like named program's
    // hash. The test `test::const_subprogram_hashes_are_correct` tests for this
    // equality, and if unequal, gives the right expression to be plopped in
    // here.
    // These consts are used as hardcoded values in the rust-simulated Triton
    // environment and in the tasm code.
    const REMOVAL_RECORDS_INTEGRITY_PROGRAM_DIGEST: Digest = Digest::new([
        BFieldElement::new(10197736943087578891),
        BFieldElement::new(2589769213358159759),
        BFieldElement::new(12122773685110410087),
        BFieldElement::new(8367354675667512484),
        BFieldElement::new(16535400299838854923),
    ]);
    const KERNEL_TO_OUTPUTS_PROGRAM_DIGEST: Digest = Digest::new([
        BFieldElement::new(9484121616554123823),
        BFieldElement::new(15164679326249580050),
        BFieldElement::new(4679237130163686053),
        BFieldElement::new(363984525662568692),
        BFieldElement::new(5520958396043760478),
    ]);
    const COLLECT_LOCK_SCRIPTS_PROGRAM_DIGEST: Digest = Digest::new([
        BFieldElement::new(5187444292971315662),
        BFieldElement::new(17273354872178271392),
        BFieldElement::new(2636812070734160948),
        BFieldElement::new(18034626495561690787),
        BFieldElement::new(7006565901007114771),
    ]);
    const COLLECT_TYPE_SCRIPTS_PROGRAM_DIGEST: Digest = Digest::new([
        BFieldElement::new(10915177323912360979),
        BFieldElement::new(12776913432129457760),
        BFieldElement::new(8007322389620416096),
        BFieldElement::new(11496378975485667762),
        BFieldElement::new(10952609886739350684),
    ]);
    fn extract_specific_witnesses(
        primitive_witness: &PrimitiveWitness,
    ) -> (
        RemovalRecordsIntegrityWitness,
        CollectLockScriptsWitness,
        KernelToOutputsWitness,
        CollectTypeScriptsWitness,
    ) {
        // collect witnesses
        let removal_records_integrity_witness =
            RemovalRecordsIntegrityWitness::from(primitive_witness);
        let collect_lock_scripts_witness = CollectLockScriptsWitness::from(primitive_witness);
        let kernel_to_outputs_witness = KernelToOutputsWitness::from(primitive_witness);
        let collect_type_scripts_witness = CollectTypeScriptsWitness::from(primitive_witness);

        (
            removal_records_integrity_witness,
            collect_lock_scripts_witness,
            kernel_to_outputs_witness,
            collect_type_scripts_witness,
        )
    }
    pub fn can_produce(primitive_witness: &PrimitiveWitness) -> bool {
        let (
            removal_records_integrity_witness,
            collect_lock_scripts_witness,
            kernel_to_outputs_witness,
            collect_type_scripts_witness,
        ) = Self::extract_specific_witnesses(primitive_witness);

        // verify graceful halts
        let removal_records_integrity_halts = if let Ok(output) = RemovalRecordsIntegrity.run_rust(
            &removal_records_integrity_witness.standard_input(),
            removal_records_integrity_witness.nondeterminism(),
        ) {
            output == removal_records_integrity_witness.output()
        } else {
            false
        };

        let collect_lock_scripts_halts = if let Ok(output) = CollectLockScripts.run_rust(
            &collect_lock_scripts_witness.standard_input(),
            collect_lock_scripts_witness.nondeterminism(),
        ) {
            output == collect_lock_scripts_witness.output()
        } else {
            false
        };

        let kernel_to_outputs_halts = if let Ok(output) = KernelToOutputs.run_rust(
            &kernel_to_outputs_witness.standard_input(),
            kernel_to_outputs_witness.nondeterminism(),
        ) {
            output == kernel_to_outputs_witness.output()
        } else {
            false
        };

        let collect_type_scripts_halts = if let Ok(output) = CollectTypeScripts.run_rust(
            &collect_type_scripts_witness.standard_input(),
            collect_type_scripts_witness.nondeterminism(),
        ) {
            output == collect_type_scripts_witness.output()
        } else {
            false
        };

        let txk_mast_hash = primitive_witness.kernel.mast_hash();
        let txk_mast_hash_as_input = PublicInput::new(txk_mast_hash.reversed().values().to_vec());
        let salted_inputs_hash = Hash::hash(&primitive_witness.input_utxos);
        let salted_outputs_hash = Hash::hash(&primitive_witness.output_utxos);

        let all_lock_scripts_halt = primitive_witness
            .lock_scripts_and_witnesses
            .iter()
            .all(|lsaw| lsaw.halts_gracefully(txk_mast_hash_as_input.clone()));
        let all_type_scripts_halt = primitive_witness
            .type_scripts_and_witnesses
            .iter()
            .all(|ts| ts.halts_gracefully(txk_mast_hash, salted_inputs_hash, salted_outputs_hash));

        removal_records_integrity_halts
            || collect_lock_scripts_halts
            || kernel_to_outputs_halts
            || collect_type_scripts_halts
            || all_lock_scripts_halt
            || all_type_scripts_halt
    }

    pub fn produce(primitive_witness: &PrimitiveWitness) -> Self {
        let (
            removal_records_integrity_witness,
            collect_lock_scripts_witness,
            kernel_to_outputs_witness,
            collect_type_scripts_witness,
        ) = Self::extract_specific_witnesses(primitive_witness);

        let txk_mast_hash = primitive_witness.kernel.mast_hash();
        let txk_mast_hash_as_input = PublicInput::new(txk_mast_hash.reversed().values().to_vec());
        let salted_inputs_hash = Hash::hash(&primitive_witness.input_utxos);
        let salted_outputs_hash = Hash::hash(&primitive_witness.output_utxos);
        println!("proving, txk hash: {}", txk_mast_hash);
        println!("proving, salted inputs hash: {}", salted_inputs_hash);
        println!("proving, salted outputs hash: {}", salted_outputs_hash);

        // prove
        let removal_records_integrity = RemovalRecordsIntegrity.prove(
            &removal_records_integrity_witness.claim(),
            removal_records_integrity_witness.nondeterminism(),
        );
        let collect_lock_scripts = CollectLockScripts.prove(
            &collect_lock_scripts_witness.claim(),
            collect_lock_scripts_witness.nondeterminism(),
        );
        let kernel_to_outputs = KernelToOutputs.prove(
            &kernel_to_outputs_witness.claim(),
            kernel_to_outputs_witness.nondeterminism(),
        );
        let collect_type_scripts = CollectTypeScripts.prove(
            &collect_type_scripts_witness.claim(),
            collect_type_scripts_witness.nondeterminism(),
        );
        let lock_scripts_halt = primitive_witness
            .lock_scripts_and_witnesses
            .iter()
            .map(|lsaw| lsaw.prove(txk_mast_hash_as_input.clone()))
            .collect_vec();
        let type_scripts_halt = primitive_witness
            .type_scripts_and_witnesses
            .iter()
            .map(|tsaw| tsaw.prove(txk_mast_hash, salted_inputs_hash, salted_outputs_hash))
            .collect_vec();

        // collect hashes
        let lock_script_hashes = primitive_witness
            .lock_scripts_and_witnesses
            .iter()
            .map(|lsaw| lsaw.program.hash::<Hash>())
            .collect_vec();
        let type_script_hashes = primitive_witness
            .type_scripts_and_witnesses
            .iter()
            .map(|tsaw| tsaw.program.hash::<Hash>())
            .collect_vec();

        // assemble data into struct and return
        ProofCollection {
            removal_records_integrity,
            collect_lock_scripts,
            lock_scripts_halt,
            kernel_to_outputs,
            collect_type_scripts,
            type_scripts_halt,
            lock_script_hashes,
            type_script_hashes,
            kernel_mast_hash: txk_mast_hash,
            salted_inputs_hash,
            salted_outputs_hash,
        }
    }

    pub fn verify(&self, txk_mast_hash: Digest) -> bool {
        println!("verifying, txk hash: {}", txk_mast_hash);
        println!("verifying, salted inputs hash: {}", self.salted_inputs_hash);
        println!(
            "verifying, salted outputs hash: {}",
            self.salted_outputs_hash
        );
        // make sure we are talking about the same tx
        if self.kernel_mast_hash != txk_mast_hash {
            return false;
        }

        // compile claims
        let removal_records_integrity_claim = Claim {
            program_digest: RemovalRecordsIntegrity.program().hash::<Hash>(),
            input: self.kernel_mast_hash.reversed().values().to_vec(),
            output: self.salted_inputs_hash.values().to_vec(),
        };
        let kernel_to_outputs_claim = Claim {
            program_digest: KernelToOutputs.program().hash::<Hash>(),
            input: self.kernel_mast_hash.reversed().values().to_vec(),
            output: self.salted_outputs_hash.values().to_vec(),
        };
        let collect_lock_scripts_claim = Claim {
            program_digest: CollectLockScripts.program().hash::<Hash>(),
            input: self.salted_inputs_hash.reversed().values().to_vec(),
            output: self
                .lock_script_hashes
                .iter()
                .flat_map(|d| d.values())
                .collect_vec(),
        };
        let collect_type_scripts_claim = Claim {
            program_digest: CollectTypeScripts.program().hash::<Hash>(),
            input: [self.salted_inputs_hash, self.salted_outputs_hash]
                .into_iter()
                .flat_map(|d| d.reversed().values())
                .collect_vec(),
            output: self
                .type_script_hashes
                .iter()
                .flat_map(|d| d.values())
                .collect_vec(),
        };
        let lock_script_claims = self
            .lock_script_hashes
            .iter()
            .map(|lsh| Claim {
                program_digest: *lsh,
                input: self.kernel_mast_hash.reversed().values().to_vec(),
                output: vec![],
            })
            .collect_vec();
        let type_script_claims = self
            .type_script_hashes
            .iter()
            .map(|tsh| {
                Claim::new(*tsh).with_input(
                    [
                        self.kernel_mast_hash,
                        self.salted_inputs_hash,
                        self.salted_outputs_hash,
                    ]
                    .into_iter()
                    .flat_map(|d| d.reversed().values())
                    .collect_vec(),
                )
            })
            .collect_vec();

        // verify
        println!("verifying removal records integrity ...");
        let rri = triton_vm::verify(
            Stark::default(),
            &removal_records_integrity_claim,
            &self.removal_records_integrity,
        );
        println!("{rri}");
        println!("verifying kernel to outputs ...");
        let k2o = triton_vm::verify(
            Stark::default(),
            &kernel_to_outputs_claim,
            &self.kernel_to_outputs,
        );
        println!("{k2o}");
        println!("verifying collect lock scripts ...");
        let cls = triton_vm::verify(
            Stark::default(),
            &collect_lock_scripts_claim,
            &self.collect_lock_scripts,
        );
        println!("{cls}");
        println!("verifying collect type scripts ...");
        let cts = triton_vm::verify(
            Stark::default(),
            &collect_type_scripts_claim,
            &self.collect_type_scripts,
        );
        println!("{cts}");
        println!("verifying that all lock scripts halt ...");
        let lsh = lock_script_claims
            .iter()
            .zip(self.lock_scripts_halt.iter())
            .all(|(cl, pr)| triton_vm::verify(Stark::default(), cl, pr));
        println!("{lsh}");
        println!("verifying that all type scripts halt ...");
        let tsh = type_script_claims
            .iter()
            .zip(self.type_scripts_halt.iter())
            .all(|(cl, pr)| triton_vm::verify(Stark::default(), cl, pr));
        println!("{tsh}");

        // and all bits together and return
        rri && k2o && cls && cts && lsh && tsh
    }
}

impl SecretWitness for ProofCollection {
    fn standard_input(&self) -> PublicInput {
        PublicInput::new(self.kernel_mast_hash.reversed().values().to_vec())
    }

    fn program(&self) -> Program {
        Program::new(&self.code())
    }

    fn nondeterminism(&self) -> NonDeterminism {
        // set memory
        let mut memory = HashMap::default();
        encode_to_memory(
            &mut memory,
            FIRST_NON_DETERMINISTICALLY_INITIALIZED_MEMORY_ADDRESS,
            self.clone(),
        );

        NonDeterminism::default().with_ram(memory)
    }
}

impl ConsensusProgram for ProofCollection {
    fn source(&self) {
        let txk_digest: Digest = tasmlib::tasm_io_read_stdin___digest();
        let start_address: BFieldElement = FIRST_NON_DETERMINISTICALLY_INITIALIZED_MEMORY_ADDRESS;
        let pc: ProofCollection = tasmlib::decode_from_memory(start_address);

        assert_eq!(txk_digest, pc.kernel_mast_hash);

        let txk_mast_hash_as_input: Vec<BFieldElement> = txk_digest.reversed().values().to_vec();
        let salted_inputs_hash_as_input: Vec<BFieldElement> =
            pc.salted_inputs_hash.reversed().values().to_vec();
        let salted_inputs_hash_as_output: Vec<BFieldElement> =
            pc.salted_inputs_hash.values().to_vec();
        let salted_outputs_hash_as_input: Vec<BFieldElement> =
            pc.salted_outputs_hash.reversed().values().to_vec();
        let salted_outputs_hash_as_output: Vec<BFieldElement> =
            pc.salted_outputs_hash.values().to_vec();

        let removal_records_integrity_claim: Claim = Claim {
            program_digest: ProofCollection::REMOVAL_RECORDS_INTEGRITY_PROGRAM_DIGEST,
            input: txk_mast_hash_as_input.clone(),
            output: salted_inputs_hash_as_output.clone(),
        };
        let rri = tasmlib::verify(
            Stark::default(),
            removal_records_integrity_claim,
            &pc.removal_records_integrity,
        );
        assert!(rri);

        let kernel_to_outputs_claim: Claim = Claim {
            program_digest: ProofCollection::KERNEL_TO_OUTPUTS_PROGRAM_DIGEST,
            input: txk_mast_hash_as_input.clone(),
            output: salted_outputs_hash_as_output.clone(),
        };
        let k2o = tasmlib::verify(
            Stark::default(),
            kernel_to_outputs_claim,
            &pc.kernel_to_outputs,
        );
        assert!(k2o);

        let mut lock_script_hashes_as_output: Vec<BFieldElement> = Vec::<BFieldElement>::new();
        let mut i: usize = 0;
        while i < pc.lock_script_hashes.len() {
            let lock_script_hash: Digest = pc.lock_script_hashes[i];
            let mut j: usize = 0;
            while j < Digest::LEN {
                lock_script_hashes_as_output.push(lock_script_hash.values()[j]);
                j += 1;
            }
            i += 1;
        }
        let collect_lock_scripts_claim: Claim = Claim {
            program_digest: ProofCollection::COLLECT_LOCK_SCRIPTS_PROGRAM_DIGEST,
            input: salted_inputs_hash_as_input.clone(),
            output: lock_script_hashes_as_output,
        };
        let cls: bool = tasmlib::verify(
            Stark::default(),
            collect_lock_scripts_claim,
            &pc.collect_lock_scripts,
        );
        assert!(cls);

        let mut type_script_hashes_as_output: Vec<BFieldElement> = Vec::<BFieldElement>::new();
        let mut i: usize = 0;
        while i < pc.type_script_hashes.len() {
            let type_script_hash: Digest = pc.type_script_hashes[i];
            let mut j: usize = 0;
            while j < Digest::LEN {
                type_script_hashes_as_output.push(type_script_hash.values()[j]);
                j += 1;
            }
            i += 1;
        }
        let collect_type_scripts_claim: Claim = Claim {
            program_digest: ProofCollection::COLLECT_TYPE_SCRIPTS_PROGRAM_DIGEST,
            input: [
                salted_inputs_hash_as_input.clone(),
                salted_outputs_hash_as_input.clone(),
            ]
            .concat(),
            output: type_script_hashes_as_output,
        };
        let cts: bool = tasmlib::verify(
            Stark::default(),
            collect_type_scripts_claim,
            &pc.collect_type_scripts,
        );
        assert!(cts);

        let mut i: usize = 0;
        while i < pc.lock_script_hashes.len() {
            let lock_script_hash = pc.lock_script_hashes[i];
            let claim: Claim = Claim {
                program_digest: lock_script_hash,
                input: txk_mast_hash_as_input.clone(),
                output: Vec::<BFieldElement>::new(),
            };
            let lock_script_halts_proof: &Proof = &pc.lock_scripts_halt[i];
            let lock_script_halts: bool =
                tasmlib::verify(Stark::default(), claim, lock_script_halts_proof);
            assert!(lock_script_halts);

            i += 1;
        }

        let type_script_input: Vec<BFieldElement> = [
            txk_mast_hash_as_input,
            salted_inputs_hash_as_input,
            salted_outputs_hash_as_input,
        ]
        .concat();
        let mut i = 0;
        while i < pc.type_script_hashes.len() {
            let type_script_hash = pc.type_script_hashes[i];
            let claim: Claim = Claim {
                program_digest: type_script_hash,
                input: type_script_input.clone(),
                output: Vec::<BFieldElement>::new(),
            };
            let type_script_halts_proof: &Proof = &pc.type_scripts_halt[i];
            let type_script_halts: bool =
                tasmlib::verify(Stark::default(), claim, type_script_halts_proof);
            assert!(type_script_halts);
            i += 1;
        }
    }

    fn code(&self) -> Vec<LabelledInstruction> {
        todo!()
    }
}

#[cfg(test)]
pub mod test {
    use super::*;
    use proptest::prop_assert;
    use proptest::{arbitrary::Arbitrary, strategy::Strategy, test_runner::TestRunner};
    use test_strategy::proptest;

    #[proptest(cases = 5)]
    fn can_produce_valid_collection(
        #[strategy(PrimitiveWitness::arbitrary_with((2, 2, 2)))]
        primitive_witness: PrimitiveWitness,
    ) {
        prop_assert!(ProofCollection::can_produce(&primitive_witness));
    }

    #[test]
    fn can_verify_valid_collection() {
        let mut test_runner = TestRunner::deterministic();
        let primitive_witness = PrimitiveWitness::arbitrary_with((2, 2, 2))
            .new_tree(&mut test_runner)
            .unwrap()
            .current();
        let proof_collection = ProofCollection::produce(&primitive_witness);
        let txk_mast_hash = primitive_witness.kernel.mast_hash();
        assert!(proof_collection.verify(txk_mast_hash));
        let txk_mast_hash_as_input_as_public_input =
            PublicInput::new(txk_mast_hash.reversed().values().encode());
        proof_collection
            .run_rust(
                &txk_mast_hash_as_input_as_public_input,
                proof_collection.nondeterminism(),
            )
            .expect("rust run failed");
    }

    #[test]
    fn const_subprogram_hashes_are_correct() {
        let print_hash_nicely = |h: Digest| {
            format!(
                "Digest::new([{}])",
                h.values()
                    .iter()
                    .map(|bfe| format!("BFieldElement::new({})", bfe.value()))
                    .join(", ")
            )
        };
        assert_eq!(
            ProofCollection::REMOVAL_RECORDS_INTEGRITY_PROGRAM_DIGEST,
            RemovalRecordsIntegrity.program().hash::<Hash>(),
            "Removal Records Integrity: {}",
            print_hash_nicely(RemovalRecordsIntegrity.program().hash::<Hash>())
        );

        assert_eq!(
            ProofCollection::KERNEL_TO_OUTPUTS_PROGRAM_DIGEST,
            KernelToOutputs.program().hash::<Hash>(),
            "Kernel To Outputs: {}",
            print_hash_nicely(KernelToOutputs.program().hash::<Hash>())
        );

        assert_eq!(
            ProofCollection::COLLECT_LOCK_SCRIPTS_PROGRAM_DIGEST,
            CollectLockScripts.program().hash::<Hash>(),
            "Collect Lock Scripts: {}",
            print_hash_nicely(CollectLockScripts.program().hash::<Hash>())
        );

        assert_eq!(
            ProofCollection::COLLECT_TYPE_SCRIPTS_PROGRAM_DIGEST,
            CollectTypeScripts.program().hash::<Hash>(),
            "Collect Type Scripts: {}",
            print_hash_nicely(CollectTypeScripts.program().hash::<Hash>())
        );
    }
}
