use anyhow::Result;
use clap::Parser;
use itertools::Itertools;
use tasm_lib::triton_vm;
use tasm_lib::triton_vm::prelude::BFieldCodec;
use tasm_lib::triton_vm::prelude::BFieldElement;
use tasm_lib::triton_vm::proof::Claim;
use tasm_lib::triton_vm::proof::Proof;
use tasm_lib::triton_vm::stark::Stark;
use tokio::fs::File;
use tokio::io::AsyncReadExt;

#[derive(Debug, Parser)]
#[clap(name = "triton-vm-verifier", about = "Verify a Triton VM proof")]
struct Args {
    claim: String,
    proof: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let Args { claim, proof } = Args::parse();

    let claim = try_load_claim(claim).await?;
    let proof = try_load_proof(proof).await?;

    println!("** Calling triton_vm::verify to verify proof ...");
    let tick = std::time::Instant::now();
    let verdict =
        // task::spawn_blocking(move ||
            triton_vm::verify(Stark::default(), &claim, &proof)
        // )
            // .await
            // .expect("should be able to verify proof in new tokio task")
            ;
    let tock = tick.elapsed();
    println!(
        "** Call to triton_vm::verify to verify proof completed in {tock:?}; verdict: {verdict}."
    );

    Ok(())
}

async fn try_load_claim(claim_file_name: String) -> Result<Claim> {
    let claim_as_bfes = try_load_bfes(claim_file_name).await?;
    let claim = *Claim::decode(&claim_as_bfes)?;
    Ok(claim)
}

async fn try_load_proof(proof_file_name: String) -> Result<Proof> {
    let proof_as_bfes = try_load_bfes(proof_file_name).await?;
    let proof = Proof(proof_as_bfes);
    Ok(proof)
}

async fn try_load_bfes(file_name: String) -> Result<Vec<BFieldElement>> {
    let mut file = File::open(file_name).await?;

    let mut contents = vec![];
    file.read_to_end(&mut contents).await?;

    Ok(contents
        .chunks(8)
        .map(|a| {
            a.try_into()
                .expect("cast of chunk of 8 bytes to length 8 array of bytes should succeed")
        })
        .map(|a: [u8; 8]| u64::from_be_bytes(a))
        .map(BFieldElement::new)
        .collect_vec())
}
