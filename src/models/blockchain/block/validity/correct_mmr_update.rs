use get_size::GetSize;
use serde::Deserialize;
use serde::Serialize;
use tasm_lib::triton_vm::program::NonDeterminism;
use tasm_lib::triton_vm::program::PublicInput;
use tasm_lib::twenty_first::math::bfield_codec::BFieldCodec;
use tasm_lib::twenty_first::util_types::mmr::mmr_accumulator::MmrAccumulator;

use crate::models::consensus::tasm::program::ConsensusProgram;
use crate::models::consensus::SecretWitness;
use crate::Hash;

#[derive(Debug, Clone, BFieldCodec, GetSize, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorrectMmrUpdateWitness {
    pub previous_mmr_accumulator: MmrAccumulator<Hash>,
}

impl SecretWitness for CorrectMmrUpdateWitness {
    fn nondeterminism(&self) -> NonDeterminism {
        todo!()
    }

    fn standard_input(&self) -> PublicInput {
        todo!()
    }

    fn program(&self) -> tasm_lib::prelude::triton_vm::program::Program {
        todo!()
    }
}

#[derive(Debug, Clone, BFieldCodec, GetSize, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorrectMmrUpdate {
    pub witness: CorrectMmrUpdateWitness,
}

impl ConsensusProgram for CorrectMmrUpdate {
    fn source(&self) {
        todo!()
    }

    fn code(&self) -> Vec<tasm_lib::prelude::triton_vm::prelude::LabelledInstruction> {
        todo!()
    }
}
