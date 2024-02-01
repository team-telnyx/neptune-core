use tasm_lib::twenty_first::util_types::algebraic_hasher::AlgebraicHasher;
use tasm_lib::twenty_first::util_types::mmr::mmr_accumulator::MmrAccumulator;
use tasm_lib::twenty_first::util_types::mmr::mmr_trait::Mmr;

use crate::util_types::mutator_set::chunk::Chunk;
use crate::util_types::mutator_set::mutator_set_trait::MutatorSet;
use crate::util_types::mutator_set::shared::{CHUNK_SIZE, WINDOW_SIZE};
use crate::Hash;
use crate::{
    models::blockchain::block::Block, util_types::mutator_set::chunk_dictionary::ChunkDictionary,
};

/// LightState is just a thread-safe Block.
/// (always representing the latest block)
#[derive(Debug, Clone)]
pub struct LightState {
    pub block: Block,
    pub mutator_set_active_window: ChunkDictionary<Hash>,
}

impl LightState {
    pub fn before_genesis() -> Self {
        let mut block_placeholder = Block::empty_block();
        let empty_chunk = Chunk::empty_chunk();
        let empty_digest = Hash::hash(&empty_chunk);
        let num_chunks_in_active_window = WINDOW_SIZE / CHUNK_SIZE as usize;
        let mut swbf = MmrAccumulator::<Hash>::new(vec![]);
        let mut active_window_chunks = ChunkDictionary::<Hash>::empty_active_window_chunks();
        for _ in 0..num_chunks_in_active_window {
            for (_index, (mmr_mp, _chunk)) in active_window_chunks.dictionary.iter_mut() {
                mmr_mp.update_from_append(swbf.count_leaves(), empty_digest, &swbf.get_peaks());
            }
            let mmr_mp = swbf.append(empty_digest);
            active_window_chunks
                .dictionary
                .insert(mmr_mp.leaf_index, (mmr_mp, empty_chunk.clone()));
        }

        block_placeholder
            .kernel
            .body
            .mutator_set_accumulator
            .kernel
            .swbf_inactive = swbf;

        Self {
            block: block_placeholder,
            mutator_set_active_window: active_window_chunks,
        }
    }

    /// Produces the light state after the genesis block.
    pub fn after_genesis() -> Self {
        let genesis_block = Block::genesis_block();
        let mut light_state = Self::before_genesis();
        light_state.update_with_block(&genesis_block);
        light_state
    }

    /// Update the light state with the given block. Assumes the block is valid and an
    /// immediate successor.
    pub fn update_with_block(&mut self, block: &Block) {
        let mut msa = self.block.body().mutator_set_accumulator.clone();

        for ar in block.kernel.body.transaction.kernel.outputs.iter() {
            if let Some((new_index, (new_mmr_mp, new_chunk))) = msa.add(ar) {
                let new_chunk_digest = Hash::hash(&new_chunk);
                for (_index, (mmr_mp, _chunk)) in
                    self.mutator_set_active_window.dictionary.iter_mut()
                {
                    mmr_mp.update_from_append(
                        msa.kernel.swbf_inactive.count_leaves(),
                        new_chunk_digest,
                        &msa.kernel.swbf_inactive.get_peaks(),
                    );
                }
                msa.kernel.swbf_inactive.append(new_chunk_digest);
                self.mutator_set_active_window
                    .dictionary
                    .insert(new_index, (new_mmr_mp, new_chunk));
            }
        }
    }
}
