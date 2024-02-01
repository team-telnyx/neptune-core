use crate::prelude::twenty_first;

use tasm_lib::twenty_first::util_types::mmr::mmr_membership_proof::MmrMembershipProof;
use twenty_first::shared_math::tip5::Digest;
use twenty_first::util_types::algebraic_hasher::AlgebraicHasher;

use super::addition_record::AdditionRecord;
use super::chunk::Chunk;
use super::chunk_dictionary::ChunkDictionary;
use super::ms_membership_proof::MsMembershipProof;
use super::removal_record::RemovalRecord;

/// Generates an addition record from an item and explicit random-
/// ness. The addition record is itself a commitment to the item.
pub fn commit<H: AlgebraicHasher>(
    item: Digest,
    sender_randomness: Digest,
    receiver_digest: Digest,
) -> AdditionRecord {
    let canonical_commitment = H::hash_pair(H::hash_pair(item, sender_randomness), receiver_digest);

    AdditionRecord::new(canonical_commitment)
}

pub trait MutatorSet<H: AlgebraicHasher> {
    /// Generates a membership proof that will be valid when the item
    /// is added to the mutator set.
    fn prove(
        &mut self,
        item: Digest,
        sender_randomness: Digest,
        receiver_preimage: Digest,
        active_window_chunks: &ChunkDictionary<H>,
    ) -> MsMembershipProof<H>;

    fn verify(&self, item: Digest, membership_proof: &MsMembershipProof<H>) -> bool;

    /// Generates a removal record with which to update the set commitment.
    fn drop(&self, item: Digest, membership_proof: &MsMembershipProof<H>) -> RemovalRecord<H>;

    /// Updates the set-commitment with an addition record. If the window slides, return
    /// the new chunk (along with index and MMR membership proof); otherwise return None.
    fn add(
        &mut self,
        addition_record: &AdditionRecord,
    ) -> Option<(u64, (MmrMembershipProof<H>, Chunk))>;

    /// Updates the mutator set so as to remove the item determined by
    /// its removal record.
    fn remove(&mut self, removal_record: &RemovalRecord<H>);

    /// batch_remove
    /// Apply multiple removal records, and update a list of membership proofs to
    /// be valid after the application of these removal records.
    fn batch_remove(
        &mut self,
        removal_records: Vec<RemovalRecord<H>>,
        preserved_membership_proofs: &mut [&mut MsMembershipProof<H>],
    );

    /// hash
    /// Return single hash digest that commits to the entire mutator set
    fn hash(&self) -> Digest;
}
