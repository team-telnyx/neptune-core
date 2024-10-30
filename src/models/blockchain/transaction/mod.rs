use crate::models::blockchain::block::mutator_set_update::MutatorSetUpdate;
use crate::models::blockchain::type_scripts::known_type_scripts::match_type_script_and_generate_witness;
use crate::models::peer::transfer_transaction::TransactionProofQuality;
use crate::models::proof_abstractions::mast_hash::MastHash;
use crate::models::proof_abstractions::tasm::program::ConsensusProgram;
use crate::models::proof_abstractions::tasm::program::TritonProverSync;
use crate::models::proof_abstractions::SecretWitness;
use crate::models::state::transaction_details::TransactionDetails;
use crate::models::state::tx_proving_capability::TxProvingCapability;
use crate::models::state::wallet::expected_utxo::ExpectedUtxo;
use crate::models::state::wallet::unlocked_utxo::UnlockedUtxo;
use crate::prelude::twenty_first;

pub mod lock_script;
pub mod primitive_witness;
pub mod transaction_kernel;
pub mod transaction_output;
pub mod utxo;
pub mod validity;

use std::hash::Hash as StdHash;
use std::hash::Hasher as StdHasher;

use anyhow::bail;
use anyhow::Result;
use arbitrary::Arbitrary;
use get_size::GetSize;
use itertools::Itertools;
use num_bigint::BigInt;
use num_rational::BigRational;
use primitive_witness::SaltedUtxos;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::Deserialize;
use serde::Serialize;
use tasm_lib::prelude::TasmObject;
use tasm_lib::triton_vm;
use tasm_lib::triton_vm::prelude::Tip5;
use tasm_lib::triton_vm::stark::Stark;
use tasm_lib::twenty_first::util_types::mmr::mmr_successor_proof::MmrSuccessorProof;
use tasm_lib::Digest;
use tokio::sync::TryLockError;
use tracing::debug;
use tracing::info;
use twenty_first::math::b_field_element::BFieldElement;
use twenty_first::math::bfield_codec::BFieldCodec;
use twenty_first::util_types::algebraic_hasher::AlgebraicHasher;
use utxo::Utxo;
use validity::merge::Merge;
use validity::merge::MergeWitness;
use validity::proof_collection::ProofCollection;
use validity::single_proof::SingleProof;
use validity::single_proof::SingleProofWitness;
use validity::update::Update;
use validity::update::UpdateWitness;

use self::primitive_witness::PrimitiveWitness;
use self::transaction_kernel::TransactionKernel;
use super::block::Block;
use super::shared::Hash;
use crate::triton_vm::proof::Claim;
use crate::triton_vm::proof::Proof;
use crate::util_types::mutator_set::addition_record::AdditionRecord;
use crate::util_types::mutator_set::ms_membership_proof::MsMembershipProof;
use crate::util_types::mutator_set::mutator_set_accumulator::MutatorSetAccumulator;
use crate::util_types::mutator_set::removal_record::RemovalRecord;

/// represents a utxo and secrets necessary for recipient to claim it.
///
/// these are built from one of:
///   onchain symmetric-key public announcements
///   onchain asymmetric-key public announcements
///   offchain expected-utxos
///
/// See [PublicAnnouncement], [UtxoNotification], [ExpectedUtxo]
#[derive(Clone, Debug)]
pub struct AnnouncedUtxo {
    pub addition_record: AdditionRecord,
    pub utxo: Utxo,
    pub sender_randomness: Digest,
    pub receiver_preimage: Digest,
}

impl From<&ExpectedUtxo> for AnnouncedUtxo {
    fn from(eu: &ExpectedUtxo) -> Self {
        Self {
            addition_record: eu.addition_record,
            utxo: eu.utxo.clone(),
            sender_randomness: eu.sender_randomness,
            receiver_preimage: eu.receiver_preimage,
        }
    }
}

/// represents arbitrary data that can be stored in a transaction on the public blockchain
///
/// initially these are used for transmitting encrypted secrets necessary
/// for a utxo recipient to identify and claim it.
///
/// See [Transaction], [UtxoNotification]
#[derive(
    Clone,
    Debug,
    Serialize,
    Deserialize,
    PartialEq,
    Eq,
    GetSize,
    BFieldCodec,
    Default,
    TasmObject,
    Arbitrary,
)]
pub struct PublicAnnouncement {
    pub message: Vec<BFieldElement>,
}

impl PublicAnnouncement {
    pub(crate) fn new(message: Vec<BFieldElement>) -> Self {
        Self { message }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, GetSize, BFieldCodec, Default)]
pub enum TransactionProof {
    #[default]
    Invalid,
    Witness(PrimitiveWitness),
    SingleProof(Proof),
    ProofCollection(ProofCollection),
}

impl TransactionProof {
    pub(crate) fn proof_quality(&self) -> Result<TransactionProofQuality> {
        match self {
            TransactionProof::Invalid => bail!("Invalid proof does not have a proof quality"),
            TransactionProof::Witness(_) => bail!("Primitive witness does not have a proof"),
            TransactionProof::ProofCollection(_) => Ok(TransactionProofQuality::ProofCollection),
            TransactionProof::SingleProof(_) => Ok(TransactionProofQuality::SingleProof),
        }
    }

    pub(crate) async fn verify(&self, kernel_mast_hash: Digest) -> bool {
        match self {
            TransactionProof::Invalid => false,
            TransactionProof::Witness(primitive_witness) => {
                primitive_witness.validate().await
                    && primitive_witness.kernel.mast_hash() == kernel_mast_hash
            }
            TransactionProof::SingleProof(single_proof) => {
                let claim = SingleProof::claim(kernel_mast_hash);
                triton_vm::verify(Stark::default(), &claim, single_proof)
            }
            TransactionProof::ProofCollection(proof_collection) => {
                proof_collection.verify(kernel_mast_hash)
            }
        }
    }
}

#[derive(Clone, Debug)]
pub enum TransactionProofError {
    CannotUpdateProofVariant,
    CannotUpdatePrimitiveWitness,
    CannotUpdateSingleProof,
    ProverLockWasTaken,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, GetSize, BFieldCodec)]
pub struct Transaction {
    pub kernel: TransactionKernel,

    #[bfield_codec(ignore)]
    pub proof: TransactionProof,
}

/// Make `Transaction` hashable with `StdHash` for using it in `HashMap`.
///
/// The Clippy warning is safe to suppress, because we do not violate the invariant: k1 == k2 => hash(k1) == hash(k2).
#[allow(clippy::derived_hash_with_manual_eq)]
impl StdHash for Transaction {
    fn hash<H: StdHasher>(&self, state: &mut H) {
        let neptune_hash = Hash::hash(self);
        StdHash::hash(&neptune_hash, state);
    }
}

impl Transaction {
    /// creates a Transaction.
    ///
    /// This API provides the caller complete control over selection of inputs
    /// and outputs.  When fine grained control is not required,
    /// [GlobalState::create_transaction()] is easier to use and should be preferred.
    ///
    /// It is the caller's responsibility to provide inputs and outputs such
    /// that sum(inputs) == sum(outputs) + fee.  Else an error will result.
    ///
    /// Note that this means the caller must calculate the `change` amount if any
    /// and provide an output for the change.
    ///
    /// The `tx_outputs` parameter should normally be generated with
    /// [GlobalState::generate_tx_outputs()] which determines which outputs should
    /// be notified `OnChain` or `OffChain`.
    ///
    /// After this call returns, it is the caller's responsibility to inform the
    /// wallet of any returned [ExpectedUtxo] for utxos that match wallet keys.
    /// Failure to do so can result in loss of funds!
    ///
    /// Note that `create_raw_transaction()` does not modify any state and does
    /// not require acquiring write lock.  This is important becauce internally
    /// it calls prove() which is a very lengthy operation.
    ///
    /// Example:
    ///
    /// See the implementation of [GlobalState::create_transaction()].
    pub(crate) async fn create_raw_transaction(
        transaction_details: TransactionDetails,
        proving_power: TxProvingCapability,
        sync_device: &TritonProverSync,
    ) -> Result<Transaction, TryLockError> {
        // note: this executes the prover which can take a very
        //       long time, perhaps minutes.  The `await` here, should avoid
        //       block the tokio executor and other async tasks.
        Self::create_transaction_from_data_worker(transaction_details, proving_power, sync_device)
            .await
    }

    // note: this executes the prover which can take a very
    //       long time, perhaps minutes. It should never be
    //       called directly.
    //       Use create_transaction_from_data() instead.
    //
    async fn create_transaction_from_data_worker(
        transaction_details: TransactionDetails,
        proving_power: TxProvingCapability,
        sync_device: &TritonProverSync,
    ) -> Result<Transaction, TryLockError> {
        let TransactionDetails {
            tx_inputs,
            tx_outputs,
            fee,
            coinbase,
            timestamp,
            mutator_set_accumulator,
        } = transaction_details;

        // complete transaction kernel
        let removal_records = tx_inputs
            .iter()
            .map(|txi| txi.removal_record(&mutator_set_accumulator))
            .collect_vec();
        let kernel = TransactionKernel {
            inputs: removal_records,
            outputs: tx_outputs.addition_records(),
            public_announcements: tx_outputs.public_announcements(),
            fee,
            timestamp,
            coinbase,
            mutator_set_hash: mutator_set_accumulator.hash(),
        };

        // populate witness
        let output_utxos = tx_outputs.utxos();
        let unlocked_utxos = tx_inputs;
        let sender_randomnesses = tx_outputs.sender_randomnesses();
        let receiver_digests = tx_outputs.receiver_digests();
        let primitive_witness = Self::generate_primitive_witness(
            unlocked_utxos,
            output_utxos,
            sender_randomnesses,
            receiver_digests,
            kernel.clone(),
            mutator_set_accumulator,
        );

        debug!("primitive witness for transaction: {}", primitive_witness);

        info!(
            "Start: generate proof for {}-in {}-out transaction",
            primitive_witness.input_utxos.utxos.len(),
            primitive_witness.output_utxos.utxos.len()
        );
        let proof = match proving_power {
            TxProvingCapability::PrimitiveWitness => TransactionProof::Witness(primitive_witness),
            TxProvingCapability::LockScript => todo!(),
            TxProvingCapability::ProofCollection => TransactionProof::ProofCollection(
                ProofCollection::produce(&primitive_witness, sync_device).await?,
            ),
            TxProvingCapability::SingleProof => TransactionProof::SingleProof(
                SingleProof::produce(&primitive_witness, sync_device).await?,
            ),
        };

        Ok(Transaction { kernel, proof })
    }

    /// Generate a primitive witness for a transaction from various disparate witness data.
    ///
    /// # Panics
    /// Panics if transaction validity cannot be satisfied.
    pub(crate) fn generate_primitive_witness(
        unlocked_utxos: Vec<UnlockedUtxo>,
        output_utxos: Vec<Utxo>,
        sender_randomnesses: Vec<Digest>,
        receiver_digests: Vec<Digest>,
        transaction_kernel: TransactionKernel,
        mutator_set_accumulator: MutatorSetAccumulator,
    ) -> PrimitiveWitness {
        /// Generate a salt to use for [SaltedUtxos], deterministically.
        fn generate_secure_pseudorandom_seed(
            input_utxos: &Vec<Utxo>,
            output_utxos: &Vec<Utxo>,
            sender_randomnesses: &Vec<Digest>,
        ) -> [u8; 32] {
            let preimage = [
                input_utxos.encode(),
                output_utxos.encode(),
                sender_randomnesses.encode(),
            ]
            .concat();
            let seed = Tip5::hash_varlen(&preimage);
            let seed: [u8; Digest::BYTES] = seed.into();

            seed[0..32].try_into().unwrap()
        }

        let input_utxos = unlocked_utxos
            .iter()
            .map(|unlocker| unlocker.utxo.to_owned())
            .collect_vec();
        let salt_seed =
            generate_secure_pseudorandom_seed(&input_utxos, &output_utxos, &sender_randomnesses);

        let mut rng: StdRng = SeedableRng::from_seed(salt_seed);
        let salted_output_utxos = SaltedUtxos::new_with_rng(output_utxos.to_vec(), &mut rng);
        let salted_input_utxos = SaltedUtxos::new_with_rng(input_utxos.clone(), &mut rng);

        let type_script_hashes = input_utxos
            .iter()
            .chain(output_utxos.iter())
            .flat_map(|utxo| utxo.coins.iter().map(|coin| coin.type_script_hash))
            .unique()
            .collect_vec();
        let type_scripts_and_witnesses = type_script_hashes
            .into_iter()
            .map(|type_script_hash| {
                match_type_script_and_generate_witness(
                    type_script_hash,
                    transaction_kernel.clone(),
                    salted_input_utxos.clone(),
                    salted_output_utxos.clone(),
                )
                .expect("type script hash should be known.")
            })
            .collect_vec();
        let input_lock_scripts_and_witnesses = unlocked_utxos
            .iter()
            .map(|unlocker| unlocker.lock_script_and_witness())
            .cloned()
            .collect_vec();
        let input_membership_proofs = unlocked_utxos
            .iter()
            .map(|unlocker| unlocker.mutator_set_mp().to_owned())
            .collect_vec();

        PrimitiveWitness {
            input_utxos: salted_input_utxos,
            lock_scripts_and_witnesses: input_lock_scripts_and_witnesses,
            type_scripts_and_witnesses,
            input_membership_proofs,
            output_utxos: salted_output_utxos,
            output_sender_randomnesses: sender_randomnesses.to_vec(),
            output_receiver_digests: receiver_digests.to_vec(),
            mutator_set_accumulator,
            kernel: transaction_kernel,
        }
    }

    /// Create a new transaction with primitive witness for a new mutator set.
    pub(crate) fn new_with_primitive_witness_ms_data(
        old_primitive_witness: PrimitiveWitness,
        new_addition_records: Vec<AdditionRecord>,
        mut new_removal_records: Vec<RemovalRecord>,
    ) -> (Transaction, MutatorSetAccumulator) {
        new_removal_records.reverse();
        let mut block_removal_records: Vec<&mut RemovalRecord> =
            new_removal_records.iter_mut().collect::<Vec<_>>();
        let mut msa_state: MutatorSetAccumulator =
            old_primitive_witness.mutator_set_accumulator.clone();
        let mut transaction_removal_records: Vec<RemovalRecord> =
            old_primitive_witness.kernel.inputs.clone();
        let mut transaction_removal_records: Vec<&mut RemovalRecord> =
            transaction_removal_records.iter_mut().collect();

        let mut primitive_witness = old_primitive_witness.clone();

        // Apply all addition records in the block
        for block_addition_record in new_addition_records {
            // Batch update block's removal records to keep them valid after next addition
            RemovalRecord::batch_update_from_addition(&mut block_removal_records, &msa_state);

            // Batch update transaction's removal records
            RemovalRecord::batch_update_from_addition(&mut transaction_removal_records, &msa_state);

            // Batch update primitive witness membership proofs
            let membership_proofs = &mut primitive_witness
                .input_membership_proofs
                .iter_mut()
                .collect_vec();
            let own_items = primitive_witness
                .input_utxos
                .utxos
                .iter()
                .map(Hash::hash)
                .collect_vec();
            MsMembershipProof::batch_update_from_addition(
                membership_proofs,
                &own_items,
                &msa_state,
                &block_addition_record,
            )
            .expect("MS MP update from add must succeed in wallet handler");

            msa_state.add(&block_addition_record);
        }

        while let Some(removal_record) = block_removal_records.pop() {
            // Batch update block's removal records to keep them valid after next removal
            RemovalRecord::batch_update_from_remove(&mut block_removal_records, removal_record);

            // batch update transaction's removal records
            // Batch update block's removal records to keep them valid after next removal
            RemovalRecord::batch_update_from_remove(
                &mut transaction_removal_records,
                removal_record,
            );

            // Batch update primitive witness membership proofs
            let membership_proofs = &mut primitive_witness
                .input_membership_proofs
                .iter_mut()
                .collect_vec();

            MsMembershipProof::batch_update_from_remove(membership_proofs, removal_record)
                .expect("MS MP update from add must succeed in wallet handler");

            msa_state.remove(removal_record);
        }

        primitive_witness.kernel.mutator_set_hash = msa_state.hash();
        primitive_witness.mutator_set_accumulator = msa_state.clone();
        primitive_witness.kernel.inputs = transaction_removal_records
            .into_iter()
            .map(|x| x.to_owned())
            .collect_vec();

        let kernel = primitive_witness.kernel.clone();
        let witness = TransactionProof::Witness(primitive_witness);

        (
            Transaction {
                kernel,
                proof: witness,
            },
            msa_state,
        )
    }

    /// Create a new `Transaction` from a `PrimitiveWitness` (which defines an old
    /// `Transaction`) by updating the mutator set records according to a new
    /// `Block`.
    pub(crate) fn new_with_updated_mutator_set_records_given_primitive_witness(
        old_primitive_witness: PrimitiveWitness,
        block: &Block,
    ) -> Transaction {
        let block_addition_records: Vec<AdditionRecord> =
            block.kernel.body.transaction_kernel.outputs.clone();
        let block_removal_records = block.kernel.body.transaction_kernel.inputs.clone();

        let (new_tx, new_msa) = Self::new_with_primitive_witness_ms_data(
            old_primitive_witness,
            block_addition_records,
            block_removal_records,
        );

        // Sanity check of block validity
        let block_msa_hash = block.kernel.body.mutator_set_accumulator.clone().hash();
        assert_eq!(
            new_msa.hash(),
            block_msa_hash,
            "Internal MSA state must match that from block"
        );

        new_tx
    }

    /// Create a new `Transaction` by updating the given one with the mutator set
    /// update contained in the `Block`. No primitive witness is present, instead
    /// a proof is given. So:
    ///  1. Verify the proof
    ///  2. Update the records
    ///  3. Prove correctness of 1 and 2
    ///  4. Use resulting proof as new witness.
    pub(crate) async fn new_with_updated_mutator_set_records_given_proof(
        old_transaction_kernel: TransactionKernel,
        previous_mutator_set_accumulator: &MutatorSetAccumulator,
        mutator_set_update: MutatorSetUpdate,
        old_single_proof: Proof,
        sync_device: &TritonProverSync,
    ) -> Result<Transaction, TryLockError> {
        // apply mutator set update to get new mutator set accumulator
        let addition_records = mutator_set_update.additions.clone();
        let mut calculated_new_mutator_set = previous_mutator_set_accumulator.clone();
        let mut new_inputs = old_transaction_kernel.inputs.clone();
        mutator_set_update
            .apply_to_accumulator_and_records(
                &mut calculated_new_mutator_set,
                &mut new_inputs.iter_mut().collect_vec(),
            )
            .unwrap_or_else(|_| panic!("Could not apply mutator set update."));

        let aocl_successor_proof = MmrSuccessorProof::new_from_batch_append(
            &previous_mutator_set_accumulator.aocl,
            &addition_records
                .iter()
                .map(|addition_record| addition_record.canonical_commitment)
                .collect_vec(),
        );

        // compute new kernel
        let mut new_kernel = old_transaction_kernel.clone();
        new_kernel.inputs = new_inputs;
        new_kernel.mutator_set_hash = calculated_new_mutator_set.hash();

        // compute updated proof through recursion
        let update_witness = UpdateWitness::from_old_transaction(
            old_transaction_kernel.clone(),
            old_single_proof.clone(),
            previous_mutator_set_accumulator.clone(),
            new_kernel.clone(),
            calculated_new_mutator_set,
            aocl_successor_proof,
        );
        let update_claim = update_witness.claim();
        let update_nondeterminism = update_witness.nondeterminism();
        info!("updating transaction; starting update proof ...");
        let update_proof = Update
            .prove(&update_claim, update_nondeterminism, sync_device)
            .await?;
        info!("done.");

        let new_single_proof_witness = SingleProofWitness::from_update(update_proof, &new_kernel);
        let new_single_proof_claim = new_single_proof_witness.claim();

        info!("starting single proof via update ...");
        let new_single_proof = SingleProof
            .prove(
                &new_single_proof_claim,
                new_single_proof_witness.nondeterminism(),
                sync_device,
            )
            .await?;
        info!("done.");

        Ok(Transaction {
            kernel: new_kernel,
            proof: TransactionProof::SingleProof(new_single_proof),
        })
    }

    /// Update mutator set data in a transaction to update its
    /// compatibility with a new block. Note that for Proof witnesses, this will
    /// invalidate the proof, requiring an update.
    pub(crate) async fn new_with_updated_mutator_set_records(
        self,
        previous_mutator_set_accumulator: &MutatorSetAccumulator,
        block: &Block,
        sync_device: &TritonProverSync,
    ) -> Result<Transaction, TransactionProofError> {
        match self.proof {
            TransactionProof::Witness(primitive_witness) => Ok(
                Self::new_with_updated_mutator_set_records_given_primitive_witness(
                    primitive_witness,
                    block,
                ),
            ),
            TransactionProof::SingleProof(proof) => {
                let block_body = block.body();
                let tx_kernel = block_body.transaction_kernel.clone();
                let ms_update = MutatorSetUpdate::new(tx_kernel.inputs, tx_kernel.outputs);
                Self::new_with_updated_mutator_set_records_given_proof(
                    self.kernel,
                    previous_mutator_set_accumulator,
                    ms_update,
                    proof,
                    sync_device,
                )
                .await
                .map_err(|_| TransactionProofError::ProverLockWasTaken)
            }
            _ => Err(TransactionProofError::CannotUpdateProofVariant),
        }
    }

    /// Determine whether the transaction is valid (forget about confirmable).
    /// This method tests the transaction's internal consistency in isolation,
    /// without the context of the canonical chain.
    pub(crate) async fn is_valid(&self) -> bool {
        let kernel_hash = self.kernel.mast_hash();
        self.proof.verify(kernel_hash).await
    }

    /// Merge two transactions. Both input transactions must have a valid
    /// Proof witness for this operation to work.
    ///
    /// # Panics
    ///
    /// Panics if the two transactions cannot be merged, if e.g. the mutator
    /// set hashes are not the same, if both transactions have coinbase a
    /// coinbase UTXO, or if either of the transactions are *not* a single
    /// proof.
    pub(crate) async fn merge_with(
        self,
        other: Transaction,
        shuffle_seed: [u8; 32],
        sync_device: &TritonProverSync,
    ) -> Result<Transaction, TryLockError> {
        assert_eq!(
            self.kernel.mutator_set_hash, other.kernel.mutator_set_hash,
            "Mutator sets must be equal for transaction merger."
        );

        assert!(
            self.kernel.coinbase.is_none() || other.kernel.coinbase.is_none(),
            "Cannot merge two coinbase transactions"
        );

        let as_single_proof = |tx_proof: &TransactionProof, indicator: &str| {
            if let TransactionProof::SingleProof(single_proof) = tx_proof {
                single_proof.to_owned()
            } else {
                let bad_type = match tx_proof {
                    TransactionProof::Invalid => "invalid",
                    TransactionProof::Witness(_primitive_witness) => "primitive_witness",
                    TransactionProof::SingleProof(_proof) => unreachable!(),
                    TransactionProof::ProofCollection(_proof_collection) => "proof_collection",
                };
                panic!("Transaction proof must be a single proof. {indicator} was: {bad_type}",);
            }
        };
        let self_single_proof = as_single_proof(&self.proof, "self");
        let other_single_proof = as_single_proof(&other.proof, "other");

        let merge_witness = MergeWitness::from_transactions(
            self.kernel,
            self_single_proof,
            other.kernel,
            other_single_proof,
            shuffle_seed,
        );
        info!("Start: creating merge proof");
        let merge_claim = merge_witness.claim();
        let merge_proof = Merge
            .prove(&merge_claim, merge_witness.nondeterminism(), sync_device)
            .await?;
        info!("Done: creating merge proof");
        let new_single_proof_witness =
            SingleProofWitness::from_merge(merge_proof, &merge_witness.new_kernel);
        let new_single_proof_claim = new_single_proof_witness.claim();
        info!("Start: creating new single proof");
        let new_single_proof = SingleProof
            .prove(
                &new_single_proof_claim,
                new_single_proof_witness.nondeterminism(),
                sync_device,
            )
            .await?;
        info!("Done: creating new single proof");

        Ok(Transaction {
            kernel: merge_witness.new_kernel,
            proof: TransactionProof::SingleProof(new_single_proof),
        })
    }

    /// Calculates a fraction representing the fee-density, defined as:
    /// `transaction_fee/transaction_size`.
    pub(crate) fn fee_density(&self) -> BigRational {
        let transaction_as_bytes = bincode::serialize(&self).unwrap();
        let transaction_size = BigInt::from(transaction_as_bytes.get_size());
        let transaction_fee = self.kernel.fee.to_nau();
        BigRational::new_raw(transaction_fee, transaction_size)
    }

    /// Determine if the transaction can be validly confirmed if the block has
    /// the given mutator set accumulator. Specifically, test whether the
    /// removal records determine indices absent in the mutator set sliding
    /// window Bloom filter, and whether the MMR membership proofs are valid.
    ///
    /// Why not testing AOCL MMR membership proofs? These are being verified in
    /// PrimitiveWitness::validate and ProofCollection/RemovalRecordsIntegrity.
    /// AOCL membership is a feature of *validity*, which is a pre-requisite to
    /// confirmability.
    pub(crate) fn is_confirmable_relative_to(
        &self,
        mutator_set_accumulator: &MutatorSetAccumulator,
    ) -> bool {
        self.kernel
            .inputs
            .iter()
            .all(|rr| rr.validate(mutator_set_accumulator))
    }
}

#[cfg(test)]
mod tests {
    use tasm_lib::Digest;
    use tests::primitive_witness::SaltedUtxos;

    use super::*;
    use crate::models::blockchain::type_scripts::neptune_coins::NeptuneCoins;

    #[test]
    fn decode_encode_test_empty() {
        let empty_kernel = TransactionKernel {
            inputs: vec![],
            outputs: vec![],
            public_announcements: vec![],
            fee: NeptuneCoins::new(0),
            coinbase: None,
            timestamp: Default::default(),
            mutator_set_hash: Digest::default(),
        };
        let primitive_witness = PrimitiveWitness {
            input_utxos: SaltedUtxos::empty(),
            type_scripts_and_witnesses: vec![],
            lock_scripts_and_witnesses: vec![],
            input_membership_proofs: vec![],
            output_utxos: SaltedUtxos::empty(),
            output_sender_randomnesses: vec![],
            output_receiver_digests: vec![],
            mutator_set_accumulator: MutatorSetAccumulator::default(),
            kernel: empty_kernel,
        };

        let encoded = primitive_witness.encode();
        let decoded = *PrimitiveWitness::decode(&encoded).unwrap();
        assert_eq!(primitive_witness, decoded);
    }
}

#[cfg(test)]
mod transaction_tests {
    use lock_script::LockScript;
    use proptest::prelude::Strategy;
    use proptest::test_runner::TestRunner;
    use rand::random;
    use tracing_test::traced_test;
    use transaction_tests::utxo::Utxo;

    use super::*;
    use crate::config_models::network::Network;
    use crate::models::blockchain::type_scripts::neptune_coins::NeptuneCoins;
    use crate::models::proof_abstractions::timestamp::Timestamp;
    use crate::tests::shared::make_mock_transaction;
    use crate::tests::shared::mock_block_from_transaction_and_msa;
    use crate::util_types::mutator_set::commit;

    #[traced_test]
    #[test]
    fn tx_get_timestamp_test() {
        let output_1 = Utxo {
            coins: NeptuneCoins::new(42).to_native_coins(),
            lock_script_hash: LockScript::anyone_can_spend().hash(),
        };
        let ar = commit(Hash::hash(&output_1), random(), random());

        // Verify that a sane timestamp is returned. `make_mock_transaction` must follow
        // the correct time convention for this test to work.
        let coinbase_transaction = make_mock_transaction(vec![], vec![ar]);
        assert!(Timestamp::now() - coinbase_transaction.kernel.timestamp < Timestamp::seconds(10));
    }

    #[test]
    fn encode_decode_empty_tx_test() {
        let empty_tx = make_mock_transaction(vec![], vec![]);
        let encoded = empty_tx.encode();
        let decoded = *Transaction::decode(&encoded).unwrap();
        assert_eq!(empty_tx, decoded);
    }

    // `traced_test` macro inserts return type that clippy doesn't like.
    // Macro is at fault.
    #[traced_test]
    #[tokio::test]
    #[allow(clippy::needless_return)]
    async fn update_single_proof_works() {
        async fn prop(to_be_updated: PrimitiveWitness, mined: PrimitiveWitness) {
            let as_single_proof = SingleProof::produce(&to_be_updated, &TritonProverSync::dummy())
                .await
                .unwrap();
            let original_tx = Transaction {
                kernel: to_be_updated.kernel,
                proof: TransactionProof::SingleProof(as_single_proof),
            };
            assert!(original_tx.is_valid().await);

            let block = mock_block_from_transaction_and_msa(
                mined.kernel,
                mined.mutator_set_accumulator,
                Network::Main,
            );
            let updated_tx = original_tx
                .new_with_updated_mutator_set_records(
                    &to_be_updated.mutator_set_accumulator,
                    &block,
                    &TritonProverSync::dummy(),
                )
                .await
                .unwrap();

            assert!(updated_tx.is_valid().await)
        }

        for (to_be_updated_params, mined_params) in [
            ((4, 4, 4), (3, 3, 3)),
            ((0, 1, 0), (1, 1, 0)),
            ((1, 1, 0), (0, 1, 0)),
            ((0, 2, 1), (1, 1, 1)),
            ((2, 2, 2), (2, 2, 2)),
        ] {
            println!("to_be_updated_params: {to_be_updated_params:?}");
            println!("mined_params: {mined_params:?}");
            let mut test_runner = TestRunner::deterministic();
            let [to_be_updated, mined] =
                PrimitiveWitness::arbitrary_tuple_with_matching_mutator_sets([
                    to_be_updated_params,
                    mined_params,
                ])
                .new_tree(&mut test_runner)
                .unwrap()
                .current();
            prop(to_be_updated, mined).await;
        }
    }

    #[traced_test]
    #[tokio::test]
    async fn primitive_witness_updaters_are_equivalent() {
        // Verify that various ways of updating a primitive witness are
        // equivalent, and that they all yield valid primitive witnesses.
        fn update_with_block(
            to_be_updated: PrimitiveWitness,
            mined: PrimitiveWitness,
        ) -> Transaction {
            let block = mock_block_from_transaction_and_msa(
                mined.kernel,
                mined.mutator_set_accumulator,
                Network::Main,
            );

            Transaction::new_with_updated_mutator_set_records_given_primitive_witness(
                to_be_updated.clone(),
                &block,
            )
        }

        fn update_with_ms_data(
            to_be_updated: PrimitiveWitness,
            mined: PrimitiveWitness,
        ) -> Transaction {
            let (updated, _msa_new) = Transaction::new_with_primitive_witness_ms_data(
                to_be_updated,
                mined.kernel.outputs,
                mined.kernel.inputs,
            );

            updated
        }

        async fn assert_valid_as_pw(transaction: &Transaction) {
            let TransactionProof::Witness(pw) = &transaction.proof else {
                panic!("Expected primitive witness variant");
            };
            assert!(pw.validate().await)
        }

        let mut test_runner = TestRunner::deterministic();
        let [to_be_updated, mined] =
            PrimitiveWitness::arbitrary_tuple_with_matching_mutator_sets([(4, 4, 4), (3, 3, 3)])
                .new_tree(&mut test_runner)
                .unwrap()
                .current();
        assert!(to_be_updated.validate().await);
        assert!(mined.validate().await);

        let updated_with_block = update_with_block(to_be_updated.clone(), mined.clone());
        assert_valid_as_pw(&updated_with_block).await;

        let updated_with_ms_data = update_with_ms_data(to_be_updated.clone(), mined.clone());
        assert_valid_as_pw(&updated_with_ms_data).await;

        assert_eq!(updated_with_block, updated_with_ms_data);

        assert_eq!(
            to_be_updated.kernel.coinbase,
            updated_with_ms_data.kernel.coinbase
        );
        assert_eq!(to_be_updated.kernel.fee, updated_with_ms_data.kernel.fee);
        assert_eq!(
            to_be_updated.kernel.outputs,
            updated_with_ms_data.kernel.outputs
        );
        assert_eq!(
            to_be_updated.kernel.public_announcements,
            updated_with_ms_data.kernel.public_announcements
        );
        assert_ne!(
            to_be_updated.kernel.mutator_set_hash,
            updated_with_ms_data.kernel.mutator_set_hash
        );
    }
}
