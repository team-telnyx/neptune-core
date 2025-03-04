pub mod archival_state;
pub mod block_proposal;
pub mod blockchain_state;
pub mod light_state;
pub mod mempool;
pub mod mining_status;
pub mod networking_state;
pub mod shared;
pub(crate) mod transaction_details;
pub(crate) mod transaction_kernel_id;
pub(crate) mod tx_initiation_config;
pub mod tx_proving_capability;
pub mod wallet;

use std::cmp::max;
use std::collections::HashMap;
use std::collections::HashSet;
use std::net::SocketAddr;
use std::ops::Deref;
use std::ops::DerefMut;
use std::time::SystemTime;

use anyhow::bail;
use anyhow::Result;
use block_proposal::BlockProposal;
use blockchain_state::BlockchainState;
use mempool::Mempool;
use mempool::TransactionOrigin;
use mining_status::ComposingWorkInfo;
use mining_status::GuessingWorkInfo;
use mining_status::MiningStatus;
use networking_state::NetworkingState;
use num_traits::CheckedSub;
use num_traits::Zero;
use tasm_lib::triton_vm::prelude::*;
use tracing::debug;
use tracing::info;
use tracing::warn;
use transaction_details::TransactionDetails;
use twenty_first::math::digest::Digest;
use tx_proving_capability::TxProvingCapability;
use wallet::address::ReceivingAddress;
use wallet::address::SpendingKey;
use wallet::wallet_state::WalletState;
use wallet::wallet_status::WalletStatus;

use super::blockchain::block::block_header::BlockHeader;
use super::blockchain::block::block_height::BlockHeight;
use super::blockchain::block::difficulty_control::ProofOfWork;
use super::blockchain::block::Block;
use super::blockchain::transaction::primitive_witness::PrimitiveWitness;
use super::blockchain::transaction::Transaction;
use super::blockchain::type_scripts::native_currency_amount::NativeCurrencyAmount;
use super::peer::handshake_data::HandshakeData;
use super::peer::handshake_data::VersionString;
use super::peer::transfer_block::TransferBlock;
use super::peer::SyncChallenge;
use super::peer::SyncChallengeResponse;
use super::proof_abstractions::timestamp::Timestamp;
use crate::config_models::cli_args;
use crate::database::storage::storage_schema::traits::StorageWriter as SW;
use crate::database::storage::storage_vec::traits::*;
use crate::database::storage::storage_vec::Index;
use crate::job_queue::triton_vm::TritonVmJobPriority;
use crate::job_queue::triton_vm::TritonVmJobQueue;
use crate::locks::tokio as sync_tokio;
use crate::main_loop::proof_upgrader::UpdateMutatorSetDataJob;
use crate::mine_loop::composer_parameters::ComposerParameters;
use crate::models::blockchain::block::block_header::BlockHeaderWithBlockHashWitness;
use crate::models::blockchain::block::mutator_set_update::MutatorSetUpdate;
use crate::models::blockchain::transaction::validity::proof_collection::ProofCollection;
use crate::models::blockchain::transaction::validity::single_proof::SingleProof;
use crate::models::blockchain::transaction::TransactionProof;
use crate::models::peer::SYNC_CHALLENGE_POW_WITNESS_LENGTH;
use crate::models::proof_abstractions::tasm::program::TritonVmProofJobOptions;
use crate::models::state::block_proposal::BlockProposalRejectError;
use crate::models::state::wallet::expected_utxo::ExpectedUtxo;
use crate::models::state::wallet::monitored_utxo::MonitoredUtxo;
use crate::models::state::wallet::transaction_output::TxOutput;
use crate::models::state::wallet::transaction_output::TxOutputList;
use crate::models::state::wallet::utxo_notification::UtxoNotificationMedium;
use crate::prelude::twenty_first;
use crate::time_fn_call_async;
use crate::util_types::mutator_set::addition_record::AdditionRecord;
use crate::util_types::mutator_set::mutator_set_accumulator::MutatorSetAccumulator;
use crate::Hash;
use crate::VERSION;

/// `GlobalStateLock` holds a [`tokio::AtomicRw`](crate::locks::tokio::AtomicRw)
/// ([`RwLock`](tokio::sync::RwLock)) over [`GlobalState`].
///
/// Conceptually** all reads and writes of application state
/// require acquiring this lock.
///
/// Having a single lock is useful for a few reasons:
///  1. Enables write serialization over all application state.
///     (blockchain, mempool, wallet, global flags)
///  2. Readers see a consistent view of data.
///  3. makes it easy to reason about locking.
///  4. simplifies the codebase.
///
/// The primary drawback is that long write operations can
/// block readers.  As such, every effort should be made to keep
/// write operations as short as possible, though
/// correctness/atomicity have first priority.
///
/// Using an `RwLock` is beneficial for concurrency vs using a `Mutex`.
/// Readers do not block eachother.  Only a writer blocks readers.
/// See [`RwLock`](std::sync::RwLock) docs for details.
///
/// ** unless some type uses interior mutability.  We have made
/// efforts to eradicate interior mutability in this crate.
///
/// Usage conventions:
///
/// ```text
///
/// // property naming:
/// struct Foo {
///     global_state_lock: GlobalStateLock
/// }
///
/// // read guard naming:
/// let global_state = foo.global_state_lock.lock_guard().await;
///
/// // write guard naming:
/// let global_state_mut = foo.global_state_lock.lock_guard_mut().await;
/// ```
///
/// These conventions make it easy to distinguish read access from write
/// access when reading and reviewing code.
///
/// When using a read-guard or write-guard, always drop it as soon as possible.
/// Failure to do so can result in poor concurrency or deadlock.
///
/// Deadlocks are generally not hard to track down.  Lock events are traced.
/// The app log records each `TryAcquire`, `Acquire` and `Release` event
/// when run with `RUST_LOG='info,neptune_cash=trace'`.
///
/// If a deadlock has occurred, the log will end with a `TryAcquire` event
/// (read or write) and just scroll up to find the previous `Acquire` for
/// write event to see which thread is holding the lock.
#[derive(Debug, Clone)]
pub struct GlobalStateLock {
    global_state_lock: sync_tokio::AtomicRw<GlobalState>,

    /// The `cli_args::Args` are read-only and accessible by all tasks/threads.
    cli: cli_args::Args,

    vm_job_queue: TritonVmJobQueue,
}

impl GlobalStateLock {
    pub fn new(
        wallet_state: WalletState,
        chain: BlockchainState,
        net: NetworkingState,
        cli: cli_args::Args,
        mempool: Mempool,
    ) -> Self {
        let global_state = GlobalState::new(wallet_state, chain, net, cli.clone(), mempool);
        let global_state_lock = sync_tokio::AtomicRw::from((
            global_state,
            Some("GlobalState"),
            Some(crate::LOG_TOKIO_LOCK_EVENT_CB),
        ));
        Self {
            global_state_lock,
            cli,
            vm_job_queue: TritonVmJobQueue::start(),
        }
    }

    /// returns reference-counted clone of the triton vm job queue.
    ///
    /// callers should execute resource intensive triton-vm tasks in this
    /// queue to avoid running simultaneous tasks that could exceed hardware
    /// capabilities.
    pub(crate) fn vm_job_queue(&self) -> &TritonVmJobQueue {
        &self.vm_job_queue
    }

    // check if mining
    pub async fn mining(&self) -> bool {
        self.lock(|s| match s.mining_status {
            MiningStatus::Guessing(_) => true,
            MiningStatus::Composing(_) => true,
            MiningStatus::Inactive => false,
        })
        .await
    }

    pub async fn set_mining_status_to_inactive(&mut self) {
        self.lock_guard_mut().await.mining_status = MiningStatus::Inactive;
        tracing::debug!("set mining status: inactive");
    }

    /// Indicate if we are guessing
    pub async fn set_mining_status_to_guessing(&mut self, block: &Block) {
        let now = SystemTime::now();
        let block_info = GuessingWorkInfo::new(now, block);
        self.lock_guard_mut().await.mining_status = MiningStatus::Guessing(block_info);
        tracing::debug!("set mining status: guessing");
    }

    /// Indicate if we are composing
    pub async fn set_mining_status_to_composing(&mut self) {
        let now = SystemTime::now();
        let work_info = ComposingWorkInfo::new(now);
        self.lock_guard_mut().await.mining_status = MiningStatus::Composing(work_info);
        tracing::debug!("set mining status: composing");
    }

    // persist wallet state to disk
    pub async fn persist_wallet(&mut self) -> Result<()> {
        self.lock_guard_mut().await.persist_wallet().await
    }

    // flush databases (persist to disk)
    pub async fn flush_databases(&mut self) -> Result<()> {
        self.lock_guard_mut().await.flush_databases().await
    }

    /// Set tip to a block that we composed.
    pub async fn set_new_self_composed_tip(
        &mut self,
        new_block: Block,
        composer_reward_utxo_infos: Vec<ExpectedUtxo>,
    ) -> Result<Vec<UpdateMutatorSetDataJob>> {
        let mut state = self.lock_guard_mut().await;
        state
            .wallet_state
            .add_expected_utxos(composer_reward_utxo_infos)
            .await;
        state.set_new_tip(new_block).await
    }

    /// store a block (non coinbase)
    pub async fn set_new_tip(&mut self, new_block: Block) -> Result<Vec<UpdateMutatorSetDataJob>> {
        self.lock_guard_mut().await.set_new_tip(new_block).await
    }

    /// resync membership proofs
    pub async fn resync_membership_proofs(&mut self) -> Result<()> {
        self.lock_guard_mut().await.resync_membership_proofs().await
    }

    pub async fn prune_abandoned_monitored_utxos(
        &mut self,
        block_depth_threshold: usize,
    ) -> Result<usize> {
        self.lock_guard_mut()
            .await
            .prune_abandoned_monitored_utxos(block_depth_threshold)
            .await
    }

    /// Return the read-only arguments set at startup.
    #[inline]
    pub fn cli(&self) -> &cli_args::Args {
        &self.cli
    }

    /// Test helper function for fine control of CLI parameters.
    #[cfg(test)]
    pub async fn set_cli(&mut self, cli: cli_args::Args) {
        self.lock_guard_mut().await.cli = cli.clone();
        self.cli = cli;
    }
}

impl Deref for GlobalStateLock {
    type Target = sync_tokio::AtomicRw<GlobalState>;

    fn deref(&self) -> &Self::Target {
        &self.global_state_lock
    }
}

impl DerefMut for GlobalStateLock {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.global_state_lock
    }
}

/// `GlobalState` handles all state of a Neptune node that is shared across its tasks.
///
/// Some fields are only written to by certain tasks.
#[derive(Debug)]
pub struct GlobalState {
    /// The `WalletState` may be updated by the main task and the RPC server.
    pub wallet_state: WalletState,

    /// The `BlockchainState` may only be updated by the main task.
    pub chain: BlockchainState,

    /// The `NetworkingState` may be updated by both the main task and peer tasks.
    pub net: NetworkingState,

    /// The `cli_args::Args` are read-only and accessible by all tasks.
    cli: cli_args::Args,

    /// The `Mempool` may only be updated by the main task.
    pub mempool: Mempool,

    /// The block proposal to which guessers contribute proof-of-work.
    pub(crate) block_proposal: BlockProposal,

    /// Indicates whether the guessing or composing task is running, and if so,
    /// since when.
    // Only the mining task should write to this, anyone can read.
    pub(crate) mining_status: MiningStatus,
}

impl GlobalState {
    pub fn new(
        wallet_state: WalletState,
        chain: BlockchainState,
        net: NetworkingState,
        cli: cli_args::Args,
        mempool: Mempool,
    ) -> Self {
        Self {
            wallet_state,
            chain,
            net,
            cli,
            mempool,
            block_proposal: BlockProposal::default(),
            mining_status: MiningStatus::Inactive,
        }
    }

    /// Return a seed used to randomize shuffling.
    pub(crate) fn shuffle_seed(&self) -> [u8; 32] {
        let next_block_height = self.chain.light_state().header().height.next();
        self.wallet_state
            .wallet_secret
            .shuffle_seed(next_block_height)
    }

    pub async fn get_wallet_status_for_tip(&self) -> WalletStatus {
        let tip_digest = self.chain.light_state().hash();
        let mutator_set_accumulator = self.chain.light_state().mutator_set_accumulator_after();
        self.wallet_state
            .get_wallet_status(tip_digest, &mutator_set_accumulator)
            .await
    }

    pub async fn get_latest_balance_height(&self) -> Option<BlockHeight> {
        let (height, time_secs) =
            time_fn_call_async(self.get_latest_balance_height_internal()).await;

        debug!("call to get_latest_balance_height() took {time_secs} seconds");

        height
    }

    /// Determine whether the conditions are met to enter into sync mode.
    ///
    /// Specifically, compute a boolean value based on
    ///  - whether the foreign cumulative proof-of-work exceeds that of our own;
    ///  - whether the foreign block has a bigger block height and the height
    ///    difference exceeds the threshold set by the CLI.
    ///
    /// The main loop relies on this criterion to decide whether to enter sync
    /// mode. If the main loop activates sync mode, it affects the entire
    /// application.
    pub(crate) fn sync_mode_threshold_stateless(
        own_block_tip_header: &BlockHeader,
        claimed_height: BlockHeight,
        claimed_cumulative_pow: ProofOfWork,
        sync_mode_threshold: usize,
    ) -> bool {
        own_block_tip_header.cumulative_proof_of_work < claimed_cumulative_pow
            && claimed_height - own_block_tip_header.height > sync_mode_threshold as i128
    }

    /// Determine whether the conditions are met to enter into sync mode.
    ///
    /// Specifically, compute a boolean value based on
    ///  - whether the foreign cumulative proof-of-work exceeds that of our own;
    ///  - whether the foreign block has a bigger block height and the height
    ///    difference exceeds the threshold set by the CLI.
    ///
    /// The main loop relies on this criterion to decide whether to enter sync
    /// mode. If the main loop activates sync mode, it affects the entire
    /// application.
    pub(crate) fn sync_mode_criterion(
        &self,
        claimed_max_height: BlockHeight,
        claimed_cumulative_pow: ProofOfWork,
    ) -> bool {
        let own_block_tip_header = self.chain.light_state().header();
        Self::sync_mode_threshold_stateless(
            own_block_tip_header,
            claimed_max_height,
            claimed_cumulative_pow,
            self.cli().sync_mode_threshold,
        )
    }

    pub(crate) fn composer_parameters(
        &self,
        reward_address: ReceivingAddress,
    ) -> ComposerParameters {
        let next_block_height: BlockHeight = self.chain.light_state().header().height.next();
        let sender_randomness_for_composer = self
            .wallet_state
            .wallet_secret
            .generate_sender_randomness(next_block_height, reward_address.privacy_digest());

        ComposerParameters::new(
            reward_address,
            sender_randomness_for_composer,
            self.cli.guesser_fraction,
        )
    }

    /// Returns true iff the incoming block proposal is more favorable than the
    /// one we're currently working on. Returns false if client is a composer,
    /// as it's assumed that they prefer guessing on their own block.
    pub(crate) fn favor_incoming_block_proposal(
        &self,
        incoming_block_height: BlockHeight,
        incoming_guesser_fee: NativeCurrencyAmount,
    ) -> Result<(), BlockProposalRejectError> {
        if self.cli().compose {
            return Err(BlockProposalRejectError::Composing);
        }

        let expected_height = self.chain.light_state().header().height.next();
        if incoming_block_height != expected_height {
            return Err(BlockProposalRejectError::WrongHeight {
                received: incoming_block_height,
                expected: expected_height,
            });
        }

        let maybe_existing_fee = self.block_proposal.map(|x| x.total_guesser_reward());
        if maybe_existing_fee.is_some_and(|current| current >= incoming_guesser_fee)
            || incoming_guesser_fee.is_zero()
        {
            Err(BlockProposalRejectError::InsufficientFee {
                current: maybe_existing_fee,
                received: incoming_guesser_fee,
            })
        } else {
            Ok(())
        }
    }

    /// Determine whether the incoming block is more canonical than the current
    /// tip, *i.e.*, wins the fork choice rule.
    ///
    /// If the incoming block equals the current tip, this function returns
    /// false.
    pub fn incoming_block_is_more_canonical(&self, incoming_block: &Block) -> bool {
        let winner = Block::fork_choice_rule(self.chain.light_state(), incoming_block);
        winner.hash() != self.chain.light_state().hash()
    }

    /// Retrieve block height of last change to wallet balance.
    ///
    /// note: this fn could be implemented as:
    ///   1. get_balance_history()
    ///   2. sort by height
    ///   3. return height of last entry.
    ///
    /// this implementation is a bit more efficient as it avoids
    ///   the sort and some minor work looking up amount and confirmation
    ///   height of each utxo.
    ///
    /// Presently this is o(n) with the number of monitored utxos.
    /// if storage could keep track of latest spend utxo for the active
    /// tip, then this could be o(1).
    async fn get_latest_balance_height_internal(&self) -> Option<BlockHeight> {
        let current_tip_digest = self.chain.light_state().hash();
        let monitored_utxos = self.wallet_state.wallet_db.monitored_utxos();

        if monitored_utxos.is_empty().await {
            return None;
        }

        let mut max_spent_in_block: Option<BlockHeight> = None;
        let mut max_confirmed_in_block: Option<BlockHeight> = None;

        // monitored_utxos are ordered by confirmed_in_block ascending.
        // To efficiently find max(confirmed_in_block) we can start at the end
        // and work backward until we find the first utxo with a valid
        // membership proof for current tip.
        //
        // We then continue working backward through all entries to
        // determine max(spent_in_block)

        // note: Stream trait does not have a way to reverse, so instead
        // of stream_values() we use stream_many_values() and supply
        // an iterator of indexes that are already reversed.

        let stream = monitored_utxos
            .stream_many_values((0..monitored_utxos.len().await).rev())
            .await;
        pin_mut!(stream); // needed for iteration

        while let Some(mutxo) = stream.next().await {
            if max_confirmed_in_block.is_none() {
                if let Some((.., confirmed_in_block)) = mutxo.confirmed_in_block {
                    if mutxo
                        .get_membership_proof_for_block(current_tip_digest)
                        .is_some()
                    {
                        max_confirmed_in_block = Some(confirmed_in_block);
                    }
                }
            }

            if let Some((.., spent_in_block)) = mutxo.spent_in_block {
                if mutxo
                    .get_membership_proof_for_block(current_tip_digest)
                    .is_some()
                    && (max_spent_in_block.is_none()
                        || max_spent_in_block.is_some_and(|x| x < spent_in_block))
                {
                    max_spent_in_block = Some(spent_in_block);
                }
            }
        }

        max(max_confirmed_in_block, max_spent_in_block)
    }

    /// Retrieve wallet balance history
    pub async fn get_balance_history(
        &self,
    ) -> Vec<(Digest, Timestamp, BlockHeight, NativeCurrencyAmount)> {
        let current_tip_digest = self.chain.light_state().hash();
        let current_msa = self.chain.light_state().mutator_set_accumulator_after();

        let monitored_utxos = self.wallet_state.wallet_db.monitored_utxos();

        let mut history = vec![];

        let stream = monitored_utxos.stream_values().await;
        pin_mut!(stream); // needed for iteration
        while let Some(monitored_utxo) = stream.next().await {
            let Some(msmp) = monitored_utxo.membership_proof_ref_for_block(current_tip_digest)
            else {
                continue;
            };

            if let Some((confirming_block, confirmation_timestamp, confirmation_height)) =
                monitored_utxo.confirmed_in_block
            {
                let amount = monitored_utxo.utxo.get_native_currency_amount();
                history.push((
                    confirming_block,
                    confirmation_timestamp,
                    confirmation_height,
                    amount,
                ));

                if let Some((spending_block, spending_timestamp, spending_height)) =
                    monitored_utxo.spent_in_block
                {
                    let actually_spent =
                        !current_msa.verify(Tip5::hash(&monitored_utxo.utxo), msmp);
                    if actually_spent {
                        history.push((
                            spending_block,
                            spending_timestamp,
                            spending_height,
                            -amount,
                        ));
                    }
                }
            }
        }

        history
    }

    /// Generate a change UTXO to ensure that the difference in input amount
    /// and output amount goes back to us. Return the UTXO in a format compatible
    /// with claiming it later on.
    //
    // "Later on" meaning: as an [ExpectedUtxo].
    pub fn create_change_output(
        &self,
        change_amount: NativeCurrencyAmount,
        change_key: SpendingKey,
        change_utxo_notify_method: UtxoNotificationMedium,
    ) -> Result<TxOutput> {
        let Some(own_receiving_address) = change_key.to_address() else {
            bail!("Cannot create change output when supplied spending key has no corresponding address.");
        };

        let receiver_digest = own_receiving_address.privacy_digest();
        let change_sender_randomness = self.wallet_state.wallet_secret.generate_sender_randomness(
            self.chain.light_state().kernel.header.height,
            receiver_digest,
        );

        let owned = true;
        let change_output = match change_utxo_notify_method {
            UtxoNotificationMedium::OnChain => TxOutput::onchain_native_currency(
                change_amount,
                change_sender_randomness,
                own_receiving_address,
                owned,
            ),
            UtxoNotificationMedium::OffChain => TxOutput::offchain_native_currency(
                change_amount,
                change_sender_randomness,
                own_receiving_address,
                owned,
            ),
        };

        Ok(change_output)
    }

    /// generates TxOutputList from a list of address:amount pairs (outputs).
    ///
    /// This is a helper method for generating the `TxOutputList` that
    /// is required by [Self::create_transaction()].
    ///
    /// Each output may use either `OnChain` or `OffChain` notifications.
    ///
    /// If a different behavior is desired, the TxOutputList can be
    /// constructed manually.
    pub fn generate_tx_outputs(
        &self,
        outputs: impl IntoIterator<Item = (ReceivingAddress, NativeCurrencyAmount)>,
        owned_utxo_notify_medium: UtxoNotificationMedium,
        unowned_utxo_notify_medium: UtxoNotificationMedium,
    ) -> TxOutputList {
        let block_height = self.chain.light_state().header().height;

        // Convert outputs.  [address:amount] --> TxOutputList
        let tx_outputs: Vec<_> = outputs
            .into_iter()
            .map(|(address, amount)| {
                let sender_randomness = self
                    .wallet_state
                    .wallet_secret
                    .generate_sender_randomness(block_height, address.privacy_digest());

                // The UtxoNotifyMethod (Onchain or Offchain) is auto-detected
                // based on whether the address belongs to our wallet or not
                TxOutput::auto(
                    &self.wallet_state,
                    address,
                    amount,
                    sender_randomness,
                    owned_utxo_notify_medium,
                    unowned_utxo_notify_medium,
                )
            })
            .collect();

        tx_outputs.into()
    }

    /// creates a Transaction.
    ///
    /// This API provides a simple-to-use interface for creating a transaction.
    /// [Utxo](crate::models::blockchain::transaction::utxo::Utxo) inputs are automatically chosen and a change output is
    /// automatically created, such that:
    ///
    ///   change = sum(inputs) - sum(outputs) - fee.
    ///
    /// The `tx_outputs` parameter should normally be generated with
    /// [Self::generate_tx_outputs()] which determines which outputs should be
    /// `OnChain` or `OffChain`.
    ///
    /// The return value is the created transaction and some change UTXO with
    /// associated data or none if the transaction is already balanced. The
    /// associated data allows the caller to expect and later claim the change
    /// UTXO.
    ///
    /// After this call returns, it is the caller's responsibility to inform the
    /// wallet of any returned ExpectedUtxo, ie `OffChain` secret
    /// notifications, for utxos that match wallet keys.  Failure to do so can
    /// result in loss of funds!
    ///
    /// The `change_utxo_notify_method` parameter should normally be
    /// [UtxoNotificationMedium::OnChain] for safest transfer.
    ///
    /// The change_key should normally be a [SpendingKey::Symmetric] in
    /// order to save blockchain space compared to a regular address.
    ///
    /// Note that `create_transaction()` does not modify any state and does not
    /// require acquiring write lock.  This is important because internally it
    /// calls prove() which is a very lengthy operation.
    ///
    /// Example:
    ///
    /// ```text
    ///
    /// // obtain a change key
    /// // note that this is a SymmetricKey, not a regular (Generation) address.
    /// let change_key = global_state_lock
    ///     .lock_guard_mut()
    ///     .await
    ///     .wallet_state
    ///     .wallet_secret
    ///     .next_unused_spending_key(KeyType::Symmetric).await;
    ///
    /// // on-chain notification for all utxos destined for our wallet.
    /// let change_notify_medium = UtxoNotificationMedium::OnChain;
    ///
    /// // obtain read lock
    /// let state = self.state.lock_guard().await;
    ///
    /// // generate the tx_outputs
    /// let mut tx_outputs = state.generate_tx_outputs(outputs, change_notify_medium)?;
    ///
    /// // Create the transaction
    /// let (
    ///         transaction,
    ///         transaction_details,
    ///         maybe_change_utxo
    ///     ) = state
    ///     .create_transaction(
    ///         tx_outputs,                     // all outputs except `change`
    ///         change_key,                     // send `change` to this key
    ///         change_notify_medium,           // how to notify about `change` utxo
    ///         NativeCurrencyAmount::coins(2), // fee
    ///         Timestamp::now(),               // Timestamp of transaction
    ///     )
    ///     .await?;
    ///
    /// // drop read lock.
    /// drop(state);
    ///
    /// // Inform wallet of any expected incoming utxos.
    /// if let Some(change_utxo) = maybe_change_utxo {
    ///     state
    ///         .lock_guard_mut()
    ///         .await
    ///         .wallet_state.add_expected_utxos_to_wallet(change_utxo.expected_utxo())
    ///         .await?;
    /// }
    /// ```
    pub async fn create_transaction(
        &self,
        tx_outputs: TxOutputList,
        change_key: SpendingKey,
        change_utxo_notify_medium: UtxoNotificationMedium,
        fee: NativeCurrencyAmount,
        timestamp: Timestamp,
        triton_vm_job_queue: &TritonVmJobQueue,
    ) -> Result<(Transaction, Option<TxOutput>)> {
        self.create_transaction_with_prover_capability(
            tx_outputs,
            change_key,
            change_utxo_notify_medium,
            fee,
            timestamp,
            self.proving_capability(),
            triton_vm_job_queue,
        )
        .await
        .map(|(tx, _tx_details, output)| (tx, output))
    }

    /// Variant of [Self::create_transaction] that allows caller to specify
    /// prover capability. [Self::create_transaction] is the preferred interface
    /// for anything but tests.
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn create_transaction_with_prover_capability(
        &self,
        mut tx_outputs: TxOutputList,
        change_key: SpendingKey,
        change_utxo_notify_medium: UtxoNotificationMedium,
        fee: NativeCurrencyAmount,
        timestamp: Timestamp,
        prover_capability: TxProvingCapability,
        triton_vm_job_queue: &TritonVmJobQueue,
    ) -> Result<(Transaction, TransactionDetails, Option<TxOutput>)> {
        // TODO: Attempt to simplify method interface somehow, maybe by moving
        // it to GlobalStateLock?
        let tip = self.chain.light_state();
        let tip_mutator_set_accumulator = tip.mutator_set_accumulator_after();
        let tip_digest = tip.hash();

        // 1. create/add change output if necessary.
        let total_spend = tx_outputs.total_native_coins() + fee;

        // collect spendable inputs
        let tx_inputs = self
            .wallet_state
            .allocate_sufficient_input_funds(
                total_spend,
                tip_digest,
                &tip_mutator_set_accumulator,
                timestamp,
            )
            .await?;

        let total_spendable = tx_inputs
            .iter()
            .map(|x| x.utxo.get_native_currency_amount())
            .sum();

        // Add change, if required to balance tx.
        let mut maybe_change_output = None;
        if total_spend < total_spendable {
            let amount = total_spendable.checked_sub(&total_spend).ok_or_else(|| {
                anyhow::anyhow!("overflow subtracting total_spend from input_amount")
            })?;

            let change_utxo =
                self.create_change_output(amount, change_key, change_utxo_notify_medium)?;
            tx_outputs.push(change_utxo.clone());
            maybe_change_output = Some(change_utxo);
        }

        let transaction_details = TransactionDetails::new_without_coinbase(
            tx_inputs,
            tx_outputs.to_owned(),
            fee,
            timestamp,
            tip_mutator_set_accumulator,
        )?;

        // note: if this task is cancelled, the proving job will continue
        // because TritonVmJobOptions::cancel_job_rx is None.
        // see how compose_task handles cancellation in mine_loop.

        // 2. Create the transaction
        let transaction = Self::create_raw_transaction(
            &transaction_details,
            prover_capability,
            triton_vm_job_queue,
            (
                TritonVmJobPriority::High,
                self.cli.max_log2_padded_height_for_proofs,
            )
                .into(),
        )
        .await?;

        Ok((transaction, transaction_details, maybe_change_output))
    }

    /// creates a Transaction.
    ///
    /// This API provides the caller complete control over selection of inputs
    /// and outputs.  When fine grained control is not required,
    /// [Self::create_transaction()] is easier to use and should be preferred.
    ///
    /// It is the caller's responsibility to provide inputs and outputs such
    /// that sum(inputs) == sum(outputs) + fee.  Else an error will result.
    ///
    /// Note that this means the caller must calculate the `change` amount if any
    /// and provide an output for the change.
    ///
    /// The `tx_outputs` parameter should normally be generated with
    /// [Self::generate_tx_outputs()] which determines which outputs should be
    /// notified `OnChain` or `OffChain`.
    ///
    /// After this call returns, it is the caller's responsibility to inform the
    /// wallet of any returned [ExpectedUtxo] for utxos that match wallet keys.
    /// Failure to do so can result in loss of funds!
    ///
    /// Note that `create_raw_transaction()` does not modify any state and does
    /// not require acquiring write lock.  This is important because internally
    /// it calls prove() which is a very lengthy operation.
    ///
    /// Example:
    ///
    /// See the implementation of [Self::create_transaction()].
    pub(crate) async fn create_raw_transaction(
        transaction_details: &TransactionDetails,
        proving_power: TxProvingCapability,
        triton_vm_job_queue: &TritonVmJobQueue,
        proof_job_options: TritonVmProofJobOptions,
    ) -> anyhow::Result<Transaction> {
        // note: this executes the prover which can take a very
        //       long time, perhaps minutes.  The `await` here, should avoid
        //       block the tokio executor and other async tasks.
        Self::create_transaction_from_data_worker(
            transaction_details,
            proving_power,
            triton_vm_job_queue,
            proof_job_options,
        )
        .await
    }

    // note: this executes the prover which can take a very
    //       long time, perhaps minutes. It should never be
    //       called directly.
    //       Use create_transaction_from_data() instead.
    //
    async fn create_transaction_from_data_worker(
        transaction_details: &TransactionDetails,
        proving_power: TxProvingCapability,
        triton_vm_job_queue: &TritonVmJobQueue,
        proof_job_options: TritonVmProofJobOptions,
    ) -> anyhow::Result<Transaction> {
        let primitive_witness = PrimitiveWitness::from_transaction_details(transaction_details);

        debug!("primitive witness for transaction: {}", primitive_witness);

        info!(
            "Start: generate proof for {}-in {}-out transaction",
            primitive_witness.input_utxos.utxos.len(),
            primitive_witness.output_utxos.utxos.len()
        );
        let kernel = primitive_witness.kernel.clone();
        let proof = match proving_power {
            TxProvingCapability::PrimitiveWitness => TransactionProof::Witness(primitive_witness),
            TxProvingCapability::LockScript => todo!(),
            TxProvingCapability::ProofCollection => TransactionProof::ProofCollection(
                ProofCollection::produce(
                    &primitive_witness,
                    triton_vm_job_queue,
                    proof_job_options,
                )
                .await?,
            ),
            TxProvingCapability::SingleProof => TransactionProof::SingleProof(
                SingleProof::produce(&primitive_witness, triton_vm_job_queue, proof_job_options)
                    .await?,
            ),
        };

        Ok(Transaction { kernel, proof })
    }

    pub(crate) async fn get_own_handshakedata(&self) -> HandshakeData {
        let listen_port = self.cli().own_listen_port();
        HandshakeData {
            tip_header: *self.chain.light_state().header(),
            listen_port,
            network: self.cli().network,
            instance_id: self.net.instance_id,
            version: VersionString::try_from_str(VERSION).unwrap_or_else(|_| {
                panic!(
                "Must be able to convert own version number to fixed-size string. Got {VERSION}")
            }),
            // For now, all nodes are archival nodes
            is_archival_node: self.chain.is_archival_node(),
            timestamp: SystemTime::now(),
        }
    }

    /// In case the wallet database is corrupted or deleted, this method will restore
    /// monitored UTXO data structures from recovery data. This method should only be
    /// called on startup, not while the program is running, since it will only restore
    /// a wallet state, if the monitored UTXOs have been deleted. Not merely if they
    /// are not synced with a valid mutator set membership proof. And this corruption
    /// can only happen if the wallet database is deleted or corrupted.
    ///
    /// # Panics
    ///
    /// Panics if the mutator set is not synced to current tip.
    pub(crate) async fn restore_monitored_utxos_from_recovery_data(&mut self) -> Result<()> {
        let tip_hash = self.chain.light_state().hash();
        let ams_ref = &self.chain.archival_state().archival_mutator_set;

        let asm_sync_label = ams_ref.get_sync_label().await;
        assert_eq!(
            tip_hash, asm_sync_label,
            "Error: sync label in archival mutator set database disagrees with \
            block tip. Archival mutator set must be synced to tip for successful \
            MUTXO recovery.\n\
            Possible causes: different or new genesis block; corrupt file system.\n\
            Possible solution: try deleting the database at `DATA_DIR/databases/`. \
            Get the value of `DATA_DIR` from the first message in the log, and \
            *do not* delete the wallet file or directory.\n\n\
            Tip:\n{tip_hash};\nsync label:\n{asm_sync_label}"
        );

        // Fetch all incoming UTXOs from recovery data. Then make a HashMap for
        // fast lookup.
        let incoming_utxos = self.wallet_state.read_utxo_ms_recovery_data().await?;
        let incoming_utxo_count = incoming_utxos.len();
        info!("Checking {} incoming UTXOs", incoming_utxo_count);

        let mut recovery_data_for_missing_mutxos = vec![];
        {
            // Two UTXOs are considered the same iff their AOCL index and
            // their addition record agree. Otherwise, they are different.
            let mutxos: HashMap<(u64, AdditionRecord), MonitoredUtxo> = self
                .wallet_state
                .wallet_db
                .monitored_utxos()
                .stream_values()
                .await
                .map(|x| ((x.aocl_index(), x.addition_record()), x))
                .collect()
                .await;
            let mut seen_recovery_entries = HashSet::<Digest>::default();

            for incoming_utxo in incoming_utxos {
                let new_value = seen_recovery_entries.insert(Tip5::hash(&incoming_utxo));
                assert!(
                    new_value,
                    "Recovery data may not contain duplicated entries. Entry with AOCL index {} \
                     was duplicated. Try removing the duplicated entry from the file.",
                    incoming_utxo.aocl_index
                );

                if mutxos
                    .get(&(incoming_utxo.aocl_index, incoming_utxo.addition_record()))
                    .map(|x| x.get_latest_membership_proof_entry())
                    .is_some()
                {
                    continue;
                }

                // If no match is found, add the UTXO to the list of missing UTXOs
                recovery_data_for_missing_mutxos.push(incoming_utxo);
            }
        }

        if recovery_data_for_missing_mutxos.is_empty() {
            info!(
                "No missing monitored UTXOs found in wallet database. Wallet database looks good."
            );
            return Ok(());
        }

        let existing_guesser_preimages = self
            .wallet_state
            .wallet_db
            .guesser_preimages()
            .get_all()
            .await;
        let mut new_guesser_preimages = vec![];

        // For all recovery data where we did not find a matching monitored UTXO,
        // recover the MS membership proof, and insert a new monitored UTXO into the
        // wallet database.
        info!(
            "Attempting to restore {} missing monitored UTXOs to wallet database",
            recovery_data_for_missing_mutxos.len()
        );
        let current_aocl_leaf_count = ams_ref.ams().aocl.num_leafs().await;
        let mut restored_mutxos = 0;
        for incoming_utxo in recovery_data_for_missing_mutxos {
            // If the referenced UTXO is in the future from our tip, do not attempt to recover it. Instead: warn the user of this.
            if current_aocl_leaf_count <= incoming_utxo.aocl_index {
                warn!("Cannot restore UTXO with AOCL index {} because it is in the future from our tip. Current AOCL leaf count is {current_aocl_leaf_count}. Maybe this UTXO can be recovered once more blocks are downloaded from peers?", incoming_utxo.aocl_index);
                continue;
            }

            // Check if UTXO is guesser-reward and associated key doesn't already exist.
            if incoming_utxo.is_guesser_fee()
                && !existing_guesser_preimages.contains(&incoming_utxo.receiver_preimage)
            {
                new_guesser_preimages.push(incoming_utxo.receiver_preimage);
            }

            let ms_item = Hash::hash(&incoming_utxo.utxo);
            let restored_msmp_res = ams_ref
                .ams()
                .restore_membership_proof(
                    ms_item,
                    incoming_utxo.sender_randomness,
                    incoming_utxo.receiver_preimage,
                    incoming_utxo.aocl_index,
                )
                .await;
            let restored_msmp = match restored_msmp_res {
                Ok(msmp) => {
                    // Verify that the restored MSMP is valid
                    if !ams_ref.ams().verify(ms_item, &msmp).await {
                        warn!("Restored MSMP is invalid. Skipping restoration of UTXO with AOCL index {}. Maybe this UTXO is on an abandoned chain? Or maybe it was spent?", incoming_utxo.aocl_index);
                        continue;
                    }

                    msmp
                }
                Err(err) => bail!("Could not restore MS membership proof. Got: {err}"),
            };

            let mut restored_mutxo =
                MonitoredUtxo::new(incoming_utxo.utxo, self.wallet_state.number_of_mps_per_utxo);
            restored_mutxo.add_membership_proof_for_tip(tip_hash, restored_msmp);

            // Add block info for restored MUTXO
            let confirming_block_digest = self
                .chain
                .archival_state()
                .canonical_block_digest_of_aocl_index(incoming_utxo.aocl_index)
                .await?
                .expect("Confirming block must exist");
            let confirming_block_header = self
                .chain
                .archival_state()
                .get_block_header(confirming_block_digest)
                .await
                .expect("Confirming block header must exist");
            restored_mutxo.confirmed_in_block = Some((
                confirming_block_digest,
                confirming_block_header.timestamp,
                confirming_block_header.height,
            ));

            self.wallet_state
                .wallet_db
                .monitored_utxos_mut()
                .push(restored_mutxo)
                .await;
            restored_mutxos += 1;
        }

        // Update state with all guesser-preimage keys from guesser-fee UTXOs
        for new_guesser_preimage in new_guesser_preimages {
            self.wallet_state
                .add_raw_hash_key(new_guesser_preimage)
                .await;
        }

        self.wallet_state.wallet_db.persist().await;
        info!("Successfully restored {restored_mutxos} monitored UTXOs to wallet database");

        Ok(())
    }

    ///  Locking:
    ///   * acquires `monitored_utxos_lock` for write
    pub async fn resync_membership_proofs_from_stored_blocks(
        &mut self,
        tip_hash: Digest,
    ) -> Result<()> {
        // loop over all monitored utxos
        let monitored_utxos = self.wallet_state.wallet_db.monitored_utxos_mut();

        'outer: for i in 0..monitored_utxos.len().await {
            let i = i as Index;
            let monitored_utxo = monitored_utxos.get(i).await;

            // Ignore those MUTXOs that were marked as abandoned
            if monitored_utxo.abandoned_at.is_some() {
                continue;
            }

            // ignore synced ones
            if monitored_utxo.is_synced_to(tip_hash) {
                continue;
            }

            debug!(
                "Resyncing monitored UTXO number {}, with hash {}",
                i,
                Hash::hash(&monitored_utxo.utxo)
            );

            // If the UTXO was not confirmed yet, there is no
            // point in synchronizing its membership proof.
            let (confirming_block_digest, confirming_block_height) =
                match monitored_utxo.confirmed_in_block {
                    Some((confirmed_block_hash, _timestamp, block_height)) => {
                        (confirmed_block_hash, block_height)
                    }
                    None => {
                        continue;
                    }
                };

            // try latest (block hash, membership proof) entry
            let (block_hash, mut membership_proof) = monitored_utxo
                .get_latest_membership_proof_entry()
                .expect("Database not in consistent state. Monitored UTXO must have at least one membership proof.");

            // request path-to-tip
            let (backwards, _luca, forwards) = self
                .chain
                .archival_state()
                .find_path(block_hash, tip_hash)
                .await;

            // after this point, we may be modifying it.
            let mut monitored_utxo = monitored_utxo.clone();

            // walk backwards, reverting
            for revert_block_hash in backwards.into_iter() {
                // Was the UTXO confirmed in this block? If so, there
                // is nothing we can do except orphan the UTXO: that
                // is, leave it without a synced membership proof.
                // Whenever current owned UTXOs are queried, one
                // should take care to filter for UTXOs that have a
                // membership proof synced to the current block tip.
                if confirming_block_digest == revert_block_hash {
                    warn!(
                        "Could not recover MSMP as transaction appears to be on an abandoned chain"
                    );
                    break 'outer;
                }

                let revert_block = self
                    .chain
                    .archival_state()
                    .get_block(revert_block_hash)
                    .await?
                    .unwrap();
                let revert_block_parent = self
                    .chain
                    .archival_state()
                    .get_block(revert_block.kernel.header.prev_block_digest)
                    .await?
                    .expect("All blocks that are reverted must have a parent, since genesis block can never be reverted.");
                let previous_mutator_set =
                    revert_block_parent.mutator_set_accumulator_after().clone();

                debug!("MUTXO confirmed at height {confirming_block_height}, reverting for height {} on abandoned chain", revert_block.kernel.header.height);

                // revert removals
                let removal_records = revert_block.kernel.body.transaction_kernel.inputs.clone();
                for removal_record in removal_records.iter().rev() {
                    membership_proof.revert_update_from_remove(removal_record);
                }

                // revert additions
                membership_proof.revert_update_from_batch_addition(&previous_mutator_set);

                // unset spent_in_block field if the UTXO was spent in this block
                if let Some((spent_block_hash, _, _)) = monitored_utxo.spent_in_block {
                    if spent_block_hash == revert_block_hash {
                        monitored_utxo.spent_in_block = None;
                    }
                }

                // assert valid (if unspent)
                assert!(monitored_utxo.spent_in_block.is_some() || previous_mutator_set
                    .verify(Hash::hash(&monitored_utxo.utxo), &membership_proof), "Failed to verify monitored UTXO {monitored_utxo:?}\n against previous MSA in block {revert_block:?}");
            }

            // walk forwards, applying
            for apply_block_hash in forwards.into_iter() {
                // Was the UTXO confirmed in this block?
                // This can occur in some edge cases of forward-only
                // resynchronization. In this case, assume the
                // membership proof is already synced to this block.
                if confirming_block_digest == apply_block_hash {
                    continue;
                }

                let apply_block = self
                    .chain
                    .archival_state()
                    .get_block(apply_block_hash)
                    .await?
                    .unwrap();
                let predecessor_block = self
                    .chain
                    .archival_state()
                    .get_block(apply_block.kernel.header.prev_block_digest)
                    .await?;
                let mut block_msa = match &predecessor_block {
                    Some(block) => block.mutator_set_accumulator_after().clone(),
                    None => MutatorSetAccumulator::default(),
                };
                let MutatorSetUpdate {
                    additions,
                    removals,
                } = apply_block.mutator_set_update();

                // apply additions
                for addition_record in additions.iter() {
                    membership_proof
                        .update_from_addition(
                            Hash::hash(&monitored_utxo.utxo),
                            &block_msa,
                            addition_record,
                        )
                        .expect("Could not update membership proof with addition record.");
                    block_msa.add(addition_record);
                }

                // apply removals
                for removal_record in removals.iter() {
                    membership_proof.update_from_remove(removal_record);
                    block_msa.remove(removal_record);
                }

                assert_eq!(
                    block_msa.hash(),
                    apply_block.mutator_set_accumulator_after().hash()
                );
            }

            // store updated membership proof
            monitored_utxo.add_membership_proof_for_tip(tip_hash, membership_proof);

            // update storage.
            monitored_utxos.set(i, monitored_utxo).await
        }

        // Update sync label and persist
        self.wallet_state.wallet_db.set_sync_label(tip_hash).await;
        self.wallet_state.wallet_db.persist().await;

        Ok(())
    }

    /// Delete from the database all monitored UTXOs from abandoned chains with a depth deeper than
    /// `block_depth_threshold`. Use `prune_mutxos_of_unknown_depth = true` to remove MUTXOs from
    /// abandoned chains of unknown depth.
    /// Returns the number of monitored UTXOs that were marked as abandoned.
    ///
    /// Locking:
    ///  * acquires `monitored_utxos` lock for write
    pub async fn prune_abandoned_monitored_utxos(
        &mut self,
        block_depth_threshold: usize,
    ) -> Result<usize> {
        const MIN_BLOCK_DEPTH_FOR_MUTXO_PRUNING: usize = 10;
        if block_depth_threshold < MIN_BLOCK_DEPTH_FOR_MUTXO_PRUNING {
            bail!(
                "
                Cannot prune monitored UTXOs with a depth threshold less than
                {MIN_BLOCK_DEPTH_FOR_MUTXO_PRUNING}. Got threshold {block_depth_threshold}"
            )
        }

        // Find monitored_utxo for updating
        let current_tip_header = self.chain.light_state().header();
        let current_tip_digest = self.chain.light_state().hash();
        let current_tip_info: (Digest, Timestamp, BlockHeight) = (
            current_tip_digest,
            current_tip_header.timestamp,
            current_tip_header.height,
        );
        let monitored_utxos = self.wallet_state.wallet_db.monitored_utxos_mut();
        let mut removed_count = 0;
        for i in 0..monitored_utxos.len().await {
            let mut mutxo = monitored_utxos.get(i).await;

            // 1. Spent MUTXOs are not marked as abandoned, as there's no reason to maintain them
            //    once the spending block is buried sufficiently deep
            // 2. If synced to current tip, there is nothing more to do with this MUTXO
            // 3. If already marked as abandoned, we don't do that again
            if mutxo.spent_in_block.is_some()
                || mutxo.is_synced_to(current_tip_info.0)
                || mutxo.abandoned_at.is_some()
            {
                continue;
            }

            // MUTXO is neither spent nor synced. Mark as abandoned
            // if it was confirmed in block that is now abandoned,
            // and if that block is older than threshold.
            if let Some((_, _, block_height_confirmed)) = mutxo.confirmed_in_block {
                let depth = current_tip_header.height - block_height_confirmed + 1;

                let abandoned = depth >= block_depth_threshold as i128
                    && mutxo.was_abandoned(self.chain.archival_state()).await;

                if abandoned {
                    mutxo.abandoned_at = Some(current_tip_info);
                    monitored_utxos.set(i, mutxo).await;
                    removed_count += 1;
                }
            }
        }

        Ok(removed_count)
    }

    pub async fn persist_wallet(&mut self) -> Result<()> {
        // flush wallet databases
        self.wallet_state.wallet_db.persist().await;
        Ok(())
    }

    pub async fn flush_databases(&mut self) -> Result<()> {
        // flush wallet databases
        self.wallet_state.wallet_db.persist().await;

        // flush block_index database
        self.chain.archival_state_mut().block_index_db.flush().await;

        // persist archival_mutator_set, with sync label
        let hash = self.chain.archival_state().get_tip().await.hash();
        self.chain
            .archival_state_mut()
            .archival_mutator_set
            .set_sync_label(hash)
            .await;

        self.chain
            .archival_state_mut()
            .archival_mutator_set
            .persist()
            .await;

        self.chain
            .archival_state_mut()
            .archival_block_mmr
            .persist()
            .await;

        // flush peer_standings
        self.net.peer_databases.peer_standings.flush().await;

        debug!("Flushed all databases");

        Ok(())
    }

    /// Update client's state with a new block.
    ///
    /// The new block is assumed to be valid, also wrt. to proof-of-work.
    /// The new block will be set as the new tip, regardless of its
    /// cumulative proof-of-work number.
    ///
    /// Returns a list of update-jobs that should be
    /// performed by this client.
    pub(crate) async fn set_new_tip(
        &mut self,
        new_block: Block,
    ) -> Result<Vec<UpdateMutatorSetDataJob>> {
        self.set_new_tip_internal(new_block).await
    }

    /// Store a block to client's state *without* marking this block as a new
    /// tip. No validation of block happens, as this is the caller's
    /// responsibility.
    pub(crate) async fn store_block_not_tip(&mut self, block: Block) -> Result<()> {
        crate::macros::log_scope_duration!();

        self.chain
            .archival_state_mut()
            .write_block_not_tip(&block)
            .await?;

        // Mempool is not updated, as it's only defined relative to the tip.
        // Wallet is not updated, as it can be synced to tip at any point.

        // Flush databases
        self.flush_databases().await?;

        Ok(())
    }

    /// Update client's state with a new block. Block is assumed to be valid, also wrt. to PoW.
    /// The received block will be set as the new tip, regardless of its accumulated PoW. or its
    /// validity.
    ///
    /// Returns a list of update-jobs that should be
    /// performed by this client.
    async fn set_new_tip_internal(
        &mut self,
        new_block: Block,
    ) -> Result<Vec<UpdateMutatorSetDataJob>> {
        crate::macros::log_scope_duration!();

        // Apply the updates
        self.chain
            .archival_state_mut()
            .write_block_as_tip(&new_block)
            .await?;

        self.chain
            .archival_state_mut()
            .append_to_archival_block_mmr(&new_block)
            .await;

        // update the mutator set with the UTXOs from this block
        self.chain
            .archival_state_mut()
            .update_mutator_set(&new_block)
            .await
            .expect("Updating mutator set must succeed");

        // Get parent of tip for mutator-set data needed for various updates. Parent of the
        // stored block will always exist since all blocks except the genesis block have a
        // parent, and the genesis block is considered code, not data, so the genesis block
        // will never be changed or updated through this method.
        let tip_parent = self
            .chain
            .archival_state()
            .get_tip_parent()
            .await
            .expect("Parent must exist when storing a new block");

        // Sanity check that must always be true for a valid block
        assert_eq!(
            tip_parent.hash(),
            new_block.header().prev_block_digest,
            "Tip parent has must match indicated parent hash"
        );
        let previous_ms_accumulator = tip_parent.mutator_set_accumulator_after().clone();

        // Update mempool with UTXOs from this block. This is done by
        // removing all transaction that became invalid/was mined by this
        // block. Also returns the list of update-jobs that should be
        // performed by this client.
        let (mempool_events, update_jobs) = self.mempool.update_with_block_and_predecessor(
            &new_block,
            &tip_parent,
            self.proving_capability(),
            self.cli().compose,
        );

        // update wallet state with relevant UTXOs from this block
        self.wallet_state
            .update_wallet_state_with_new_block(&previous_ms_accumulator, &new_block)
            .await?;
        self.wallet_state
            .handle_mempool_events(mempool_events)
            .await;

        self.chain.light_state_mut().set_block(new_block);

        // Reset block proposal, as that field pertains to the block that
        // was just set as new tip.
        self.block_proposal = BlockProposal::none();

        // Flush databases
        self.flush_databases().await?;

        Ok(update_jobs)
    }

    /// resync membership proofs
    pub async fn resync_membership_proofs(&mut self) -> Result<()> {
        // Do not fix memberhip proofs if node is in sync mode, as we would otherwise
        // have to sync many times, instead of just *one* time once we have caught up.
        if self.net.sync_anchor.is_some() {
            debug!("Not syncing MS membership proofs because we are syncing");
            return Ok(());
        }

        // is it necessary?
        let current_tip_digest = self.chain.light_state().hash();
        if self.wallet_state.is_synced_to(current_tip_digest).await {
            debug!("Membership proof syncing not needed");
            return Ok(());
        }

        // do we have blocks?
        if self.chain.is_archival_node() {
            return self
                .resync_membership_proofs_from_stored_blocks(current_tip_digest)
                .await;
        }

        // request blocks from peers
        todo!("We don't yet support non-archival nodes");

        // Ok(())
    }

    pub(crate) async fn response_to_sync_challenge(
        &self,
        sync_challenge: SyncChallenge,
    ) -> Result<SyncChallengeResponse> {
        async fn fetch_block_pair(
            state: &GlobalState,
            child_digest: Digest,
        ) -> Option<(Block, Block)> {
            let child = state
                .chain
                .archival_state()
                .get_block(child_digest)
                .await
                .expect("fetching block from archival state should work.");
            let Some(child) = child else {
                warn!("Got sync challenge for unknown tip");

                return None;
            };
            if child.header().height < 2.into() {
                warn!("Got sync challenge for tip of too low height; cannot send genesis block");

                return None;
            }

            let parent_digest = child.header().prev_block_digest;
            let parent = state
                .chain
                .archival_state()
                .get_block(parent_digest)
                .await
                .expect("fetching block from archival state should work.")
                .expect(
                    "parent of known block from archival state must exist, if height exceeds 1.",
                );

            Some((parent, child))
        }

        let Some((tip_parent, tip)) = fetch_block_pair(self, sync_challenge.tip_digest).await
        else {
            bail!("could not fetch tip and tip predecessor");
        };

        let tip_height = tip.header().height;
        if tip_height < (SYNC_CHALLENGE_POW_WITNESS_LENGTH as u64).into() {
            bail!("tip height is too small for sync mode")
        }

        let mut block_pairs: Vec<(TransferBlock, TransferBlock)> = vec![];
        let mut block_mmr_mps = vec![];
        for child_height in sync_challenge.challenges {
            if child_height < 2u64.into() {
                bail!("challenge asks for genesis block");
            }
            if child_height >= tip.header().height {
                bail!("challenge asks for height that's not ancestor to tip.");
            }

            let Some(child_digest) = self
                .chain
                .archival_state()
                .archival_block_mmr
                .ammr()
                .try_get_leaf(child_height.into())
                .await
            else {
                bail!("could not get leaf from archival block mmr");
            };
            let Some((p, c)) = fetch_block_pair(self, child_digest).await else {
                bail!("could not fetch indicated block pair");
            };

            // Notice that the MMR membership proofs are relative to an MMR
            // where the tip digest *has* been added. So it is not relative to
            // the block MMR accumulator present in the tip block, as it only
            // refers to its ancestors. Rather, it's relative to the block MMR
            // accumulator present in the tip's child.
            block_mmr_mps.push(
                self.chain
                    .archival_state()
                    .archival_block_mmr
                    .ammr()
                    .prove_membership_relative_to_smaller_mmr(
                        child_height.into(),
                        tip_height.next().into(),
                    )
                    .await,
            );
            block_pairs.push((
                p.try_into()
                    .expect("blocks from archive must be transferable"),
                c.try_into()
                    .expect("blocks from archive must be transferable"),
            ));
        }

        let mut pow_witnesses: Vec<BlockHeaderWithBlockHashWitness> = vec![];
        let mut block_hash = tip.hash();
        while pow_witnesses.len() < SYNC_CHALLENGE_POW_WITNESS_LENGTH {
            let pow_witness = self
                .chain
                .archival_state()
                .block_header_with_hash_witness(block_hash)
                .await
                .unwrap_or_else(|| {
                    panic!("Pow-witness for block with hash {block_hash} must exist")
                });
            block_hash = pow_witness.header.prev_block_digest;
            pow_witnesses.push(pow_witness);
        }

        let response = SyncChallengeResponse {
            tip: tip
                .try_into()
                .expect("All blocks from archival state should be transferable."),
            tip_parent: tip_parent
                .try_into()
                .expect("All blocks from archival state should be transferable."),
            blocks: block_pairs.try_into().unwrap(),
            membership_proofs: block_mmr_mps.try_into().unwrap(),
            pow_witnesses: pow_witnesses.try_into().unwrap(),
        };

        Ok(response)
    }

    #[inline]
    fn cli(&self) -> &cli_args::Args {
        &self.cli
    }

    /// Return the list of peers that were supplied as CLI arguments.
    pub(crate) fn cli_peers(&self) -> Vec<SocketAddr> {
        self.cli().peers.clone()
    }

    pub(crate) fn proving_capability(&self) -> TxProvingCapability {
        self.cli().proving_capability()
    }

    pub(crate) fn min_gobbling_fee(&self) -> NativeCurrencyAmount {
        self.cli().min_gobbling_fee
    }

    pub(crate) fn gobbling_fraction(&self) -> f64 {
        self.cli().gobbling_fraction
    }

    pub(crate) fn max_num_proofs(&self) -> usize {
        self.cli().max_num_proofs
    }

    /// clears all Tx from mempool and notifies wallet of changes.
    pub async fn mempool_clear(&mut self) {
        let events = self.mempool.clear();
        self.wallet_state.handle_mempool_events(events).await
    }

    /// adds Tx to mempool and notifies wallet of change.
    pub(crate) async fn mempool_insert(
        &mut self,
        transaction: Transaction,
        origin: TransactionOrigin,
    ) {
        let events = self.mempool.insert(transaction, origin);
        self.wallet_state.handle_mempool_events(events).await
    }

    /// prunes stale tx in mempool and notifies wallet of changes.
    pub async fn mempool_prune_stale_transactions(&mut self) {
        let events = self.mempool.prune_stale_transactions();
        self.wallet_state.handle_mempool_events(events).await
    }
}

#[cfg(test)]
mod global_state_tests {
    use itertools::Itertools;
    use num_traits::Zero;
    use rand::random;
    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;
    use tracing_test::traced_test;
    use wallet::address::generation_address::GenerationSpendingKey;
    use wallet::address::KeyType;
    use wallet::expected_utxo::UtxoNotifier;
    use wallet::WalletSecret;

    use super::*;
    use crate::config_models::network::Network;
    use crate::mine_loop::mine_loop_tests::make_coinbase_transaction_from_state;
    use crate::models::blockchain::block::Block;
    use crate::models::state::wallet::address::hash_lock_key::HashLockKey;
    use crate::tests::shared::fake_valid_successor_for_tests;
    use crate::tests::shared::make_mock_block;
    use crate::tests::shared::make_mock_block_guesser_preimage_and_guesser_fraction;
    use crate::tests::shared::mock_genesis_global_state;
    use crate::tests::shared::wallet_state_has_all_valid_mps;

    mod handshake {
        use super::*;

        #[tokio::test]
        async fn generating_own_handshake_doesnt_crash() {
            mock_genesis_global_state(
                Network::Main,
                2,
                WalletSecret::devnet_wallet(),
                cli_args::Args::default(),
            )
            .await
            .lock_guard()
            .await
            .get_own_handshakedata()
            .await;
        }

        #[traced_test]
        #[tokio::test]
        async fn handshakes_listen_port_is_some_when_max_peers_is_default() {
            let network = Network::Main;
            let bob = mock_genesis_global_state(
                network,
                2,
                WalletSecret::devnet_wallet(),
                cli_args::Args::default(),
            )
            .await;

            let handshake_data = bob
                .global_state_lock
                .lock_guard()
                .await
                .get_own_handshakedata()
                .await;
            assert!(handshake_data.listen_port.is_some());
        }

        #[traced_test]
        #[tokio::test]
        async fn handshakes_listen_port_is_none_when_max_peers_is_zero() {
            let network = Network::Main;
            let mut bob = mock_genesis_global_state(
                network,
                2,
                WalletSecret::devnet_wallet(),
                cli_args::Args::default(),
            )
            .await;
            let no_incoming_connections = cli_args::Args {
                max_num_peers: 0,
                ..Default::default()
            };
            bob.set_cli(no_incoming_connections).await;

            let handshake_data = bob
                .global_state_lock
                .lock_guard()
                .await
                .get_own_handshakedata()
                .await;
            assert!(handshake_data.listen_port.is_none());
        }
    }

    #[traced_test]
    #[tokio::test]
    async fn premine_recipient_cannot_spend_premine_before_and_can_after_release_date() {
        let network = Network::Main;
        let mut rng = StdRng::seed_from_u64(u64::from_str_radix("3014221", 6).unwrap());

        let alice = WalletSecret::new_pseudorandom(rng.random());
        let bob = mock_genesis_global_state(
            network,
            2,
            WalletSecret::devnet_wallet(),
            cli_args::Args::default(),
        )
        .await;
        assert!(
            !bob.lock_guard()
                .await
                .wallet_state
                .wallet_db
                .monitored_utxos()
                .get_all()
                .await
                .is_empty(),
            "Bob must be premine recipient"
        );

        let bob_spending_key = bob
            .lock_guard()
            .await
            .wallet_state
            .wallet_secret
            .nth_generation_spending_key_for_tests(0);

        let genesis_block = Block::genesis(network);
        let alice_address = alice.nth_generation_spending_key_for_tests(0).to_address();
        let nine_money_output = TxOutput::offchain_native_currency(
            NativeCurrencyAmount::coins(9),
            rng.random(),
            alice_address.into(),
            false,
        );
        let tx_outputs: TxOutputList = vec![nine_money_output].into();

        // one month before release date, we should not be able to create the transaction
        let launch = genesis_block.kernel.header.timestamp;
        let six_months = Timestamp::months(6);
        let one_month = Timestamp::months(1);
        assert!(bob
            .lock_guard()
            .await
            .create_transaction_with_prover_capability(
                tx_outputs.clone(),
                bob_spending_key.into(),
                UtxoNotificationMedium::OffChain,
                NativeCurrencyAmount::coins(1),
                launch + six_months - one_month,
                TxProvingCapability::ProofCollection,
                &TritonVmJobQueue::dummy()
            )
            .await
            .is_err());

        // one month after though, we should be
        let (tx, _, _change_output) = bob
            .lock_guard()
            .await
            .create_transaction_with_prover_capability(
                tx_outputs,
                bob_spending_key.into(),
                UtxoNotificationMedium::OffChain,
                NativeCurrencyAmount::coins(1),
                launch + six_months + one_month,
                TxProvingCapability::ProofCollection,
                &TritonVmJobQueue::dummy(),
            )
            .await
            .unwrap();
        assert!(tx.is_valid().await);

        assert_eq!(
            2,
            tx.kernel.outputs.len(),
            "tx must have a send output and a change output"
        );
        assert_eq!(
            1,
            tx.kernel.inputs.len(),
            "tx must have exactly one input, a genesis UTXO"
        );

        // Test with a transaction with three outputs and one (premine) input
        let mut output_utxos = vec![];
        for i in 2..5 {
            let that_much_money: NativeCurrencyAmount = NativeCurrencyAmount::coins(i);
            let output_utxo = TxOutput::offchain_native_currency(
                that_much_money,
                rng.random(),
                alice_address.into(),
                false,
            );
            output_utxos.push(output_utxo);
        }

        let (new_tx, _, _change) = bob
            .lock_guard()
            .await
            .create_transaction_with_prover_capability(
                output_utxos.into(),
                bob_spending_key.into(),
                UtxoNotificationMedium::OffChain,
                NativeCurrencyAmount::coins(1),
                launch + six_months + one_month,
                TxProvingCapability::ProofCollection,
                &TritonVmJobQueue::dummy(),
            )
            .await
            .unwrap();
        assert!(new_tx.is_valid().await);
        assert_eq!(
            4,
            new_tx.kernel.outputs.len(),
            "tx must have three send outputs and a change output"
        );
        assert_eq!(
            1,
            new_tx.kernel.inputs.len(),
            "tx must have exactly one input, a genesis UTXO"
        );
    }

    #[traced_test]
    #[tokio::test]
    async fn restore_monitored_utxos_from_recovery_data_test() {
        let network = Network::Main;
        let mut rng = rand::rng();
        let wallet = WalletSecret::devnet_wallet();
        let own_key = wallet.nth_generation_spending_key_for_tests(0);
        let mut global_state_lock =
            mock_genesis_global_state(network, 2, wallet.clone(), cli_args::Args::default()).await;
        let genesis_block = Block::genesis(network);
        let guesser_preimage = wallet.guesser_preimage(genesis_block.hash());
        let (block1, composer_utxos) = make_mock_block_guesser_preimage_and_guesser_fraction(
            &genesis_block,
            None,
            own_key,
            rng.random(),
            0.5,
            guesser_preimage,
        )
        .await;

        global_state_lock
            .set_new_self_composed_tip(block1.clone(), composer_utxos)
            .await
            .unwrap();

        // Delete everything from monitored UTXO and from raw-hash keys.
        let mut global_state = global_state_lock.lock_guard_mut().await;
        {
            let monitored_utxos = global_state.wallet_state.wallet_db.monitored_utxos_mut();
            assert_eq!(
                5,
                monitored_utxos.len().await,
                "MUTXO must have genesis element, composer rewards, and guesser rewards"
            );
            monitored_utxos.pop().await;
            monitored_utxos.pop().await;
            monitored_utxos.pop().await;
            monitored_utxos.pop().await;
            monitored_utxos.pop().await;

            let guesser_preimage_keys = global_state.wallet_state.wallet_db.guesser_preimages();
            assert_eq!(
                1,
                guesser_preimage_keys.len().await,
                "Exactly Nonce-preimage must be stored to DB"
            );
            global_state.wallet_state.clear_raw_hash_keys().await;
            global_state
                .wallet_state
                .wallet_db
                .expected_utxos_mut()
                .clear()
                .await;

            assert!(
                global_state
                    .wallet_state
                    .wallet_db
                    .monitored_utxos()
                    .is_empty()
                    .await
            );
            assert!(
                global_state
                    .wallet_state
                    .wallet_db
                    .expected_utxos()
                    .is_empty()
                    .await
            );
            assert!(
                global_state
                    .wallet_state
                    .wallet_db
                    .guesser_preimages()
                    .is_empty()
                    .await
            );
            assert_eq!(
                0,
                global_state
                    .wallet_state
                    .get_known_raw_hash_lock_keys()
                    .count()
            );
        }

        // Recover the MUTXO from the recovery data, and verify that MUTXOs are restored
        // Also verify that this operation is idempotent by running it multiple times.
        for _ in 0..3 {
            global_state
                .restore_monitored_utxos_from_recovery_data()
                .await
                .unwrap();
            let monitored_utxos = global_state.wallet_state.wallet_db.monitored_utxos();
            assert_eq!(
                5,
                monitored_utxos.len().await,
                "MUTXO must have genesis elements and premine after recovery"
            );

            let mutxos = monitored_utxos.get_all().await;
            assert_eq!(
                Some((
                    genesis_block.hash(),
                    genesis_block.header().timestamp,
                    genesis_block.header().height
                )),
                mutxos[0].confirmed_in_block,
                "Historical information must be restored for premine UTXO"
            );

            for (i, mutxo) in mutxos.iter().enumerate().skip(1).take((1..=4).count()) {
                assert_eq!(
                    Some((
                        block1.hash(),
                        block1.header().timestamp,
                        block1.header().height
                    )),
                    mutxo.confirmed_in_block,
                    "Historical information must be restored for composer and guesser UTXOs, i={i}"
                );
            }

            // Verify that the restored MUTXOs have MSMPs, and that they're
            // valid.
            for mutxo in mutxos {
                let ms_item = Hash::hash(&mutxo.utxo);
                assert!(global_state
                    .chain
                    .light_state()
                    .mutator_set_accumulator_after()
                    .verify(
                        ms_item,
                        &mutxo.get_latest_membership_proof_entry().unwrap().1,
                    ));
                assert_eq!(
                    block1.hash(),
                    mutxo.get_latest_membership_proof_entry().unwrap().0,
                    "MUTXO must have the correct latest block digest value"
                );
            }

            // Verify that guesser-fee UTXO keys have also been restored.
            let cached_hash_lock_keys = global_state
                .wallet_state
                .get_known_raw_hash_lock_keys()
                .collect_vec();
            assert_eq!(
                vec![SpendingKey::RawHashLock(HashLockKey::from_preimage(
                    guesser_preimage
                ))],
                cached_hash_lock_keys,
                "Cached hash lock keys must match expected value after recovery"
            );
            let persisted_hash_lock_keys = global_state
                .wallet_state
                .wallet_db
                .guesser_preimages()
                .get_all()
                .await;
            assert_eq!(
                vec![guesser_preimage],
                persisted_hash_lock_keys,
                "Persisted hash lock keys must match expected value after recovery"
            );
        }
    }

    #[traced_test]
    #[tokio::test]
    async fn resync_ms_membership_proofs_simple_test() -> Result<()> {
        let mut rng = rand::rng();
        let network = Network::RegTest;
        let mut alice_state_lock = mock_genesis_global_state(
            network,
            2,
            WalletSecret::devnet_wallet(),
            cli_args::Args::default(),
        )
        .await;
        let mut alice = alice_state_lock.lock_guard_mut().await;

        let bob_wallet_secret = WalletSecret::new_random();
        let bob_key = bob_wallet_secret.nth_generation_spending_key(0);

        // 1. Create new block 1 and store it
        let genesis_block = Block::genesis(network);
        let launch = genesis_block.kernel.header.timestamp;
        let seven_months = Timestamp::months(7);
        let (mock_block_1a, _) = make_mock_block(&genesis_block, None, bob_key, rng.random()).await;
        {
            alice
                .chain
                .archival_state_mut()
                .write_block_as_tip(&mock_block_1a)
                .await?;
        }

        // Verify that Alice has a monitored UTXO (from genesis)
        assert!(!alice
            .get_wallet_status_for_tip()
            .await
            .synced_unspent_available_amount(launch + seven_months)
            .is_zero());
        assert!(!alice.get_balance_history().await.is_empty());

        // Verify that this is unsynced with mock_block_1a
        assert!(alice.wallet_state.is_synced_to(genesis_block.hash()).await);
        assert!(!alice.wallet_state.is_synced_to(mock_block_1a.hash()).await);

        // Call resync
        alice
            .resync_membership_proofs_from_stored_blocks(mock_block_1a.hash())
            .await
            .unwrap();

        // Verify that it is synced
        assert!(alice.wallet_state.is_synced_to(mock_block_1a.hash()).await);

        // Verify that MPs are valid
        assert!(wallet_state_has_all_valid_mps(&alice.wallet_state, &mock_block_1a).await);

        Ok(())
    }

    #[traced_test]
    #[tokio::test]
    async fn resync_ms_membership_proofs_fork_test() -> Result<()> {
        let network = Network::Main;
        let mut rng = rand::rng();

        let mut alice = mock_genesis_global_state(
            network,
            2,
            WalletSecret::devnet_wallet(),
            cli_args::Args::default(),
        )
        .await;
        let mut alice = alice.lock_guard_mut().await;
        let alice_key = alice
            .wallet_state
            .wallet_secret
            .nth_generation_spending_key(0);

        // 1. Create new block 1a where we receive a coinbase UTXO, store it
        let genesis_block = alice.chain.archival_state().get_tip().await;
        let (mock_block_1a, composer_expected_utxos_1a) =
            make_mock_block(&genesis_block, None, alice_key, rng.random()).await;
        alice
            .wallet_state
            .add_expected_utxos(composer_expected_utxos_1a)
            .await;
        alice.set_new_tip(mock_block_1a.clone()).await.unwrap();

        // Verify that wallet has monitored UTXOs, 1 from genesis, 2 from
        // block 1a.
        assert_eq!(
            3,
            alice
                .wallet_state
                .get_wallet_status(
                    mock_block_1a.hash(),
                    &mock_block_1a.mutator_set_accumulator_after()
                )
                .await
                .synced_unspent
                .len()
        );
        assert_eq!(3, alice.get_balance_history().await.len());

        // Make a new fork from genesis that makes us lose the composer UTXOs
        // of block 1a.
        let bob_wallet_secret = WalletSecret::new_random();
        let bob_key = bob_wallet_secret.nth_generation_spending_key(0);
        let mut parent_block = genesis_block;
        for _ in 0..5 {
            let (next_block, _) = make_mock_block(&parent_block, None, bob_key, rng.random()).await;
            alice.set_new_tip(next_block.clone()).await.unwrap();
            parent_block = next_block;
        }

        // Call resync which fails to sync the UTXO that was abandoned when block 1a was abandoned
        alice
            .resync_membership_proofs_from_stored_blocks(parent_block.hash())
            .await
            .unwrap();

        // Verify that two MUTXOs are unsynced, and that 1 (from genesis) is synced
        let alice_wallet_status_after_reorg = alice
            .wallet_state
            .get_wallet_status(
                parent_block.hash(),
                &parent_block.mutator_set_accumulator_after(),
            )
            .await;
        assert_eq!(1, alice_wallet_status_after_reorg.synced_unspent.len());
        assert_eq!(2, alice_wallet_status_after_reorg.unsynced.len());

        // Verify that the MUTXO from block 1a is considered abandoned, and that the one from
        // genesis block is not.
        let monitored_utxos = alice.wallet_state.wallet_db.monitored_utxos();
        assert!(
            !monitored_utxos
                .get(0)
                .await
                .was_abandoned(alice.chain.archival_state())
                .await
        );
        assert!(
            monitored_utxos
                .get(1)
                .await
                .was_abandoned(alice.chain.archival_state())
                .await
        );

        Ok(())
    }

    #[tokio::test]
    async fn resync_ms_membership_proofs_across_stale_fork() {
        /// Create 3 branches and return them in an array.
        ///
        /// First two branches share common ancestor `first_for_0_1`, last
        /// branch starts from `first_for_2`. All branches have the same length.
        ///
        /// Factored out to parallel function to make this test run faster.
        async fn make_3_branches(
            first_for_0_1: &Block,
            first_for_2: &Block,
            num_blocks_per_branch: usize,
            cb_recipient: &GenerationSpendingKey,
        ) -> [Vec<Block>; 3] {
            let mut final_ret = Vec::with_capacity(3);
            for i in 0..3 {
                let mut rng = rand::rng();
                let mut ret = Vec::with_capacity(num_blocks_per_branch);

                let mut block = if i < 2 {
                    first_for_0_1.to_owned()
                } else {
                    first_for_2.to_owned()
                };
                for _ in 0..num_blocks_per_branch {
                    let (next_block, _) =
                        make_mock_block(&block, None, cb_recipient.to_owned(), rng.random()).await;
                    ret.push(next_block.clone());
                    block = next_block;
                }

                final_ret.push(ret);
            }

            final_ret.try_into().unwrap()
        }

        let network = Network::Main;
        let mut rng = rand::rng();
        let mut alice = mock_genesis_global_state(
            network,
            2,
            WalletSecret::devnet_wallet(),
            cli_args::Args::default(),
        )
        .await;
        let mut alice = alice.lock_guard_mut().await;
        let alice_key = alice
            .wallet_state
            .wallet_secret
            .nth_generation_spending_key(0);
        let bob_secret = WalletSecret::new_random();
        let bob_key = bob_secret.nth_generation_spending_key(0);

        // 1. Create new block 1 where Alice receives two composer UTXOs, store it.
        let genesis_block = alice.chain.archival_state().get_tip().await;
        let (block_1, alice_composer_expected_utxos_1) =
            make_mock_block(&genesis_block, None, alice_key, rng.random()).await;
        {
            alice
                .wallet_state
                .add_expected_utxos(alice_composer_expected_utxos_1)
                .await;
            alice.set_new_tip(block_1.clone()).await.unwrap();

            // Verify that composer UTXOs were recorded
            assert_eq!(
                3,
                alice
                    .wallet_state
                    .get_wallet_status(block_1.hash(), &block_1.mutator_set_accumulator_after())
                    .await
                    .synced_unspent
                    .len()
            );
        }

        let [a_blocks, b_blocks, c_blocks] =
            make_3_branches(&block_1, &genesis_block, 60, &bob_key).await;

        // Add 60 blocks on top of 1, *not* mined by Alice
        let fork_a_block = a_blocks.last().unwrap().to_owned();
        for branch_block in a_blocks.into_iter() {
            alice.set_new_tip(branch_block).await.unwrap();
        }

        // Verify that all both MUTXOs have synced MPs
        let wallet_status_on_a_fork = alice
            .wallet_state
            .get_wallet_status(
                fork_a_block.hash(),
                &fork_a_block.mutator_set_accumulator_after(),
            )
            .await;

        assert_eq!(3, wallet_status_on_a_fork.synced_unspent.len());

        // Fork away from the "a" chain to the "b" chain, with block 1 as LUCA
        let fork_b_block = b_blocks.last().unwrap().to_owned();
        for branch_block in b_blocks.into_iter() {
            alice.set_new_tip(branch_block).await.unwrap();
        }

        // Verify that there are zero MUTXOs with synced MPs
        let alice_wallet_status_on_b_fork_before_resync = alice
            .wallet_state
            .get_wallet_status(
                fork_b_block.hash(),
                &fork_b_block.mutator_set_accumulator_after(),
            )
            .await;
        assert_eq!(
            0,
            alice_wallet_status_on_b_fork_before_resync
                .synced_unspent
                .len()
        );
        assert_eq!(
            3,
            alice_wallet_status_on_b_fork_before_resync.unsynced.len()
        );

        // Run the resync and verify that MPs are synced
        alice
            .resync_membership_proofs_from_stored_blocks(fork_b_block.hash())
            .await
            .unwrap();
        let wallet_status_on_b_fork_after_resync = alice
            .wallet_state
            .get_wallet_status(
                fork_b_block.hash(),
                &fork_b_block.mutator_set_accumulator_after(),
            )
            .await;
        assert_eq!(3, wallet_status_on_b_fork_after_resync.synced_unspent.len());
        assert_eq!(0, wallet_status_on_b_fork_after_resync.unsynced.len());

        // Make a new chain c with genesis block as LUCA. Verify that the genesis UTXO can be synced
        // to this new chain
        let fork_c_block = c_blocks.last().unwrap().to_owned();
        for branch_block in c_blocks.into_iter() {
            alice.set_new_tip(branch_block).await.unwrap();
        }

        // Verify that there are zero MUTXOs with synced MPs
        let alice_wallet_status_on_c_fork_before_resync = alice
            .wallet_state
            .get_wallet_status(
                fork_c_block.hash(),
                &fork_c_block.mutator_set_accumulator_after(),
            )
            .await;
        assert_eq!(
            0,
            alice_wallet_status_on_c_fork_before_resync
                .synced_unspent
                .len()
        );
        assert_eq!(
            3,
            alice_wallet_status_on_c_fork_before_resync.unsynced.len()
        );

        // Run the resync and verify that UTXO from genesis is synced, but that
        // UTXO from 1a is not synced.
        alice
            .resync_membership_proofs_from_stored_blocks(fork_c_block.hash())
            .await
            .unwrap();
        let alice_ws_c_after_resync = alice
            .wallet_state
            .get_wallet_status(
                fork_c_block.hash(),
                &fork_c_block.mutator_set_accumulator_after(),
            )
            .await;
        assert_eq!(1, alice_ws_c_after_resync.synced_unspent.len());
        assert_eq!(2, alice_ws_c_after_resync.unsynced.len());

        // Also check that UTXO from 1a is considered abandoned
        let alice_mutxos = alice.wallet_state.wallet_db.monitored_utxos();
        assert!(
            !alice_mutxos
                .get(0)
                .await
                .was_abandoned(alice.chain.archival_state())
                .await
        );
        for i in 1..=2 {
            assert!(
                alice_mutxos
                    .get(i)
                    .await
                    .was_abandoned(alice.chain.archival_state())
                    .await
            );
        }
    }

    #[traced_test]
    #[tokio::test]
    async fn flaky_mutator_set_test() {
        // Test various parts of the state update when a block contains multiple inputs and outputs
        // Scenario: Three parties: Alice, Bob, and Premine Receiver, mine blocks and pass coins
        // around.

        let mut rng: StdRng = StdRng::seed_from_u64(0x03ce12210c467f93u64);
        let network = Network::Main;

        let mut premine_receiver = mock_genesis_global_state(
            network,
            3,
            WalletSecret::devnet_wallet(),
            cli_args::Args::default(),
        )
        .await;
        let genesis_spending_key = premine_receiver
            .lock_guard()
            .await
            .wallet_state
            .wallet_secret
            .nth_generation_spending_key_for_tests(0);

        let wallet_secret_alice = WalletSecret::new_pseudorandom(rng.random());
        let alice_spending_key = wallet_secret_alice.nth_generation_spending_key_for_tests(0);
        let mut alice =
            mock_genesis_global_state(network, 3, wallet_secret_alice, cli_args::Args::default())
                .await;

        let wallet_secret_bob = WalletSecret::new_pseudorandom(rng.random());
        let bob_spending_key = wallet_secret_bob.nth_generation_spending_key_for_tests(0);
        let mut bob =
            mock_genesis_global_state(network, 3, wallet_secret_bob, cli_args::Args::default())
                .await;

        let genesis_block = Block::genesis(network);
        let in_seven_months = genesis_block.kernel.header.timestamp + Timestamp::months(7);
        let in_eight_months = in_seven_months + Timestamp::months(1);

        let guesser_fraction = 0f64;
        let (coinbase_transaction, coinbase_expected_utxos) = make_coinbase_transaction_from_state(
            &genesis_block,
            &premine_receiver,
            guesser_fraction,
            in_seven_months,
            TxProvingCapability::SingleProof,
            TritonVmJobPriority::Normal.into(),
        )
        .await
        .unwrap();

        assert!(coinbase_transaction.is_valid().await);
        assert!(coinbase_transaction
            .is_confirmable_relative_to(&genesis_block.mutator_set_accumulator_after()));

        // Send two outputs each to Alice and Bob, from genesis receiver
        let sender_randomness: Digest = rng.random();
        let tx_outputs_for_alice = vec![
            TxOutput::onchain_native_currency(
                NativeCurrencyAmount::coins(1),
                sender_randomness,
                alice_spending_key.to_address().into(),
                false,
            ),
            TxOutput::onchain_native_currency(
                NativeCurrencyAmount::coins(2),
                sender_randomness,
                alice_spending_key.to_address().into(),
                false,
            ),
        ];

        // Two outputs for Bob
        let tx_outputs_for_bob = vec![
            TxOutput::onchain_native_currency(
                NativeCurrencyAmount::coins(3),
                sender_randomness,
                bob_spending_key.to_address().into(),
                false,
            ),
            TxOutput::onchain_native_currency(
                NativeCurrencyAmount::coins(4),
                sender_randomness,
                bob_spending_key.to_address().into(),
                false,
            ),
        ];

        let fee = NativeCurrencyAmount::one();
        let genesis_key = premine_receiver
            .lock_guard_mut()
            .await
            .wallet_state
            .next_unused_spending_key(KeyType::Generation)
            .await
            .unwrap();
        let (tx_to_alice_and_bob, _, maybe_change_output) = premine_receiver
            .lock_guard()
            .await
            .create_transaction_with_prover_capability(
                [tx_outputs_for_alice.clone(), tx_outputs_for_bob.clone()]
                    .concat()
                    .into(),
                genesis_key,
                UtxoNotificationMedium::OffChain,
                fee,
                in_seven_months,
                TxProvingCapability::SingleProof,
                &TritonVmJobQueue::dummy(),
            )
            .await
            .unwrap();
        let Some(change_output) = maybe_change_output else {
            panic!("Expected change output to genesis receiver");
        };

        assert!(tx_to_alice_and_bob.is_valid().await);
        assert!(tx_to_alice_and_bob
            .is_confirmable_relative_to(&genesis_block.mutator_set_accumulator_after()));

        // Expect change output
        premine_receiver
            .global_state_lock
            .lock_guard_mut()
            .await
            .wallet_state
            .add_expected_utxo(ExpectedUtxo::new(
                change_output.utxo(),
                change_output.sender_randomness(),
                genesis_key.privacy_preimage().unwrap(),
                UtxoNotifier::Myself,
            ))
            .await;

        let block_transaction = tx_to_alice_and_bob
            .merge_with(
                coinbase_transaction,
                Default::default(),
                &TritonVmJobQueue::dummy(),
                TritonVmJobPriority::default().into(),
            )
            .await
            .unwrap();
        assert!(block_transaction.is_valid().await);
        assert!(block_transaction
            .is_confirmable_relative_to(&genesis_block.mutator_set_accumulator_after()));

        let block_1 = Block::compose(
            &genesis_block,
            block_transaction,
            in_seven_months,
            None,
            &TritonVmJobQueue::dummy(),
            TritonVmJobPriority::default().into(),
        )
        .await
        .unwrap();

        assert!(block_1.is_valid(&genesis_block, in_seven_months).await);

        println!("Accumulated transaction into block_1.");
        println!(
            "Transaction has {} inputs (removal records) and {} outputs (addition records)",
            block_1.kernel.body.transaction_kernel.inputs.len(),
            block_1.kernel.body.transaction_kernel.outputs.len()
        );

        // Update states with `block_1`
        let expected_utxos_for_alice = alice
            .lock_guard()
            .await
            .wallet_state
            .extract_expected_utxos(tx_outputs_for_alice.into(), UtxoNotifier::Cli);
        alice
            .lock_guard_mut()
            .await
            .wallet_state
            .add_expected_utxos(expected_utxos_for_alice)
            .await;

        let expected_utxos_for_bob_1 = bob
            .lock_guard()
            .await
            .wallet_state
            .extract_expected_utxos(tx_outputs_for_bob.into(), UtxoNotifier::Cli);
        bob.lock_guard_mut()
            .await
            .wallet_state
            .add_expected_utxos(expected_utxos_for_bob_1)
            .await;

        premine_receiver
            .set_new_self_composed_tip(
                block_1.clone(),
                coinbase_expected_utxos
                    .into_iter()
                    .map(|expected_utxo| {
                        ExpectedUtxo::new(
                            expected_utxo.utxo,
                            expected_utxo.sender_randomness,
                            genesis_spending_key.privacy_preimage(),
                            UtxoNotifier::OwnMinerComposeBlock,
                        )
                    })
                    .collect_vec(),
            )
            .await
            .unwrap();

        for state_lock in [&mut alice, &mut bob] {
            state_lock.set_new_tip(block_1.clone()).await.unwrap();
        }

        assert_eq!(
            4,
            premine_receiver
                .lock_guard_mut()
                .await
                .wallet_state
                .wallet_db
                .monitored_utxos()
                .len().await, "Premine receiver must have 4 monitored UTXOs after block 1: change from transaction, 2 coinbases from block 1, and premine UTXO"
        );

        assert_eq!(
            NativeCurrencyAmount::coins(3),
            alice
                .lock_guard()
                .await
                .get_wallet_status_for_tip()
                .await
                .synced_unspent_available_amount(in_seven_months)
        );
        assert_eq!(
            NativeCurrencyAmount::coins(7),
            bob.lock_guard()
                .await
                .get_wallet_status_for_tip()
                .await
                .synced_unspent_available_amount(in_seven_months)
        );
        // TODO: No idea why this isn't working.
        // {
        //     let expected = NativeCurrencyAmount::coins(110);
        //     let got = premine_receiver
        //         .lock_guard()
        //         .await
        //         .get_wallet_status_for_tip()
        //         .await
        //         .synced_unspent_available_amount(in_seven_months);
        //     assert_eq!(
        //         expected, got,
        //         "premine receiver's balance should be 110: mining reward + premine - sent - fee + fee. Expected: {expected:?}\nGot: {got}"
        //     );
        // }

        // Make two transactions: Alice sends two UTXOs to Genesis and Bob sends three UTXOs to genesis
        let tx_outputs_from_alice = vec![
            TxOutput::onchain_native_currency(
                NativeCurrencyAmount::coins(1),
                rng.random(),
                genesis_spending_key.to_address().into(),
                false,
            ),
            TxOutput::onchain_native_currency(
                NativeCurrencyAmount::coins(1),
                rng.random(),
                genesis_spending_key.to_address().into(),
                false,
            ),
        ];
        // About prover capability: we need `SingleProof` transactions for the
        // miner to merge them later. The thing being tested here is that the
        // state is being updated correctly with new blocks; not the
        // use-`ProofCollection`-instead-of-`SingleProof` functionality.
        // Weaker machines need to use the proof server.
        let (tx_from_alice, _, maybe_change_for_alice) = alice
            .lock_guard()
            .await
            .create_transaction_with_prover_capability(
                tx_outputs_from_alice.clone().into(),
                alice_spending_key.into(),
                UtxoNotificationMedium::OffChain,
                NativeCurrencyAmount::coins(1),
                in_seven_months,
                TxProvingCapability::SingleProof,
                &TritonVmJobQueue::dummy(),
            )
            .await
            .unwrap();
        assert!(
            maybe_change_for_alice.is_none(),
            "No change for Alice as she spent it all"
        );

        assert!(tx_from_alice.is_valid().await);
        assert!(tx_from_alice.is_confirmable_relative_to(&block_1.mutator_set_accumulator_after()));

        // make bob's transaction
        let tx_outputs_from_bob = vec![
            TxOutput::onchain_native_currency(
                NativeCurrencyAmount::coins(2),
                rng.random(),
                genesis_spending_key.to_address().into(),
                false,
            ),
            TxOutput::onchain_native_currency(
                NativeCurrencyAmount::coins(2),
                rng.random(),
                genesis_spending_key.to_address().into(),
                false,
            ),
            TxOutput::onchain_native_currency(
                NativeCurrencyAmount::coins(2),
                rng.random(),
                genesis_spending_key.to_address().into(),
                false,
            ),
        ];
        let (tx_from_bob, _, maybe_change_for_bob) = bob
            .lock_guard()
            .await
            .create_transaction_with_prover_capability(
                tx_outputs_from_bob.clone().into(),
                bob_spending_key.into(),
                UtxoNotificationMedium::OffChain,
                NativeCurrencyAmount::coins(1),
                in_seven_months,
                TxProvingCapability::SingleProof,
                &TritonVmJobQueue::dummy(),
            )
            .await
            .unwrap();

        assert!(
            maybe_change_for_bob.is_none(),
            "No change for Bob as he spent it all"
        );

        assert!(tx_from_bob.is_valid().await);
        assert!(tx_from_bob.is_confirmable_relative_to(&block_1.mutator_set_accumulator_after()));

        // Make block_2 with tx that contains:
        // - 4 inputs: 2 from Alice and 2 from Bob
        // - 7 outputs: 2 from Alice to Genesis, 3 from Bob to Genesis, and 2 coinbases
        let (coinbase_transaction2, _expected_utxo) = make_coinbase_transaction_from_state(
            &premine_receiver
                .global_state_lock
                .lock_guard()
                .await
                .chain
                .light_state()
                .clone(),
            &premine_receiver,
            guesser_fraction,
            in_seven_months,
            TxProvingCapability::SingleProof,
            TritonVmJobPriority::Normal.into(),
        )
        .await
        .unwrap();
        assert!(coinbase_transaction2.is_valid().await);
        assert!(coinbase_transaction2
            .is_confirmable_relative_to(&block_1.mutator_set_accumulator_after()));

        let block_transaction2 = coinbase_transaction2
            .merge_with(
                tx_from_alice,
                Default::default(),
                &TritonVmJobQueue::dummy(),
                TritonVmJobPriority::default().into(),
            )
            .await
            .unwrap()
            .merge_with(
                tx_from_bob,
                Default::default(),
                &TritonVmJobQueue::dummy(),
                TritonVmJobPriority::default().into(),
            )
            .await
            .unwrap();
        assert!(block_transaction2.is_valid().await);
        assert!(
            block_transaction2.is_confirmable_relative_to(&block_1.mutator_set_accumulator_after())
        );

        let block_2 = Block::compose(
            &block_1,
            block_transaction2,
            in_eight_months,
            None,
            &TritonVmJobQueue::dummy(),
            TritonVmJobPriority::default().into(),
        )
        .await
        .unwrap();
        assert!(block_2.is_valid(&block_1, in_eight_months).await);

        assert_eq!(4, block_2.kernel.body.transaction_kernel.inputs.len());
        assert_eq!(7, block_2.kernel.body.transaction_kernel.outputs.len());
    }

    #[traced_test]
    #[tokio::test]
    async fn mock_global_state_is_valid() {
        // Verify that the states, not just the blocks, are valid.

        let network = Network::Main;
        let mut global_state_lock = mock_genesis_global_state(
            network,
            2,
            WalletSecret::devnet_wallet(),
            cli_args::Args::default(),
        )
        .await;
        let genesis_block = Block::genesis(network);
        let now = genesis_block.kernel.header.timestamp + Timestamp::hours(1);

        let block1 = fake_valid_successor_for_tests(&genesis_block, now, Default::default()).await;

        global_state_lock.set_new_tip(block1).await.unwrap();

        assert!(
            global_state_lock
                .lock_guard()
                .await
                .chain
                .light_state()
                .is_valid(&genesis_block, now)
                .await,
            "light state tip must be a valid block"
        );
        assert!(
            global_state_lock
                .lock_guard()
                .await
                .chain
                .archival_state()
                .get_tip()
                .await
                .is_valid(&genesis_block, now)
                .await,
            "archival state tip must be a valid block"
        );
    }

    #[tokio::test]
    async fn favor_incoming_block_proposal_test() {
        async fn block1_proposal(
            guesser_fraction: f64,
            global_state_lock: &GlobalStateLock,
        ) -> Block {
            let genesis_block = Block::genesis(global_state_lock.cli().network);
            let timestamp = genesis_block.header().timestamp + Timestamp::hours(1);
            let (cb, _) = make_coinbase_transaction_from_state(
                &genesis_block,
                global_state_lock,
                guesser_fraction,
                timestamp,
                TxProvingCapability::PrimitiveWitness,
                (TritonVmJobPriority::Normal, None).into(),
            )
            .await
            .unwrap();

            Block::block_template_invalid_proof(&genesis_block, cb, timestamp, None)
        }

        let network = Network::Main;
        let mut global_state_lock = mock_genesis_global_state(
            network,
            2,
            WalletSecret::devnet_wallet(),
            cli_args::Args::default_with_network(network),
        )
        .await;
        let small_guesser_fraction = block1_proposal(0.1, &global_state_lock).await;
        let big_guesser_fraction = block1_proposal(0.5, &global_state_lock).await;

        let mut state = global_state_lock.global_state_lock.lock_guard_mut().await;
        assert!(
            state
                .favor_incoming_block_proposal(
                    small_guesser_fraction.header().height,
                    small_guesser_fraction.total_guesser_reward()
                )
                .is_ok(),
            "Must favor low guesser fee over none"
        );

        state.block_proposal = BlockProposal::foreign_proposal(small_guesser_fraction.clone());
        assert!(
            state
                .favor_incoming_block_proposal(
                    big_guesser_fraction.header().height,
                    big_guesser_fraction.total_guesser_reward()
                )
                .is_ok(),
            "Must favor big guesser fee over low"
        );

        state.block_proposal = BlockProposal::foreign_proposal(big_guesser_fraction.clone());
        assert_eq!(
            BlockProposalRejectError::InsufficientFee {
                current: Some(big_guesser_fraction.total_guesser_reward()),
                received: big_guesser_fraction.total_guesser_reward()
            },
            state
                .favor_incoming_block_proposal(
                    big_guesser_fraction.header().height,
                    big_guesser_fraction.total_guesser_reward()
                )
                .unwrap_err(),
            "Must favor existing over incoming equivalent"
        );
    }

    mod state_update_on_reorganizations {
        use twenty_first::prelude::Mmr;

        use super::*;

        async fn assert_correct_global_state(
            global_state: &GlobalState,
            expected_tip: Block,
            expected_parent: Block,
            expected_num_blocks_at_tip_height: usize,
            expected_num_spendable_utxos: usize,
        ) {
            // Verifying light state integrity
            let expected_tip_digest = expected_tip.hash();
            assert_eq!(expected_tip_digest, global_state.chain.light_state().hash());

            // Peeking into archival state
            assert_eq!(
                expected_tip_digest,
                global_state
                    .chain
                    .archival_state()
                    .archival_mutator_set
                    .get_sync_label()
                    .await,
                "Archival state must have expected sync-label"
            );
            assert_eq!(
                expected_tip.mutator_set_accumulator_after(),
                global_state
                    .chain
                    .archival_state()
                    .archival_mutator_set
                    .ams()
                    .accumulator()
                    .await,
                "Archival mutator set must match that in expected tip"
            );

            assert_eq!(
                expected_tip_digest,
                global_state
                    .chain
                    .archival_state()
                    .get_block(expected_tip.hash())
                    .await
                    .unwrap()
                    .unwrap()
                    .hash(),
                "Expected block must be returned"
            );

            assert_eq!(
                expected_tip_digest,
                global_state
                    .chain
                    .archival_state()
                    .archival_block_mmr
                    .ammr()
                    .get_latest_leaf()
                    .await
                    .unwrap(),
                "Latest leaf in archival block MMR must match expected block"
            );

            // Verify that archival-block MMR matches that of block
            {
                let mut expected_archival_block_mmr_value =
                    expected_tip.body().block_mmr_accumulator.clone();
                expected_archival_block_mmr_value.append(expected_tip_digest);
                assert_eq!(
                    expected_archival_block_mmr_value,
                    global_state
                        .chain
                        .archival_state()
                        .archival_block_mmr
                        .ammr()
                        .to_accumulator_async()
                        .await,
                    "archival block-MMR must match that in tip after adding tip digest"
                );
            }

            let tip_height = expected_tip.header().height;
            assert_eq!(
                expected_num_blocks_at_tip_height,
                global_state
                    .chain
                    .archival_state()
                    .block_height_to_block_digests(tip_height)
                    .await
                    .len(),
                "Exactly {expected_num_blocks_at_tip_height} blocks at height must be known"
            );

            let expected_parent_digest = expected_parent.hash();
            assert_eq!(
                expected_parent_digest,
                global_state
                    .chain
                    .archival_state()
                    .get_tip_parent()
                    .await
                    .unwrap()
                    .hash()
            );

            // Peek into wallet
            let tip_msa = expected_tip.mutator_set_accumulator_after().clone();
            let mutxos = global_state
                .wallet_state
                .wallet_db
                .monitored_utxos()
                .get_all()
                .await;
            let mut mutxos_on_tip = vec![];
            assert!(
                mutxos.iter().all(|x| x.confirmed_in_block.is_some()),
                "All monitored UTXOs must be mined."
            );
            for mutxo in mutxos {
                if !mutxo
                    .was_abandoned(global_state.chain.archival_state())
                    .await
                {
                    mutxos_on_tip.push(mutxo);
                }
            }

            assert_eq!(
                expected_num_spendable_utxos,
                mutxos_on_tip.len(),
                "Number of monitored UTXOS at height {tip_height} must match expected value of {expected_num_spendable_utxos}"
            );
            assert!(
                mutxos_on_tip.iter().all(|mutxo| tip_msa.verify(
                    Tip5::hash(&mutxo.utxo),
                    &mutxo
                        .get_membership_proof_for_block(expected_tip.hash())
                        .unwrap()
                )),
                "All wallet's membership proofs must still be valid"
            );
        }

        #[traced_test]
        #[tokio::test]
        async fn can_handle_deep_reorganization() {
            // Mine 60 blocks, then switch to a new chain branching off from
            // genesis block. Verify that state is integral after each block.
            let network = Network::Main;
            let mut rng = rand::rng();
            let genesis_block = Block::genesis(network);
            let wallet_secret = WalletSecret::devnet_wallet();
            let spending_key = wallet_secret.nth_generation_spending_key(0);

            let mut global_state_lock = mock_genesis_global_state(
                network,
                2,
                wallet_secret.clone(),
                cli_args::Args::default(),
            )
            .await;

            // Branch A
            let mut previous_block = genesis_block.clone();
            for block_height in 1..60 {
                let (next_block, expected) =
                    make_mock_block(&previous_block, None, spending_key, rng.random()).await;
                global_state_lock
                    .set_new_self_composed_tip(next_block.clone(), expected)
                    .await
                    .unwrap();
                let global_state = global_state_lock.lock_guard().await;
                assert_correct_global_state(
                    &global_state,
                    next_block.clone(),
                    previous_block.clone(),
                    1,
                    2 * block_height + 1,
                )
                .await;
                previous_block = next_block;
            }

            // Branch B
            previous_block = genesis_block.clone();
            for block_height in 1..60 {
                let (next_block, expected) =
                    make_mock_block(&previous_block, None, spending_key, rng.random()).await;
                global_state_lock
                    .set_new_self_composed_tip(next_block.clone(), expected)
                    .await
                    .unwrap();

                // Resync membership proofs after block 1 on branch B, otherwise
                // the genesis block's premine UTXO will not have a valid
                // membership proof.
                let mut global_state = global_state_lock.lock_guard_mut().await;
                if block_height == 1 {
                    global_state.resync_membership_proofs().await.unwrap();
                }

                assert_correct_global_state(
                    &global_state,
                    next_block.clone(),
                    previous_block.clone(),
                    2,
                    2 * block_height + 1,
                )
                .await;
                previous_block = next_block;
            }
        }

        #[traced_test]
        #[tokio::test]
        async fn can_store_block_without_marking_it_as_tip_1_block() {
            // Verify that [GlobalState::store_block_not_tip] stores block
            // correctly, and that [GlobalState::set_new_tip] can be used to
            // build upon blocks stored through the former method.
            let network = Network::Main;
            let mut rng = rand::rng();
            let genesis_block = Block::genesis(network);
            let wallet_secret = WalletSecret::new_random();

            let mut alice = mock_genesis_global_state(
                network,
                2,
                wallet_secret.clone(),
                cli_args::Args::default(),
            )
            .await;

            let mut alice = alice.global_state_lock.lock_guard_mut().await;
            assert_eq!(genesis_block.hash(), alice.chain.light_state().hash());

            let cb_key = WalletSecret::new_random().nth_generation_spending_key(0);
            let (block_1, _) = make_mock_block(&genesis_block, None, cb_key, rng.random()).await;

            alice.store_block_not_tip(block_1.clone()).await.unwrap();
            assert_eq!(
                genesis_block.hash(),
                alice.chain.light_state().hash(),
                "method may not update light state's tip"
            );
            assert_eq!(
                genesis_block.hash(),
                alice.chain.archival_state().get_tip().await.hash(),
                "method may not update archival state's tip"
            );

            alice.set_new_tip(block_1.clone()).await.unwrap();
            assert_correct_global_state(&alice, block_1.clone(), genesis_block, 1, 0).await;
        }

        /// Return a list of (Block, parent) pairs, of length N.
        async fn chain_of_blocks_and_parents(
            network: Network,
            length: usize,
        ) -> Vec<(Block, Block)> {
            let mut rng = rand::rng();
            let cb_key = WalletSecret::new_random().nth_generation_spending_key(0);
            let mut parent = Block::genesis(network);
            let mut chain = vec![];
            for _ in 0..length {
                let (block, _) = make_mock_block(&parent, None, cb_key, rng.random()).await;
                chain.push((block.clone(), parent.clone()));
                parent = block;
            }

            chain
        }

        #[traced_test]
        #[tokio::test]
        async fn can_jump_to_new_tip_over_blocks_that_were_never_tips() {
            let network = Network::Main;
            let wallet_secret = WalletSecret::new_random();
            let mut alice = mock_genesis_global_state(
                network,
                2,
                wallet_secret.clone(),
                cli_args::Args::default(),
            )
            .await;
            let mut alice = alice.global_state_lock.lock_guard_mut().await;

            let a_length = 12;
            let chain_a = chain_of_blocks_and_parents(network, a_length).await;
            for (block, _) in chain_a.iter() {
                alice.set_new_tip(block.to_owned()).await.unwrap();
            }

            let chain_a_tip = &chain_a[a_length - 1].0;
            let chain_a_tip_parent = &chain_a[a_length - 1].1;
            assert_correct_global_state(
                &alice,
                chain_a_tip.to_owned(),
                chain_a_tip_parent.to_owned(),
                1,
                0,
            )
            .await;

            // Store all blocks from a new chain, except the last, without
            // marking any of them as tips.  Verify no change in tip.
            let b_length = 15;
            let chain_b = chain_of_blocks_and_parents(network, b_length).await;
            for (block, _) in chain_b.iter().take(b_length - 1) {
                alice.store_block_not_tip(block.clone()).await.unwrap();
            }
            assert_correct_global_state(
                &alice,
                chain_a_tip.to_owned(),
                chain_a_tip_parent.to_owned(),
                2,
                0,
            )
            .await;

            // Set chain B's last block to tip to verify that all the stored
            // blocks from chain B can be used to connect it to LUCA, which in
            // this case is genesis block.
            let chain_b_tip = &chain_b[b_length - 1].0;
            let chain_b_tip_parent = &chain_b[b_length - 1].1;
            alice.set_new_tip(chain_b_tip.to_owned()).await.unwrap();
            assert_correct_global_state(
                &alice,
                chain_b_tip.to_owned(),
                chain_b_tip_parent.to_owned(),
                1,
                0,
            )
            .await;
        }

        #[traced_test]
        #[tokio::test]
        async fn reorganization_with_blocks_that_were_never_tips_n_blocks_deep() {
            // Verify that [GlobalState::store_block_not_tip] stores block
            // correctly, and that [GlobalState::set_new_tip] can be used to
            // build upon blocks stored through the former method.
            let network = Network::Main;
            let genesis_block = Block::genesis(network);
            let wallet_secret = WalletSecret::new_random();

            for depth in 1..=4 {
                let mut alice = mock_genesis_global_state(
                    network,
                    2,
                    wallet_secret.clone(),
                    cli_args::Args::default(),
                )
                .await;
                let mut alice = alice.global_state_lock.lock_guard_mut().await;
                assert_eq!(genesis_block.hash(), alice.chain.light_state().hash());
                let chain_a = chain_of_blocks_and_parents(network, depth).await;
                let chain_b = chain_of_blocks_and_parents(network, depth).await;
                let blocks_and_parents = [chain_a, chain_b].concat();
                for (block, _) in blocks_and_parents.iter() {
                    alice.store_block_not_tip(block.clone()).await.unwrap();
                    assert_eq!(
                        genesis_block.hash(),
                        alice.chain.light_state().hash(),
                        "method may not update light state's tip, depth = {depth}"
                    );
                    assert_eq!(
                        genesis_block.hash(),
                        alice.chain.archival_state().get_tip().await.hash(),
                        "method may not update archival state's tip, depth = {depth}"
                    );
                }

                // Loop over all blocks and verify that all can be marked as
                // tip, resulting in a consistent, correct state.
                for (block, parent) in blocks_and_parents.iter() {
                    alice.set_new_tip(block.clone()).await.unwrap();
                    assert_correct_global_state(&alice, block.clone(), parent.to_owned(), 2, 0)
                        .await;
                }
            }
        }

        #[traced_test]
        #[tokio::test]
        async fn set_new_tip_can_roll_back() {
            // Verify that [GlobalState::set_new_tip] works for rolling back the
            // blockchain to a previous block.
            let network = Network::Main;
            let mut rng = rand::rng();
            let genesis_block = Block::genesis(network);
            let wallet_secret = WalletSecret::devnet_wallet();
            let spending_key = wallet_secret.nth_generation_spending_key(0);

            let (block_1a, composer_expected_utxos_1a) =
                make_mock_block(&genesis_block, None, spending_key, rng.random()).await;
            let (block_2a, composer_expected_utxos_2a) =
                make_mock_block(&block_1a, None, spending_key, rng.random()).await;
            let (block_3a, composer_expected_utxos_3a) =
                make_mock_block(&block_2a, None, spending_key, rng.random()).await;

            for claim_composer_fees in [false, true] {
                let mut global_state_lock = mock_genesis_global_state(
                    network,
                    2,
                    wallet_secret.clone(),
                    cli_args::Args::default(),
                )
                .await;
                let mut global_state = global_state_lock.lock_guard_mut().await;

                if claim_composer_fees {
                    global_state
                        .wallet_state
                        .add_expected_utxos(composer_expected_utxos_1a.clone())
                        .await;
                    global_state.set_new_tip(block_1a.clone()).await.unwrap();
                    global_state
                        .wallet_state
                        .add_expected_utxos(composer_expected_utxos_2a.clone())
                        .await;
                    global_state.set_new_tip(block_2a.clone()).await.unwrap();
                    global_state
                        .wallet_state
                        .add_expected_utxos(composer_expected_utxos_3a.clone())
                        .await;
                    global_state.set_new_tip(block_3a.clone()).await.unwrap();
                    global_state
                        .wallet_state
                        .add_expected_utxos(composer_expected_utxos_1a.clone())
                        .await;
                    global_state.set_new_tip(block_1a.clone()).await.unwrap();
                } else {
                    global_state.set_new_tip(block_1a.clone()).await.unwrap();
                    global_state.set_new_tip(block_2a.clone()).await.unwrap();
                    global_state.set_new_tip(block_3a.clone()).await.unwrap();
                    global_state.set_new_tip(block_1a.clone()).await.unwrap();
                }

                let expected_number_of_mutxos = if claim_composer_fees { 3 } else { 1 };

                assert_correct_global_state(
                    &global_state,
                    block_1a.clone(),
                    genesis_block.clone(),
                    1,
                    expected_number_of_mutxos,
                )
                .await;

                // Verify that we can also reorganize with last shared ancestor being
                // the genesis block.
                let (block_1b, _) =
                    make_mock_block(&genesis_block, None, spending_key, random()).await;
                global_state.set_new_tip(block_1b.clone()).await.unwrap();
                assert_correct_global_state(
                    &global_state,
                    block_1b.clone(),
                    genesis_block.clone(),
                    2,
                    1,
                )
                .await;

                // Add many blocks, verify state-validity after each.
                let mut previous_block = block_1b;
                for block_height in 2..60 {
                    let (next_block, composer_expected_utxos) =
                        make_mock_block(&previous_block, None, spending_key, rng.random()).await;
                    global_state
                        .wallet_state
                        .add_expected_utxos(composer_expected_utxos.clone())
                        .await;
                    global_state.set_new_tip(next_block.clone()).await.unwrap();
                    global_state
                        .wallet_state
                        .add_expected_utxos(composer_expected_utxos.clone())
                        .await;
                    global_state.set_new_tip(next_block.clone()).await.unwrap();
                    assert_correct_global_state(
                        &global_state,
                        next_block.clone(),
                        previous_block.clone(),
                        if block_height <= 3 { 2 } else { 1 },
                        2 * (block_height - 1) + 1,
                    )
                    .await;
                    previous_block = next_block;
                }
            }
        }

        #[traced_test]
        #[tokio::test]
        async fn setting_same_tip_twice_is_allowed() {
            let mut rng = rand::rng();
            let network = Network::Main;
            let wallet_secret = WalletSecret::devnet_wallet();
            let genesis_block = Block::genesis(network);
            let spend_key = wallet_secret.nth_generation_spending_key(0);

            let (block_1, composer_expected_utxos_1) =
                make_mock_block(&genesis_block, None, spend_key, rng.random()).await;

            for claim_cb in [false, true] {
                let expected_num_mutxos = if claim_cb { 3 } else { 1 };
                let mut global_state_lock = mock_genesis_global_state(
                    network,
                    2,
                    wallet_secret.clone(),
                    cli_args::Args::default(),
                )
                .await;
                let mut global_state = global_state_lock.lock_guard_mut().await;

                if claim_cb {
                    global_state
                        .wallet_state
                        .add_expected_utxos(composer_expected_utxos_1.clone())
                        .await;
                    global_state.set_new_tip(block_1.clone()).await.unwrap();
                    global_state
                        .wallet_state
                        .add_expected_utxos(composer_expected_utxos_1.clone())
                        .await;
                    global_state.set_new_tip(block_1.clone()).await.unwrap();
                } else {
                    global_state.set_new_tip(block_1.clone()).await.unwrap();
                    global_state.set_new_tip(block_1.clone()).await.unwrap();
                }

                assert_correct_global_state(
                    &global_state,
                    block_1.clone(),
                    genesis_block.clone(),
                    1,
                    expected_num_mutxos,
                )
                .await;
            }
        }
    }

    /// tests that pertain to restoring a wallet from seed-phrase
    /// and comparing onchain vs offchain notification methods.
    mod restore_wallet {
        use super::*;
        use crate::mine_loop::create_block_transaction_from;
        use crate::mine_loop::TxMergeOrigin;

        /// test scenario: onchain/symmetric.
        /// pass outcome: no funds loss
        ///
        /// test described in [change_exists()]
        #[traced_test]
        #[tokio::test]
        #[allow(clippy::needless_return)]
        async fn onchain_symmetric_change_exists() {
            change_exists(UtxoNotificationMedium::OnChain, KeyType::Symmetric).await
        }

        /// test scenario: onchain/generation.
        /// pass outcome: no funds loss
        ///
        /// test described in [change_exists()]
        #[traced_test]
        #[tokio::test]
        #[allow(clippy::needless_return)]
        async fn onchain_generation_change_exists() {
            change_exists(UtxoNotificationMedium::OnChain, KeyType::Generation).await
        }

        /// test scenario: offchain/symmetric.
        /// pass outcome: all funds lost!
        ///
        /// test described in [change_exists()]
        #[traced_test]
        #[tokio::test]
        #[allow(clippy::needless_return)]
        async fn offchain_symmetric_change_exists() {
            change_exists(UtxoNotificationMedium::OffChain, KeyType::Symmetric).await
        }

        /// test scenario: offchain/generation.
        /// pass outcome: all funds lost!
        ///
        /// test described in [change_exists()]
        #[traced_test]
        #[tokio::test]
        #[allow(clippy::needless_return)]
        async fn offchain_generation_change_exists() {
            change_exists(UtxoNotificationMedium::OffChain, KeyType::Generation).await
        }

        /// basic scenario:  alice receives 20 coins in the premine.  7 months
        /// after launch she sends 10 coins to bob, plus 1 coin fee.  alice should
        /// receive change of 9.  Sometime after this block is mined alice's
        /// hard drive crashes and she loses her wallet.  She still has her wallet
        /// seed and uses it to create a new wallet and scan blockchain to recover
        /// funds.  At the end alice checks her wallet balance, which should be
        /// 9.
        ///
        /// note: the pre-mine and 7-months aspects are unimportant.  This test
        /// would have same results if alice were a coinbase recipient instead.
        ///
        /// variations:
        ///   change_notify_medium: alice can choose OnChain or OffChain utxo notification.
        ///   change_key_type:    alice's change key can be Symmetric or Generation
        ///
        /// outcomes:
        ///   onchain/symmetric:    balance: 9.  no funds loss.
        ///   onchain/generation:   balance: 9.  no funds loss.
        ///   offchain/symmetric:   balance: 0. all funds lost!
        ///   offchain/generation:  balance: 0. all funds lost!
        ///
        /// this function expects the above possible outcomes.  ie, it passes when
        /// it encounters those outcomes.
        ///
        /// These outcomes highlight the danger of using off-chain notification.
        /// Even though alice stored her seed safely offline she still loses all her
        /// funds.
        ///
        /// It is important to recognize that alice's hard drive may crash (or
        /// device stolen, etc) at any moment after she sends the transaction.  If
        /// it happens 10 minutes after the transaction its unlikely she would have
        /// a wallet backup.  Or it could happen years after the transaction,
        /// demonstrating that alice's wallet needs to be backed up in perpetuity.
        ///
        /// From this, we conclude that the only way alice could really use offchain
        /// notification safely is if her wallet is stored on some kind of redundant
        /// storage media that is expected to exist in perpetuity.
        ///
        /// Since most people do not have home raid arrays and regular backup
        /// schedules it seems that offchain notifications are best suited for
        /// scenarios where the wallet is stored encrypted on some kind of cloud
        /// storage, whether centralized or decentralized.
        ///
        /// It may also be a business opportunity for hardware vendors to sell
        /// redundant-storage-in-a-box to users that want to use offchain
        /// notification but keep their wallets local.
        async fn change_exists(
            change_notification_medium: UtxoNotificationMedium,
            change_key_type: KeyType,
        ) {
            // setup initial conditions
            let network = Network::Main;
            let mut rng = StdRng::seed_from_u64(10001);

            let genesis_block = Block::genesis(network);
            let launch = genesis_block.kernel.header.timestamp;
            let seven_months_post_launch = launch + Timestamp::months(7);

            // amounts used in alice-to-bob transaction.
            let alice_to_bob_amount = NativeCurrencyAmount::coins(10);
            let alice_to_bob_fee = NativeCurrencyAmount::coins(1);

            // init global state for alice bob
            let mut alice_state_lock = mock_genesis_global_state(
                network,
                3,
                WalletSecret::devnet_wallet(),
                cli_args::Args::default(),
            )
            .await;
            let mut bob_state_lock = mock_genesis_global_state(
                network,
                3,
                WalletSecret::new_pseudorandom(rng.random()),
                cli_args::Args::default(),
            )
            .await;

            // in bob wallet: create receiving address for bob
            let bob_address = {
                bob_state_lock
                    .lock_guard_mut()
                    .await
                    .wallet_state
                    .next_unused_spending_key(KeyType::Generation)
                    .await
                    .unwrap()
                    .to_address()
                    .unwrap()
            };

            // in alice wallet: send pre-mined funds to bob
            let block_1 = {
                let vm_job_queue = alice_state_lock.vm_job_queue().clone();
                // let mut alice_state_mut = alice_state_lock.lock_guard_mut().await;

                // store and verify alice's initial balance from pre-mine.
                let alice_initial_balance = alice_state_lock
                    .lock_guard()
                    .await
                    .get_wallet_status_for_tip()
                    .await
                    .synced_unspent_available_amount(seven_months_post_launch);
                assert_eq!(alice_initial_balance, NativeCurrencyAmount::coins(20));

                // create change key for alice. change_key_type is a test param.
                let alice_change_key = alice_state_lock
                    .lock_guard_mut()
                    .await
                    .wallet_state
                    .next_unused_spending_key(change_key_type)
                    .await
                    .unwrap();

                // create an output for bob, worth 20.
                let outputs = vec![(bob_address, alice_to_bob_amount)];
                let tx_outputs = alice_state_lock.lock_guard().await.generate_tx_outputs(
                    outputs,
                    change_notification_medium,
                    UtxoNotificationMedium::OnChain,
                );

                // create tx.  utxo_notify_method is a test param.
                let (alice_to_bob_tx, _, maybe_change_utxo) = alice_state_lock
                    .lock_guard()
                    .await
                    .create_transaction_with_prover_capability(
                        tx_outputs.clone(),
                        alice_change_key,
                        change_notification_medium,
                        alice_to_bob_fee,
                        seven_months_post_launch,
                        TxProvingCapability::SingleProof,
                        &vm_job_queue,
                    )
                    .await
                    .unwrap();
                let Some(change_utxo) = maybe_change_utxo else {
                    panic!("A change Tx-output was expected");
                };

                // Inform alice wallet of any expected incoming utxos.
                // note: no-op when all utxo notifications are sent on-chain.
                let expected_utxo = alice_state_lock
                    .lock_guard()
                    .await
                    .wallet_state
                    .extract_expected_utxos(
                        tx_outputs.concat_with(vec![change_utxo]),
                        UtxoNotifier::Myself,
                    );
                alice_state_lock
                    .lock_guard_mut()
                    .await
                    .wallet_state
                    .add_expected_utxos(expected_utxo)
                    .await;

                // the block gets mined.
                // Alice's wallet does not register the composer reward because
                // it is not expecting it.
                let (block_1_tx, _) = create_block_transaction_from(
                    &genesis_block,
                    &alice_state_lock,
                    seven_months_post_launch,
                    (TritonVmJobPriority::Normal, None).into(),
                    TxMergeOrigin::ExplicitList(vec![alice_to_bob_tx]),
                )
                .await
                .unwrap();
                let block_1 = Block::compose(
                    &genesis_block,
                    block_1_tx,
                    seven_months_post_launch,
                    None,
                    &TritonVmJobQueue::dummy(),
                    TritonVmJobPriority::default().into(),
                )
                .await
                .unwrap();

                // alice's node learns of the new block.
                alice_state_lock
                    .lock_guard_mut()
                    .await
                    .set_new_tip(block_1.clone())
                    .await
                    .unwrap();

                // alice should have 2 monitored utxos.
                assert_eq!(
                    2,
                    alice_state_lock
                    .lock_guard()
                    .await
                        .wallet_state
                        .wallet_db
                        .monitored_utxos()
                        .len().await, "Alice must have 2 UTXOs after block 1: change from transaction, and the spent premine UTXO"
                );

                // Now alice should have a balance of 9.
                // balance = premine - spent = premine - sent - fee = 20 - 10 - 1
                let alice_calculated_balance = alice_initial_balance
                    .checked_sub(&alice_to_bob_amount)
                    .unwrap()
                    .checked_sub(&alice_to_bob_fee)
                    .unwrap();
                assert_eq!(alice_calculated_balance, NativeCurrencyAmount::coins(9));

                assert_eq!(
                    alice_calculated_balance,
                    alice_state_lock
                        .lock_guard()
                        .await
                        .get_wallet_status_for_tip()
                        .await
                        .synced_unspent_available_amount(seven_months_post_launch)
                );

                block_1
            };

            // in bob's wallet
            {
                let mut bob_state_mut = bob_state_lock.lock_guard_mut().await;

                // bob's node adds block1 to the chain.
                bob_state_mut.set_new_tip(block_1.clone()).await.unwrap();

                // Now Bob should have a balance of 10, from Alice
                assert_eq!(
                    alice_to_bob_amount, // 10
                    bob_state_mut
                        .get_wallet_status_for_tip()
                        .await
                        .synced_unspent_available_amount(seven_months_post_launch)
                );
            }

            // some time in the future.  minutes, months, or years...

            // oh no!  alice's hard-drive crashes and she loses her wallet.
            drop(alice_state_lock);

            // Fortunately alice still has her seed that she can restore from.
            {
                // devnet_wallet() stands in for alice's seed.
                let mut alice_restored_state_lock = mock_genesis_global_state(
                    network,
                    3,
                    WalletSecret::devnet_wallet(),
                    cli_args::Args::default(),
                )
                .await;

                let mut alice_state_mut = alice_restored_state_lock.lock_guard_mut().await;

                // ensure alice's wallet knows about the first key of each type.
                // for a real restore, we should generate perhaps 1000 of each.
                let _ = alice_state_mut
                    .wallet_state
                    .next_unused_spending_key(KeyType::Generation)
                    .await;
                let _ = alice_state_mut
                    .wallet_state
                    .next_unused_spending_key(KeyType::Symmetric)
                    .await;

                // check alice's initial balance after genesis.
                let alice_initial_balance = alice_state_mut
                    .get_wallet_status_for_tip()
                    .await
                    .synced_unspent_available_amount(seven_months_post_launch);

                // lucky alice's wallet begins with 20 balance from premine.
                assert_eq!(alice_initial_balance, NativeCurrencyAmount::coins(20));

                // now alice must replay old blocks.  (there's only one so far)
                alice_state_mut.set_new_tip(block_1).await.unwrap();

                // Now alice should have a balance of 9.
                // 20 from premine - 11
                let alice_calculated_balance = alice_initial_balance
                    .checked_sub(&alice_to_bob_amount)
                    .unwrap()
                    .checked_sub(&alice_to_bob_fee)
                    .unwrap();

                assert_eq!(alice_calculated_balance, NativeCurrencyAmount::coins(9));

                // For onchain change-notification the balance will be 9.
                // For offchain change-notification, it will be 0.  Funds are lost!!!
                let alice_expected_balance_by_method = match change_notification_medium {
                    UtxoNotificationMedium::OnChain => NativeCurrencyAmount::coins(9),
                    UtxoNotificationMedium::OffChain => NativeCurrencyAmount::coins(0),
                };

                // verify that our on/offchain prediction is correct.
                assert_eq!(
                    alice_expected_balance_by_method,
                    alice_state_mut
                        .get_wallet_status_for_tip()
                        .await
                        .synced_unspent_available_amount(seven_months_post_launch)
                );
            }
        }
    }
}
