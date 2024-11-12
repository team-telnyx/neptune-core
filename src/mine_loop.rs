use std::time::Duration;

use anyhow::bail;
use anyhow::Result;
use futures::channel::oneshot;
use num_traits::CheckedSub;
use rand::rngs::StdRng;
use rand::thread_rng;
use rand::Rng;
use rand::SeedableRng;
use tokio::select;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::*;
use transaction_output::TxOutput;
use twenty_first::math::digest::Digest;

use crate::job_queue::triton_vm::TritonVmJobPriority;
use crate::models::blockchain::block::block_height::BlockHeight;
use crate::models::blockchain::block::difficulty_control::difficulty_control;
use crate::models::blockchain::block::*;
use crate::models::blockchain::transaction::*;
use crate::models::channel::*;
use crate::models::proof_abstractions::timestamp::Timestamp;
use crate::models::shared::SIZE_20MB_IN_BYTES;
use crate::models::state::transaction_details::TransactionDetails;
use crate::models::state::tx_proving_capability::TxProvingCapability;
use crate::models::state::wallet::expected_utxo::ExpectedUtxo;
use crate::models::state::wallet::expected_utxo::UtxoNotifier;
use crate::models::state::GlobalState;
use crate::models::state::GlobalStateLock;
use crate::prelude::twenty_first;

async fn compose_block(
    latest_block: Block,
    global_state_lock: GlobalStateLock,
    sender: oneshot::Sender<(Block, Vec<ExpectedUtxo>)>,
) -> Result<()> {
    let now = Timestamp::now();
    let guesser_fee_fraction = global_state_lock.cli().guesser_fraction;

    let (transaction, composer_utxos) =
        create_block_transaction(&latest_block, &global_state_lock, now, guesser_fee_fraction)
            .await?;

    let triton_vm_job_queue = global_state_lock.vm_job_queue();
    let proposal = Block::compose(
        &latest_block,
        transaction,
        now,
        Digest::default(),
        None,
        triton_vm_job_queue,
        (
            TritonVmJobPriority::High,
            global_state_lock.cli().max_log2_padded_height_for_proofs,
        )
            .into(),
    )
    .await;

    let proposal = match proposal {
        Ok(template) => template,
        Err(_) => bail!("Miner failed to generate block template"),
    };

    // Please clap.
    match sender.send((proposal, composer_utxos)) {
        Ok(_) => Ok(()),
        Err(_) => bail!("Composer task failed to send to miner master"),
    }
}

/// Attempt to mine a valid block for the network
#[allow(clippy::too_many_arguments)]
async fn guess_nonce(
    block: Block,
    previous_block: Block,
    sender: oneshot::Sender<NewBlockFound>,
    composer_utxos: Vec<ExpectedUtxo>,
    unrestricted_mining: bool,
    target_block_interval: Option<Timestamp>,
) {
    // We wrap mining loop with spawn_blocking() because it is a
    // very lengthy and CPU intensive task, which should execute
    // on its own thread.
    //
    // Instead of spawn_blocking(), we could start a native OS
    // thread which avoids using one from tokio's threadpool
    // but that doesn't seem a concern for neptune-core.
    // Also we would need to use a oneshot channel to avoid
    // blocking while joining the thread.
    // see: https://ryhl.io/blog/async-what-is-blocking/
    //
    // note: there is no async code inside the mining loop.
    tokio::task::spawn_blocking(move || {
        guess_worker(
            block,
            previous_block,
            sender,
            composer_utxos,
            unrestricted_mining,
            target_block_interval,
        )
    })
    .await
    .unwrap()
}

fn guess_worker(
    mut block: Block,
    previous_block: Block,
    sender: oneshot::Sender<NewBlockFound>,
    composer_utxos: Vec<ExpectedUtxo>,
    unrestricted_mining: bool,
    target_block_interval: Option<Timestamp>,
) {
    // This must match the rules in `[Block::has_proof_of_work]`.
    let prev_difficulty = previous_block.header().difficulty;
    let threshold = prev_difficulty.target();
    info!(
        "Mining on block with {} outputs and difficulty {}. Attempting to find block with height {} with digest less than target: {}",
        block.body().transaction_kernel.outputs.len(),
        previous_block.header().difficulty,
        block.header().height,
        threshold
    );

    // The RNG used to sample nonces must be thread-safe, which `thread_rng()` is not.
    // Solution: use `thread_rng()` to generate a seed, and generate a thread-safe RNG
    // seeded with that seed. The `thread_rng()` object is dropped immediately.
    let mut rng: StdRng = SeedableRng::from_seed(thread_rng().gen());

    // Mining loop
    let mut nonce_preimage = Digest::default();
    while !guess_nonce_iteration(
        &mut block,
        &previous_block,
        &sender,
        DifficultyInfo {
            target_block_interval,
            threshold,
        },
        &mut nonce_preimage,
        unrestricted_mining,
        &mut rng,
    ) {}
    // If the sender is cancelled, the parent to this thread most
    // likely received a new block, and this thread hasn't been stopped
    // yet by the operating system, although the call to abort this
    // thread *has* been made.
    if sender.is_canceled() {
        info!(
            "Abandoning mining of current block with height {}",
            block.kernel.header.height
        );
        return;
    }

    let nonce = block.kernel.header.nonce;
    info!("Found valid block with nonce: ({nonce}).");

    let timestamp = block.kernel.header.timestamp;
    let timestamp_standard = timestamp.standard_format();
    let hash = block.hash();
    let hex = hash.to_hex();
    let height = block.kernel.header.height;
    let num_inputs = block.body().transaction_kernel.inputs.len();
    let num_outputs = block.body().transaction_kernel.outputs.len();
    info!(
        r#"Newly mined block details:
              Height: {height}
              Time  : {timestamp_standard} ({timestamp})
        Digest (Hex): {hex}
        Digest (Raw): {hash}
Difficulty threshold: {threshold}
          Difficulty: {prev_difficulty}
          #inputs   : {num_inputs}
          #outputs  : {num_outputs}
"#
    );

    let guesser_fee_utxo_infos = block.guesser_fee_expected_utxos(nonce_preimage);

    assert!(
        !guesser_fee_utxo_infos.is_empty(),
        "All mined blocks have guesser fees"
    );

    let new_block_found = NewBlockFound {
        block: Box::new(block),
        composer_utxos,
        guesser_fee_utxo_infos,
    };

    sender
        .send(new_block_found)
        .unwrap_or_else(|_| warn!("Receiver in mining loop closed prematurely"))
}

pub(crate) struct DifficultyInfo {
    pub(crate) target_block_interval: Option<Timestamp>,
    pub(crate) threshold: Digest,
}

/// Run a single iteration of the mining loop.
///
/// Returns true if a) a valid block is found; or b) the task is terminated.
#[inline]
fn guess_nonce_iteration(
    block: &mut Block,
    previous_block: &Block,
    sender: &oneshot::Sender<NewBlockFound>,
    difficulty_info: DifficultyInfo,
    nonce_preimage: &mut Digest,
    unrestricted_mining: bool,
    rng: &mut StdRng,
) -> bool {
    if sender.is_canceled() {
        info!(
            "Abandoning mining of current block with height {}",
            block.kernel.header.height
        );
        return true;
    }

    // Modify the nonce in the block header. In order to collect the guesser
    // fee, this nonce must be the post-image of a known pre-image under Tip5.
    *nonce_preimage = rng.gen();
    block.set_header_nonce(nonce_preimage.hash());

    // See issue #149 and test block_timestamp_represents_time_block_found()
    // this ensures header timestamp represents the moment block is found.
    // this is simplest impl.  Efficiencies can perhaps be gained by only
    // performing every N iterations, or other strategies.
    let now = Timestamp::now();
    let new_difficulty = difficulty_control(
        now,
        previous_block.header().timestamp,
        previous_block.header().difficulty,
        difficulty_info.target_block_interval,
        previous_block.header().height,
    );
    block.set_header_timestamp_and_difficulty(now, new_difficulty);

    let success = block.hash() <= difficulty_info.threshold;

    if !unrestricted_mining {
        std::thread::sleep(Duration::from_millis(100));
    }

    success
}

pub(crate) async fn make_coinbase_transaction(
    global_state_lock: &GlobalStateLock,
    guesser_block_subsidy_fraction: f64,
    timestamp: Timestamp,
) -> Result<(Transaction, Vec<ExpectedUtxo>)> {
    // note: it is Ok to always use the same key here because:
    //  1. if we find a block, the utxo will go to our wallet
    //     and notification occurs offchain, so there is no privacy issue.
    //  2. if we were to derive a new addr for each block then we would
    //     have large gaps since an address only receives funds when
    //     we actually win the mining lottery.
    //  3. also this way we do not have to modify global/wallet state.

    let coinbase_recipient_spending_key = global_state_lock
        .lock_guard()
        .await
        .wallet_state
        .wallet_secret
        .nth_generation_spending_key(0);
    let receiving_address = coinbase_recipient_spending_key.to_address();
    let latest_block = global_state_lock
        .lock_guard()
        .await
        .chain
        .light_state()
        .clone();
    let mutator_set_accumulator = latest_block.mutator_set_accumulator().clone();
    let next_block_height: BlockHeight = latest_block.header().height.next();

    #[allow(clippy::manual_range_contains)]
    if guesser_block_subsidy_fraction > 1.0 || guesser_block_subsidy_fraction < 0f64 {
        bail!("Guesser fee fraction must be in [0, 1] interval. Got: {guesser_block_subsidy_fraction}");
    }

    let coinbase_amount = Block::block_subsidy(next_block_height);
    let Some(guesser_fee) = coinbase_amount.lossy_f64_fraction_mul(guesser_block_subsidy_fraction)
    else {
        bail!("Guesser fee times block subsidy must be valid amount");
    };

    info!("Setting guesser_fee to {guesser_fee}.");

    // There is no reason to put coinbase UTXO notifications on chain, because:
    // Both sender randomness and receiver preimage are derived
    // deterministically from the wallet's seed.
    let Some(amount_to_prover) = coinbase_amount.checked_sub(&guesser_fee) else {
        bail!(
            "Guesser fee may not exceed coinbase amount. coinbase_amount: {}; guesser_fee: {}.",
            coinbase_amount.to_nau(),
            guesser_fee.to_nau()
        );
    };

    info!(
        "Setting coinbase amount to {coinbase_amount}; and amount to prover to {amount_to_prover}"
    );
    let sender_randomness: Digest = global_state_lock
        .lock_guard()
        .await
        .wallet_state
        .wallet_secret
        .generate_sender_randomness(next_block_height, receiving_address.privacy_digest);

    // TODO: Produce two outputs here, one timelocked and one not.
    let coinbase_output = TxOutput::offchain_native_currency(
        amount_to_prover,
        sender_randomness,
        receiving_address.into(),
    );

    let transaction_details = TransactionDetails::new_with_coinbase(
        vec![],
        vec![coinbase_output.clone()].into(),
        coinbase_amount,
        guesser_fee,
        timestamp,
        mutator_set_accumulator,
    )
    .expect(
        "all inputs' ms membership proofs must be valid because inputs are empty;\
 and tx must be balanced because the one output receives exactly the coinbase amount",
    );

    // 2. Create the transaction
    // A coinbase transaction implies mining. So you *must*
    // be able to create a SingleProof.

    // It's important to not hold any locks (not even read-locks), as
    // that prevents peers from connecting to this node.
    info!("Start: generate single proof for coinbase transaction");
    let vm_job_queue = global_state_lock.vm_job_queue();
    let transaction = GlobalState::create_raw_transaction(
        transaction_details,
        TxProvingCapability::SingleProof,
        vm_job_queue,
        (
            TritonVmJobPriority::High,
            global_state_lock.cli().max_log2_padded_height_for_proofs,
        )
            .into(),
    )
    .await?;
    info!("Done: generating single proof for coinbase transaction");

    let composer_utxo_not_timelocked = ExpectedUtxo::new(
        coinbase_output.utxo(),
        coinbase_output.sender_randomness(),
        coinbase_recipient_spending_key.privacy_preimage,
        UtxoNotifier::OwnMinerComposeBlock,
    );

    Ok((transaction, vec![composer_utxo_not_timelocked]))
}

/// Create the transaction that goes into the block template. The transaction is
/// built from the mempool and from the coinbase transaction. Also returns the
/// "sender randomness" used in the coinbase transaction.
pub(crate) async fn create_block_transaction(
    predecessor_block: &Block,
    global_state_lock: &GlobalStateLock,
    timestamp: Timestamp,
    guesser_fee_fraction: f64,
) -> Result<(Transaction, Vec<ExpectedUtxo>)> {
    let block_capacity_for_transactions = SIZE_20MB_IN_BYTES;

    let (coinbase_transaction, composer_utxos) =
        make_coinbase_transaction(global_state_lock, guesser_fee_fraction, timestamp).await?;

    debug!(
        "Creating block transaction with mutator set hash: {}",
        predecessor_block.mutator_set_accumulator().hash()
    );

    let mut rng: StdRng =
        SeedableRng::from_seed(global_state_lock.lock_guard().await.shuffle_seed());

    // Get most valuable transactions from mempool.
    // TODO: Change this const to be defined through CLI arguments.
    const MAX_NUM_TXS_TO_MERGE: usize = 7;
    let only_merge_single_proofs = true;
    let transactions_to_include = global_state_lock
        .lock_guard()
        .await
        .mempool
        .get_transactions_for_block(
            block_capacity_for_transactions,
            Some(MAX_NUM_TXS_TO_MERGE),
            only_merge_single_proofs,
        );

    // Merge incoming transactions with the coinbase transaction
    let num_transactions_to_include = transactions_to_include.len();
    let mut block_transaction = coinbase_transaction;
    let vm_job_queue = global_state_lock.vm_job_queue();
    for (i, transaction_to_include) in transactions_to_include.into_iter().enumerate() {
        info!(
            "Merging transaction {} / {}",
            i + 1,
            num_transactions_to_include
        );
        block_transaction = Transaction::merge_with(
            block_transaction,
            transaction_to_include,
            rng.gen(),
            vm_job_queue,
            (
                TritonVmJobPriority::High,
                global_state_lock.cli().max_log2_padded_height_for_proofs,
            )
                .into(),
        )
        .await
        .expect("Must be able to merge transactions in mining context");
    }

    Ok((block_transaction, composer_utxos))
}

///
///
/// Locking:
///   * acquires `global_state_lock` for write
pub(crate) async fn mine(
    mut from_main: mpsc::Receiver<MainToMiner>,
    to_main: mpsc::Sender<MinerToMain>,
    mut latest_block: Block,
    mut global_state_lock: GlobalStateLock,
) -> Result<()> {
    // Wait before starting mining task to ensure that peers have sent us information about
    // their latest blocks. This should prevent the client from finding blocks that will later
    // be orphaned.
    const INITIAL_MINING_SLEEP_IN_SECONDS: u64 = 60;
    tokio::time::sleep(Duration::from_secs(INITIAL_MINING_SLEEP_IN_SECONDS)).await;

    let mut pause_mine = false;
    let mut wait_for_confirmation = false;
    loop {
        let (guesser_tx, guesser_rx) = oneshot::channel::<NewBlockFound>();
        let (composer_tx, composer_rx) = oneshot::channel::<(Block, Vec<ExpectedUtxo>)>();
        global_state_lock.set_mining_status_to_inactive().await;

        let is_syncing = global_state_lock.lock(|s| s.net.syncing).await;

        let maybe_proposal = global_state_lock.lock_guard().await.block_proposal.clone();
        let guess = global_state_lock.cli().guess;
        let guesser_task: Option<JoinHandle<()>> = if !wait_for_confirmation
            && guess
            && maybe_proposal.is_some()
            && !is_syncing
            && !pause_mine
        {
            let composer_utxos = maybe_proposal.composer_utxos();
            global_state_lock.set_mining_status_to_guesing().await;
            maybe_proposal.map(|proposal| {
                let guesser_task = guess_nonce(
                    proposal.to_owned(),
                    latest_block.clone(),
                    guesser_tx,
                    composer_utxos,
                    global_state_lock.cli().sleepy_guessing,
                    None, // using default TARGET_BLOCK_INTERVAL
                );

                tokio::task::Builder::new()
                    .name("guesser")
                    .spawn(guesser_task)
                    .expect("Failed to spawn guesser task")
            })
        } else {
            None
        };

        let compose = global_state_lock.cli().compose;
        let composer_task = if !wait_for_confirmation
            && compose
            && guesser_task.is_none()
            && !is_syncing
            && !pause_mine
        {
            global_state_lock.set_mining_status_to_composing().await;
            let compose_task =
                compose_block(latest_block.clone(), global_state_lock.clone(), composer_tx);
            let task = tokio::task::Builder::new()
                .name("composer")
                .spawn(compose_task)
                .expect("Failed to spawn composer task.");

            Some(task)
        } else {
            None
        };

        // Await a message from either the worker task or from the main loop
        select! {
            Some(main_message) = from_main.recv() => {
                debug!("Miner received message type: {}", main_message.get_type());

                match main_message {
                    MainToMiner::Shutdown => {
                        debug!("Miner shutting down.");

                        if let Some(gt) = guesser_task {
                            gt.abort();
                            debug!("Abort-signal sent to guesser worker.");
                        }
                        if let Some(ct) = composer_task {
                            ct.abort();
                            debug!("Abort-signal sent to composer worker.");
                        }

                        break;
                    }
                    MainToMiner::NewBlock(block) => {
                        if let Some(gt) = guesser_task {
                            gt.abort();
                            debug!("Abort-signal sent to guesser worker.");
                        }
                        if let Some(ct) = composer_task {
                            ct.abort();
                            debug!("Abort-signal sent to composer worker.");
                        }

                        latest_block = *block;
                        info!("Miner task received {} block height {}", global_state_lock.lock(|s| s.cli().network).await, latest_block.kernel.header.height);
                    }
                    MainToMiner::NewBlockProposal => {
                        if let Some(gt) = guesser_task {
                            gt.abort();
                            debug!("Abort-signal sent to guesser worker.");
                        }
                        if let Some(ct) = composer_task {
                            ct.abort();
                            debug!("Abort-signal sent to composer worker.");
                        }

                        info!("Miner received message about new block proposal for guessing.");
                    }
                    MainToMiner::WaitForContinue => {
                        if let Some(gt) = guesser_task {
                            gt.abort();
                            debug!("Abort-signal sent to guesser worker.");
                        }
                        if let Some(ct) = composer_task {
                            ct.abort();
                            debug!("Abort-signal sent to composer worker.");
                        }

                        wait_for_confirmation = true;
                    }
                    MainToMiner::Continue => {
                        wait_for_confirmation = false;
                    }
                    MainToMiner::StopMining => {
                        pause_mine = true;

                        if let Some(gt) = guesser_task {
                            gt.abort();
                            debug!("Abort-signal sent to guesser worker.");
                        }
                        if let Some(ct) = composer_task {
                            ct.abort();
                            debug!("Abort-signal sent to composer worker.");
                        }
                    }
                    MainToMiner::StartMining => {
                        pause_mine = false;
                    }
                    MainToMiner::StopSyncing => {
                        // no need to do anything here.  Mining will
                        // resume or not at top of loop depending on
                        // pause_mine and syncing variables.
                    }
                    MainToMiner::StartSyncing => {
                        // when syncing begins, we must halt the mining
                        // task.  But we don't change the pause_mine
                        // variable, because it reflects the logical on/off
                        // of mining, which syncing can temporarily override
                        // but not alter the setting.
                        if let Some(gt) = guesser_task {
                            gt.abort();
                            debug!("Abort-signal sent to guesser worker.");
                        }
                        if let Some(ct) = composer_task {
                            ct.abort();
                            debug!("Abort-signal sent to composer worker.");
                        }
                    }
                }
            }
            new_composition = composer_rx => {
                let (new_block_proposal, composer_utxos) = match new_composition {
                    Ok(k) => k,
                    Err(e) => {warn!("composing task was cancelled prematurely. Got: {}", e);
                    continue;}
                };
                to_main.send(MinerToMain::BlockProposal(Box::new((new_block_proposal, composer_utxos)))).await?;

                wait_for_confirmation = true;
            }
            new_block = guesser_rx => {
                let new_block_found = match new_block {
                    Ok(res) => res,
                    Err(err) => {
                        warn!("Mining task was cancelled prematurely. Got: {}", err);
                        continue;
                    }
                };

                debug!("Worker task reports new block of height {}", new_block_found.block.kernel.header.height);

                // Sanity check, remove for more efficient mining.
                // The below PoW check could fail due to race conditions. So we don't panic,
                // we only ignore what the worker task sent us.
                if !new_block_found.block.has_proof_of_work(&latest_block) {
                    error!("Own mined block did not have valid PoW Discarding.");
                    continue;
                }

                if !new_block_found.block.is_valid(&latest_block, Timestamp::now()) {
                    // Block could be invalid if for instance the proof and proof-of-work
                    // took less time than the minimum block time.
                    error!("Found block with valid proof-of-work but block is invalid.");
                    continue;
                }

                info!("Found new {} block with block height {}. Hash: {}", global_state_lock.cli().network, new_block_found.block.kernel.header.height, new_block_found.block.hash());

                latest_block = *new_block_found.block.to_owned();
                to_main.send(MinerToMain::NewBlockFound(new_block_found)).await?;

                wait_for_confirmation = true;
            }
        }
    }
    debug!("Miner shut down gracefully.");
    Ok(())
}

#[cfg(test)]
pub(crate) mod mine_loop_tests {
    use std::hint::black_box;

    use block_appendix::BlockAppendix;
    use block_body::BlockBody;
    use block_header::block_header_tests::random_block_header;
    use difficulty_control::Difficulty;
    use mutator_set_update::MutatorSetUpdate;
    use num_bigint::BigUint;
    use num_traits::Pow;
    use num_traits::Zero;
    use tracing_test::traced_test;
    use transaction_output::TxOutput;
    use transaction_output::UtxoNotificationMedium;

    use super::*;
    use crate::config_models::cli_args;
    use crate::config_models::network::Network;
    use crate::job_queue::triton_vm::TritonVmJobQueue;
    use crate::models::blockchain::type_scripts::neptune_coins::NeptuneCoins;
    use crate::models::proof_abstractions::timestamp::Timestamp;
    use crate::models::state::mempool::TransactionOrigin;
    use crate::tests::shared::dummy_expected_utxo;
    use crate::tests::shared::make_mock_transaction_with_mutator_set_hash;
    use crate::tests::shared::mock_genesis_global_state;
    use crate::tests::shared::random_transaction_kernel;
    use crate::util_types::test_shared::mutator_set::random_mmra;
    use crate::util_types::test_shared::mutator_set::random_mutator_set_accumulator;
    use crate::WalletSecret;

    /// Similar to [mine_iteration] function but intended for tests.
    ///
    /// Does *not* update the timestamp of the block and therefore also does not
    /// update the difficulty field, as this applies to the next block and only
    /// changes as a result of the timestamp of this block.
    pub(crate) fn mine_iteration_for_tests(
        block: &mut Block,
        threshold: Digest,
        rng: &mut StdRng,
    ) -> bool {
        block.set_header_nonce(rng.gen());
        block.hash() <= threshold
    }

    /// Estimates the hash rate in number of hashes per milliseconds
    async fn estimate_own_hash_rate(
        target_block_interval: Option<Timestamp>,
        unrestricted_mining: bool,
    ) -> f64 {
        let mut rng: StdRng = SeedableRng::from_rng(thread_rng()).unwrap();
        let network = Network::RegTest;
        let global_state_lock = mock_genesis_global_state(
            network,
            2,
            WalletSecret::devnet_wallet(),
            cli_args::Args::default(),
        )
        .await;

        let previous_block = global_state_lock
            .lock_guard()
            .await
            .chain
            .light_state()
            .clone();

        let (transaction, _coinbase_utxo_info) = {
            (
                make_mock_transaction_with_mutator_set_hash(
                    vec![],
                    vec![],
                    previous_block.mutator_set_accumulator().hash(),
                ),
                dummy_expected_utxo(),
            )
        };
        let start_time = Timestamp::now();
        let mut block = Block::block_template_invalid_proof(
            &previous_block,
            transaction,
            start_time,
            Digest::default(),
            target_block_interval,
        );
        let threshold = previous_block.header().difficulty.target();

        let (worker_task_tx, _worker_task_rx) = oneshot::channel::<NewBlockFound>();

        let num_iterations = 10000;
        let mut nonce_preimage = Digest::default();
        let tick = std::time::SystemTime::now();
        for _ in 0..num_iterations {
            guess_nonce_iteration(
                &mut block,
                &previous_block,
                &worker_task_tx,
                DifficultyInfo {
                    target_block_interval,
                    threshold,
                },
                &mut nonce_preimage,
                unrestricted_mining,
                &mut rng,
            );
        }
        let time_spent_mining = tick.elapsed().unwrap();

        (num_iterations as f64) / (time_spent_mining.as_millis() as f64)
    }

    /// Estimate the time it takes to prepare a block so we can start guessing
    /// nonces.
    async fn estimate_block_preparation_time_invalid_proof() -> f64 {
        let network = Network::Main;
        let genesis_block = Block::genesis_block(network);

        let global_state_lock = mock_genesis_global_state(
            network,
            2,
            WalletSecret::devnet_wallet(),
            cli_args::Args::default(),
        )
        .await;
        let tick = std::time::SystemTime::now();
        let (transaction, _coinbase_utxo_info) =
            make_coinbase_transaction(&global_state_lock, 0f64, network.launch_date())
                .await
                .unwrap();

        let in_seven_months = network.launch_date() + Timestamp::months(7);
        let block = Block::block_template_invalid_proof(
            &genesis_block,
            transaction,
            in_seven_months,
            Digest::default(),
            None,
        );
        let tock = tick.elapsed().unwrap().as_millis() as f64;
        black_box(block);
        tock
    }

    #[traced_test]
    #[tokio::test]
    async fn block_template_is_valid() {
        // Verify that a block template made with transaction from the mempool is a valid block
        let network = Network::Main;
        let mut alice = mock_genesis_global_state(
            network,
            2,
            WalletSecret::devnet_wallet(),
            cli_args::Args::default(),
        )
        .await;
        let genesis_block = Block::genesis_block(network);
        let now = genesis_block.kernel.header.timestamp + Timestamp::months(7);
        assert!(
            !alice
                .lock_guard()
                .await
                .get_wallet_status_for_tip()
                .await
                .synced_unspent_available_amount(now)
                .is_zero(),
            "Assumed to be premine-recipient"
        );

        let mut rng = StdRng::seed_from_u64(u64::from_str_radix("2350404", 6).unwrap());

        let alice_key = alice
            .lock_guard()
            .await
            .wallet_state
            .wallet_secret
            .nth_generation_spending_key_for_tests(0);
        let output_to_alice = TxOutput::offchain_native_currency(
            NeptuneCoins::new(4),
            rng.gen(),
            alice_key.to_address().into(),
        );
        let (tx_from_alice, _maybe_change_output) = alice
            .lock_guard()
            .await
            .create_transaction_with_prover_capability(
                vec![output_to_alice].into(),
                alice_key.into(),
                UtxoNotificationMedium::OffChain,
                NeptuneCoins::new(1),
                now,
                TxProvingCapability::SingleProof,
                &TritonVmJobQueue::dummy(),
            )
            .await
            .unwrap();

        for guesser_fee_fraction in [0f64, 0.5, 1.0] {
            // Verify constructed coinbase transaction and block template when mempool is empty
            assert!(
                alice.lock_guard().await.mempool.is_empty(),
                "Mempool must be empty at start of loop"
            );
            let (transaction_empty_mempool, _coinbase_utxo_info) = {
                make_coinbase_transaction(&alice, guesser_fee_fraction, now)
                    .await
                    .unwrap()
            };

            assert_eq!(
                1,
                transaction_empty_mempool.kernel.outputs.len(),
                "Coinbase transaction with empty mempool must have exactly one output"
            );
            assert!(
                transaction_empty_mempool.kernel.inputs.is_empty(),
                "Coinbase transaction with empty mempool must have zero inputs"
            );
            let block_1_empty_mempool = Block::compose(
                &genesis_block,
                transaction_empty_mempool,
                now,
                Digest::default(),
                None,
                &TritonVmJobQueue::dummy(),
                TritonVmJobPriority::High.into(),
            )
            .await
            .unwrap();
            assert!(
                block_1_empty_mempool.is_valid(&genesis_block, now),
                "Block template created by miner with empty mempool must be valid"
            );

            {
                let mut alice_gsm = alice.lock_guard_mut().await;
                alice_gsm
                    .mempool_insert(tx_from_alice.clone(), TransactionOrigin::Own)
                    .await;
                assert_eq!(1, alice_gsm.mempool.len());
            }

            // Build transaction for block
            let (transaction_non_empty_mempool, _new_coinbase_sender_randomness) = {
                create_block_transaction(&genesis_block, &alice, now, guesser_fee_fraction)
                    .await
                    .unwrap()
            };
            assert_eq!(
            3,
            transaction_non_empty_mempool.kernel.outputs.len(),
            "Transaction for block with non-empty mempool must contain coinbase output, send output, and change output"
        );
            assert_eq!(1, transaction_non_empty_mempool.kernel.inputs.len(), "Transaction for block with non-empty mempool must contain one input: the genesis UTXO being spent");

            // Build and verify block template
            let block_1_nonempty_mempool = Block::compose(
                &genesis_block,
                transaction_non_empty_mempool,
                now,
                Digest::default(),
                None,
                &TritonVmJobQueue::dummy(),
                TritonVmJobPriority::default().into(),
            )
            .await
            .unwrap();
            assert!(
                block_1_nonempty_mempool.is_valid(&genesis_block, now + Timestamp::seconds(2)),
                "Block template created by miner with non-empty mempool must be valid"
            );

            alice.lock_guard_mut().await.mempool_clear().await;
        }
    }

    /// This test mines a single block at height 1 on the main network
    /// and then validates it with `Block::is_valid()` and
    /// `Block::has_proof_of_work()`.
    ///
    /// This is a regression test for issue #131.
    /// https://github.com/Neptune-Crypto/neptune-core/issues/131
    ///
    /// The cause of the failure was that `mine_block_worker()` was comparing
    /// hash(block_header) against difficulty threshold while
    /// `Block::has_proof_of_work` uses hash(block) instead.
    ///
    /// The fix was to modify `mine_block_worker()` so that it also
    /// uses hash(block) and subsequently the test passes (unmodified).
    ///
    /// This test is present and fails in commit
    /// b093631fd0d479e6c2cc252b08f18d920a1ec2e5 which is prior to the fix.
    #[traced_test]
    #[tokio::test]
    async fn mined_block_has_proof_of_work() {
        let network = Network::Main;
        let global_state_lock = mock_genesis_global_state(
            network,
            2,
            WalletSecret::devnet_wallet(),
            cli_args::Args::default(),
        )
        .await;
        let tip_block_orig = Block::genesis_block(network);
        let launch_date = tip_block_orig.header().timestamp;
        let (worker_task_tx, worker_task_rx) = oneshot::channel::<NewBlockFound>();

        let (transaction, coinbase_utxo_info) =
            make_coinbase_transaction(&global_state_lock, 0f64, launch_date)
                .await
                .unwrap();

        let block = Block::block_template_invalid_proof(
            &tip_block_orig,
            transaction,
            launch_date,
            Digest::default(),
            None,
        );

        let unrestricted_mining = true;

        guess_worker(
            block,
            tip_block_orig.clone(),
            worker_task_tx,
            coinbase_utxo_info,
            unrestricted_mining,
            None,
        );

        let mined_block_info = worker_task_rx.await.unwrap();

        assert!(mined_block_info.block.has_proof_of_work(&tip_block_orig));
    }

    /// This test mines a single block at height 1 on the main network
    /// and then validates that the header timestamp has changed and
    /// that it is within the 100 seconds (from now).
    ///
    /// This is a regression test for issue #149.
    /// https://github.com/Neptune-Crypto/neptune-core/issues/149
    ///
    /// note: this test fails in 318b7a20baf11a7a99f249660f1f70484c586012
    ///       and should always pass in later commits.
    #[traced_test]
    #[tokio::test]
    async fn block_timestamp_represents_time_block_found() -> Result<()> {
        let network = Network::Main;
        let global_state_lock = mock_genesis_global_state(
            network,
            2,
            WalletSecret::devnet_wallet(),
            cli_args::Args::default(),
        )
        .await;
        let (worker_task_tx, worker_task_rx) = oneshot::channel::<NewBlockFound>();

        let tip_block_orig = global_state_lock
            .lock_guard()
            .await
            .chain
            .light_state()
            .clone();

        let now = tip_block_orig.header().timestamp + Timestamp::minutes(10);

        // pretend/simulate that it takes at least 10 seconds to mine the block.
        let ten_seconds_ago = now - Timestamp::seconds(10);

        let (transaction, coinbase_utxo_info) =
            make_coinbase_transaction(&global_state_lock, 0f64, ten_seconds_ago)
                .await
                .unwrap();

        let template = Block::block_template_invalid_proof(
            &tip_block_orig,
            transaction,
            ten_seconds_ago,
            Digest::default(),
            None,
        );

        // sanity check that our initial state is correct.
        let initial_header_timestamp = template.header().timestamp;
        assert_eq!(ten_seconds_ago, initial_header_timestamp);

        let unrestricted_mining = true;

        guess_worker(
            template,
            tip_block_orig.clone(),
            worker_task_tx,
            coinbase_utxo_info,
            unrestricted_mining,
            None,
        );

        let mined_block_info = worker_task_rx.await.unwrap();

        let block_timestamp = mined_block_info.block.kernel.header.timestamp;

        // Mining updates the timestamp. So block timestamp will be >= to what
        // was set in the block template, and <= current time.
        assert!(block_timestamp >= initial_header_timestamp);
        assert!(block_timestamp <= Timestamp::now());

        // verify timestamp is within the last 100 seconds (allows for some CI slack).
        assert!(Timestamp::now() - block_timestamp < Timestamp::seconds(100));

        Ok(())
    }

    /// Test the difficulty adjustment algorithm.
    ///
    /// Specifically, verify that the observed concrete block interval when mining
    /// tracks the target block interval, assuming:
    ///  - No time is spent proving
    ///  - Constant mining power
    ///  - Mining power exceeds lower bound (hashing once every target interval).
    ///
    /// Note that the second assumption is broken when running the entire test suite.
    /// So if this test fails when all others pass, it is not necessarily a cause
    /// for worry.
    ///
    /// We mine ten blocks with a target block interval of 1 second, so all
    /// blocks should be mined in approx 10 seconds.
    ///
    /// We set a test time limit of 3x the expected time, ie 30 seconds, and
    /// panic if mining all blocks takes longer than that.
    ///
    /// We also assert upper and lower bounds for variance from the expected 10
    /// seconds.  The variance limit is 1.3, so the upper bound is 13 seconds
    /// and the lower bound is 7692ms.
    ///
    /// We ignore the first 2 blocks after genesis because they are typically
    /// mined very fast.
    ///
    /// We use unrestricted mining (100% CPU) to avoid complications from the
    /// sleep(100 millis) call in mining loop when restricted mining is enabled.
    ///
    /// This serves as a regression test for issue #154.
    /// https://github.com/Neptune-Crypto/neptune-core/issues/154
    async fn mine_m_blocks_in_n_seconds<const NUM_BLOCKS: usize, const NUM_SECONDS: usize>(
    ) -> Result<()> {
        let network = Network::RegTest;
        let global_state_lock = mock_genesis_global_state(
            network,
            2,
            WalletSecret::devnet_wallet(),
            cli_args::Args::default(),
        )
        .await;

        let mut prev_block = global_state_lock
            .lock_guard()
            .await
            .chain
            .light_state()
            .clone();

        // adjust these to simulate longer mining runs, possibly
        // with shorter or longer target intervals.
        // expected_duration = num_blocks * target_block_interval
        let target_block_interval =
            Timestamp::millis((1000.0 * (NUM_SECONDS as f64) / (NUM_BLOCKS as f64)).round() as u64);
        println!(
            "target block interval: {} ms",
            target_block_interval.0.value()
        );

        // set initial difficulty in accordance with own hash rate
        let unrestricted_mining = true;
        let hash_rate =
            estimate_own_hash_rate(Some(target_block_interval), unrestricted_mining).await;
        println!("estimating hash rate at {} per millisecond", hash_rate);
        let prepare_time = estimate_block_preparation_time_invalid_proof().await;
        println!("estimating block preparation time at {prepare_time} ms");
        if 1.5 * prepare_time > target_block_interval.0.value() as f64 {
            println!(
                "Cannot perform meaningful test! Block preparation time \
            too large for target block interval."
            );
            return Ok(());
        }

        let guessing_time = (target_block_interval.to_millis() as f64) - prepare_time;
        let initial_difficulty = BigUint::from((hash_rate * guessing_time) as u128);
        println!("initial difficulty: {}", initial_difficulty);
        prev_block.set_header_timestamp_and_difficulty(
            prev_block.header().timestamp,
            Difficulty::from_biguint(initial_difficulty),
        );

        let expected_duration = target_block_interval * NUM_BLOCKS;
        let stddev = (guessing_time.pow(2.0_f64) / (NUM_BLOCKS as f64)).sqrt();
        let allowed_standard_deviations = 4;
        let min_duration = (expected_duration.0.value() as f64)
            - (allowed_standard_deviations as f64) * stddev * (NUM_BLOCKS as f64);
        let max_duration = (expected_duration.0.value() as f64)
            + (allowed_standard_deviations as f64) * stddev * (NUM_BLOCKS as f64);
        let max_test_time = expected_duration * 3;

        // we ignore the first 2 blocks after genesis because they are
        // typically mined very fast.
        let ignore_first_n_blocks = 2;

        let mut durations = Vec::with_capacity(NUM_BLOCKS);
        let mut start_instant = std::time::SystemTime::now();

        for i in 0..NUM_BLOCKS + ignore_first_n_blocks {
            if i <= ignore_first_n_blocks {
                start_instant = std::time::SystemTime::now();
            }

            let start_time = Timestamp::now();
            let start_st = std::time::SystemTime::now();

            let (transaction, composer_utxos) = {
                (
                    make_mock_transaction_with_mutator_set_hash(
                        vec![],
                        vec![],
                        prev_block.mutator_set_accumulator().hash(),
                    ),
                    vec![dummy_expected_utxo()],
                )
            };

            let block = Block::block_template_invalid_proof(
                &prev_block,
                transaction,
                start_time,
                Digest::default(),
                Some(target_block_interval),
            );

            let (worker_task_tx, worker_task_rx) = oneshot::channel::<NewBlockFound>();
            let height = block.header().height;

            guess_worker(
                block,
                prev_block.clone(),
                worker_task_tx,
                composer_utxos,
                unrestricted_mining,
                Some(target_block_interval),
            );

            let mined_block_info = worker_task_rx.await.unwrap();

            // note: this assertion often fails prior to fix for #154.
            // Also note that `is_valid` is a wrapper around `is_valid_extended`
            // which is the method we need here because it allows us to override
            // default values for the target block interval and the minimum
            // block interval.
            assert!(mined_block_info.block.has_proof_of_work(&prev_block,));

            prev_block = *mined_block_info.block;

            let block_time = start_st.elapsed()?.as_millis();
            println!(
                "Found block {} in {block_time} milliseconds; \
                difficulty was {}; total time elapsed so far: {} ms",
                height,
                BigUint::from(prev_block.header().difficulty),
                start_instant.elapsed()?.as_millis()
            );
            if i > ignore_first_n_blocks {
                durations.push(block_time as f64);
            }

            let elapsed = start_instant.elapsed()?.as_millis();
            if elapsed > max_test_time.0.value().into() {
                panic!(
                    "test time limit exceeded.  \
                expected_duration: {expected_duration}, \
                limit: {max_test_time}, actual: {elapsed}"
                );
            }
        }

        let actual_duration = start_instant.elapsed()?.as_millis() as u64;

        println!(
            "actual duration: {actual_duration}\n\
        expected duration: {expected_duration}\n\
        min_duration: {min_duration}\n\
        max_duration: {max_duration}\n\
        allowed deviation: {allowed_standard_deviations}"
        );
        println!(
            "average block time: {} whereas target: {}",
            durations.into_iter().sum::<f64>() / (NUM_BLOCKS as f64),
            target_block_interval
        );

        assert!((actual_duration as f64) > min_duration);
        assert!((actual_duration as f64) < max_duration);

        Ok(())
    }

    #[traced_test]
    #[tokio::test]
    async fn mine_20_blocks_in_40_seconds() -> Result<()> {
        mine_m_blocks_in_n_seconds::<20, 40>().await.unwrap();
        Ok(())
    }

    #[test]
    fn block_hash_relates_to_predecessor_difficulty() {
        let difficulty = 100u32;
        // Difficulty X means we expect X trials before success.
        // Modeling the process as a geometric distribution gives the
        // probability of success in a single trial, p = 1/X.
        // Then the probability of seeing k failures is (1-1/X)^k.
        // We want this to be five nines certain that we do get a success
        // after k trials, so this quantity must be less than 0.0001.
        // So: log_10 0.0001 = -4 > log_10 (1-1/X)^k = k * log_10 (1 - 1/X).
        // Difficulty 100 sets k = 917.
        let cofactor = (1.0 - (1.0 / (difficulty as f64))).log10();
        let k = (-4.0 / cofactor).ceil() as usize;

        let mut predecessor_header = random_block_header();
        predecessor_header.difficulty = Difficulty::from(difficulty);
        let predecessor_body = BlockBody::new(
            random_transaction_kernel(),
            random_mutator_set_accumulator(),
            random_mmra(),
            random_mmra(),
        );
        let appendix = BlockAppendix::default();
        let predecessor_block = Block::new(
            predecessor_header,
            predecessor_body,
            appendix.clone(),
            BlockProof::Invalid,
        );

        let mut successor_header = random_block_header();
        successor_header.prev_block_digest = predecessor_block.hash();
        // note that successor's difficulty is random
        let successor_body = BlockBody::new(
            random_transaction_kernel(),
            random_mutator_set_accumulator(),
            random_mmra(),
            random_mmra(),
        );

        let mut rng = thread_rng();
        let mut counter = 0;
        let mut successor_block = Block::new(
            successor_header.clone(),
            successor_body.clone(),
            appendix,
            BlockProof::Invalid,
        );
        loop {
            successor_block.set_header_nonce(rng.gen());

            if successor_block.has_proof_of_work(&predecessor_block) {
                break;
            }

            counter += 1;

            assert!(
                counter < k,
                "number of hash trials before finding valid pow exceeds statistical limit"
            )
        }
    }

    #[traced_test]
    #[tokio::test]
    async fn guesser_fees_are_added_to_mutator_set() {
        // Mine two blocks on top of the genesis block. Verify that the guesser
        // fee for the 1st block was added to the mutator set. The genesis
        // block awards no guesser fee.

        let mut rng = thread_rng();
        let network = Network::Main;
        let genesis_block = Block::genesis_block(network);
        assert!(
            genesis_block.guesser_fee_addition_records().is_empty(),
            "Genesis block has no guesser fee UTXOs"
        );

        let launch_date = genesis_block.header().timestamp;
        let in_seven_months = launch_date + Timestamp::months(7);
        let in_eight_months = launch_date + Timestamp::months(8);
        let alice_wallet = WalletSecret::devnet_wallet();
        let alice_key = alice_wallet.nth_generation_spending_key(0);
        let alice_address = alice_key.to_address();
        let mut alice =
            mock_genesis_global_state(network, 0, alice_wallet, cli_args::Args::default()).await;

        let output = TxOutput::offchain_native_currency(
            NeptuneCoins::new(4),
            rng.gen(),
            alice_address.into(),
        );
        let fee = NeptuneCoins::new(1);
        let (tx1, _) = alice
            .lock_guard()
            .await
            .create_transaction_with_prover_capability(
                vec![output.clone()].into(),
                alice_key.into(),
                UtxoNotificationMedium::OnChain,
                fee,
                in_seven_months,
                TxProvingCapability::PrimitiveWitness,
                &TritonVmJobQueue::dummy(),
            )
            .await
            .unwrap();

        let block1 = Block::block_template_invalid_proof(
            &genesis_block,
            tx1,
            in_seven_months,
            Digest::default(),
            None,
        );
        alice.set_new_tip(block1.clone()).await.unwrap();

        let (tx2, _) = alice
            .lock_guard()
            .await
            .create_transaction_with_prover_capability(
                vec![output].into(),
                alice_key.into(),
                UtxoNotificationMedium::OnChain,
                fee,
                in_eight_months,
                TxProvingCapability::PrimitiveWitness,
                &TritonVmJobQueue::dummy(),
            )
            .await
            .unwrap();

        let block2 =
            Block::block_template_invalid_proof(&block1, tx2, in_eight_months, rng.gen(), None);

        let new_addition_records = [
            block1.guesser_fee_addition_records(),
            block2.body().transaction_kernel.outputs.clone(),
        ]
        .concat();
        let mut ms = block1.mutator_set_accumulator().clone();
        let mutator_set_update = MutatorSetUpdate::new(
            block2.body().transaction_kernel.inputs.clone(),
            new_addition_records,
        );
        mutator_set_update.apply_to_accumulator(&mut ms).expect("applying mutator set update derived from block 2 to mutator set from block 1 should work");

        assert_eq!(ms.hash(), block2.mutator_set_accumulator().hash());
    }
}
