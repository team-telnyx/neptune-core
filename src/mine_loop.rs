use crate::models::blockchain::block::block_body::BlockBody;
use crate::models::blockchain::block::block_header::BlockHeader;
use crate::models::blockchain::block::block_height::BlockHeight;
use crate::models::blockchain::block::mutator_set_update::*;
use crate::models::blockchain::block::*;
use crate::models::blockchain::digest::ordered_digest::*;
use crate::models::blockchain::digest::*;
use crate::models::blockchain::shared::*;
use crate::models::blockchain::transaction::utxo::*;
use crate::models::blockchain::transaction::*;
use crate::models::channel::*;
use crate::models::shared::*;
use anyhow::{Context, Result};
use num_traits::identities::Zero;
use rand::thread_rng;
use secp256k1::{rand::rngs::OsRng, Secp256k1};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::select;
use tokio::sync::{mpsc, watch};
use tokio::time::{sleep, Duration};
use tracing::{info, instrument};
use twenty_first::amount::u32s::U32s;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::traits::GetRandomElements;
use twenty_first::util_types::mutator_set::addition_record::AdditionRecord;
use twenty_first::util_types::mutator_set::mutator_set_accumulator::MutatorSetAccumulator;
use twenty_first::util_types::mutator_set::mutator_set_trait::MutatorSet;

const MOCK_REGTEST_MINIMUM_MINE_INTERVAL_SECONDS: u64 = 8;
const MOCK_REGTEST_MAX_MINING_DIFFERENCE_SECONDS: u64 = 8;
const MOCK_MAX_BLOCK_SIZE: u32 = 1_000_000;
const MOCK_DIFFICULTY: u32 = 1_000;

/// Return a fake block with a random hash
fn make_mock_block(height: u64, current_block_digest: Digest) -> Block {
    // TODO: Replace this with public key sent from the main thread
    let secp = Secp256k1::new();
    let mut rng = OsRng::new().expect("OsRng");
    let (_secret_key, public_key): (secp256k1::SecretKey, secp256k1::PublicKey) =
        secp.generate_keypair(&mut rng);

    let block_height: BlockHeight = height.into();
    let coinbase_utxo = Utxo {
        amount: Block::get_mining_reward(block_height),
        public_key,
    };

    let timestamp: BFieldElement = BFieldElement::new(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Got bad time timestamp in mining process")
            .as_secs(),
    );

    let output_randomness: Vec<BFieldElement> =
        BFieldElement::random_elements(RESCUE_PRIME_OUTPUT_SIZE_IN_BFES, &mut thread_rng());
    let tx = Transaction {
        inputs: vec![],
        outputs: vec![(coinbase_utxo.clone(), output_randomness.clone().into())],
        public_scripts: vec![],
        fee: U32s::zero(),
        timestamp,
    };

    // For now, we just assume that the mutator set was empty prior to this block
    let mut new_ms = MutatorSetAccumulator::default();

    let coinbase_digest: Digest = coinbase_utxo.hash();

    let coinbase_addition_record: AdditionRecord<Hash> =
        new_ms.commit(&coinbase_digest.into(), &output_randomness);
    let mutator_set_update: MutatorSetUpdate = MutatorSetUpdate {
        removals: vec![],
        additions: vec![coinbase_addition_record.clone()],
    };
    new_ms.add(&coinbase_addition_record);

    let block_body: BlockBody = BlockBody {
        transactions: vec![tx],
        next_mutator_set_accumulator: new_ms.clone(),
        mutator_set_update,
        previous_mutator_set_accumulator: MutatorSetAccumulator::default(),
        stark_proof: vec![],
    };

    let zero = BFieldElement::ring_zero();
    let difficulty: U32s<5> = U32s::new([MOCK_DIFFICULTY, 0, 0, 0, 0]);
    let mut block_header = BlockHeader {
        version: zero,
        height: BlockHeight::from(height),
        mutator_set_commitment: new_ms.get_commitment().into(),
        prev_block_digest: current_block_digest,
        timestamp,
        nonce: [zero, zero, zero],
        max_block_size: MOCK_MAX_BLOCK_SIZE,
        proof_of_work_line: U32s::zero(),
        proof_of_work_family: U32s::zero(),
        target_difficulty: difficulty,
        block_body_merkle_root: block_body.hash(),
        uncles: vec![],
    };

    // Mining takes place here
    while Into::<OrderedDigest>::into(block_header.hash())
        >= OrderedDigest::to_digest_threshold(difficulty)
    {
        if block_header.nonce[2].value() == BFieldElement::MAX {
            block_header.nonce[2] = BFieldElement::ring_zero();
            if block_header.nonce[1].value() == BFieldElement::MAX {
                block_header.nonce[1] = BFieldElement::ring_zero();
                block_header.nonce[0].increment();
                continue;
            }
            block_header.nonce[1].increment();
            continue;
        }
        block_header.nonce[2].increment();
    }
    info!(
        "Found valid block with nonce: ({}, {}, {})",
        block_header.nonce[0], block_header.nonce[1], block_header.nonce[2]
    );

    Block::new(block_header, block_body)
}

#[instrument]
pub async fn mock_regtest_mine(
    mut from_main: watch::Receiver<MainToMiner>,
    to_main: mpsc::Sender<MinerToMain>,
    latest_block_info: LatestBlockInfo,
) -> Result<()> {
    let (mut block_height, mut block_digest): (u64, Digest) =
        (latest_block_info.height.into(), latest_block_info.hash);
    loop {
        let rand_time: u64 = rand::random::<u64>() % MOCK_REGTEST_MAX_MINING_DIFFERENCE_SECONDS;
        select! {
            changed = from_main.changed() => {
                if let e@Err(_) = changed {
                    return e.context("Miner failed to read from watch channel");
                }

                let main_message: MainToMiner = from_main.borrow_and_update().clone();
                match main_message {
                    MainToMiner::NewBlock(block) => {
                        block_height = block.header.height.into();
                        info!("Miner thread received regtest block height {}", block_height);
                    }
                    MainToMiner::Empty => ()
                }
            }
            _ = sleep(Duration::from_secs(MOCK_REGTEST_MINIMUM_MINE_INTERVAL_SECONDS + rand_time)) => {
                block_height += 1;

                let new_fake_block = make_mock_block(block_height, block_digest);
                info!("Found new regtest block with block height {}. Hash: {:?}", new_fake_block.header.height, new_fake_block.hash);
                block_digest = new_fake_block.hash;
                to_main.send(MinerToMain::NewBlock(Box::new(new_fake_block))).await?;
            }
        }
    }
}
