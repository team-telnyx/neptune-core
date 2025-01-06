pub mod address;
pub mod monitored_utxo;
pub mod rusty_wallet_database;
pub mod utxo_notification_pool;
pub mod wallet_state;
pub mod wallet_status;

use anyhow::{bail, Context, Result};
use bip39::Mnemonic;
use itertools::Itertools;
use num_traits::Zero;
use rand::{thread_rng, Rng};
use serde::{Deserialize, Serialize};
use std::fs::{self};
use std::path::{Path, PathBuf};
use tracing::info;
use twenty_first::shared_math::bfield_codec::BFieldCodec;
use twenty_first::shared_math::digest::Digest;
use twenty_first::shared_math::x_field_element::XFieldElement;
use twenty_first::util_types::algebraic_hasher::AlgebraicHasher;
use zeroize::{Zeroize, ZeroizeOnDrop};

use twenty_first::shared_math::b_field_element::BFieldElement;

use crate::models::blockchain::block::block_height::BlockHeight;

use crate::Hash;

use self::address::generation_address;

pub const WALLET_DIRECTORY: &str = "wallet";
pub const WALLET_SECRET_FILE_NAME: &str = "wallet.dat";
pub const WALLET_OUTGOING_SECRETS_FILE_NAME: &str = "outgoing_randomness.dat";
pub const WALLET_INCOMING_SECRETS_FILE_NAME: &str = "incoming_randomness.dat";
const STANDARD_WALLET_NAME: &str = "standard_wallet";
const STANDARD_WALLET_VERSION: u8 = 0;
pub const WALLET_DB_NAME: &str = "wallet";
pub const WALLET_OUTPUT_COUNT_DB_NAME: &str = "wallout_output_count_db";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct SecretKeyMaterial(XFieldElement);

impl Zeroize for SecretKeyMaterial {
    fn zeroize(&mut self) {
        self.0 = XFieldElement::zero();
    }
}

/// Wallet contains the wallet-related data we want to store in a JSON file,
/// and that is not updated during regular program execution.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, ZeroizeOnDrop)]
pub struct WalletSecret {
    name: String,

    secret_seed: SecretKeyMaterial,
    version: u8,
}

/// Struct for containing file paths for secrets. To be communicated to user upon
/// wallet creation or wallet opening.
pub struct WalletSecretFileLocations {
    pub wallet_secret_path: PathBuf,
    pub incoming_randomness_file: PathBuf,
    pub outgoing_randomness_file: PathBuf,
}

impl WalletSecret {
    pub fn wallet_secret_path(wallet_directory_path: &Path) -> PathBuf {
        wallet_directory_path.join(WALLET_SECRET_FILE_NAME)
    }

    fn wallet_outgoing_secrets_path(wallet_directory_path: &Path) -> PathBuf {
        wallet_directory_path.join(WALLET_OUTGOING_SECRETS_FILE_NAME)
    }

    fn wallet_incoming_secrets_path(wallet_directory_path: &Path) -> PathBuf {
        wallet_directory_path.join(WALLET_INCOMING_SECRETS_FILE_NAME)
    }

    /// Create new `Wallet` given a `secret` key.
    fn new(secret_seed: SecretKeyMaterial) -> Self {
        Self {
            name: STANDARD_WALLET_NAME.to_string(),
            secret_seed,
            version: STANDARD_WALLET_VERSION,
        }
    }

    /// Create a new `Wallet` and populate it with a new secret seed, with entropy
    /// obtained via `thread_rng()` from the operating system.
    pub fn new_random() -> Self {
        Self {
            name: STANDARD_WALLET_NAME.to_string(),
            secret_seed: SecretKeyMaterial(thread_rng().gen()),
            version: STANDARD_WALLET_VERSION,
        }
    }

    /// Create a `Wallet` with a fixed digest
    pub fn devnet_wallet() -> Self {
        let secret_seed = SecretKeyMaterial(XFieldElement::new([
            BFieldElement::new(12063201067205522823),
            BFieldElement::new(1529663126377206632),
            BFieldElement::new(2090171368883726200),
        ]));

        WalletSecret::new(secret_seed)
    }

    /// Read wallet from `wallet_file` if the file exists, or, if none exists, create new wallet
    /// and save it to `wallet_file`.
    /// Also create files for incoming and outgoing randomness which should be appended to
    /// on each incoming and outgoing transaction.
    /// Returns an instance of self and the path in which the wallet secret was stored.
    pub fn read_from_file_or_create(
        wallet_directory_path: &Path,
    ) -> Result<(Self, WalletSecretFileLocations)> {
        let wallet_secret_path = Self::wallet_secret_path(wallet_directory_path);
        let wallet = if wallet_secret_path.exists() {
            info!(
                "***** Reading wallet from {} *****\n\n\n",
                wallet_secret_path.display()
            );
            Self::read_from_file(&wallet_secret_path)?
        } else {
            info!(
                "***** Creating new wallet in {} *****\n\n\n",
                wallet_secret_path.display()
            );
            let new_wallet: WalletSecret = WalletSecret::new_random();
            new_wallet.save_to_disk(&wallet_secret_path)?;
            new_wallet
        };

        // Generate files for outgoing and ingoing randomness if those files
        // do not already exist
        let outgoing_randomness_file: PathBuf =
            Self::wallet_outgoing_secrets_path(wallet_directory_path);
        if !outgoing_randomness_file.exists() {
            Self::create_empty_wallet_randomness_file(&outgoing_randomness_file).expect(
                "Create file for outgoing randomness must succeed. Attempted to create file: {outgoing_randomness_file}",
            );
        }

        let incoming_randomness_file = Self::wallet_incoming_secrets_path(wallet_directory_path);
        if !incoming_randomness_file.exists() {
            Self::create_empty_wallet_randomness_file(&incoming_randomness_file).expect("Create file for outgoing randomness must succeed. Attempted to create file: {incoming_randomness_file}");
        }

        // Sanity checks that files were actually created
        if !wallet_secret_path.exists() {
            bail!(
                "Wallet secret file '{}' must exist on disk after reading/creating it.",
                wallet_secret_path.to_string_lossy()
            );
        }
        if !outgoing_randomness_file.exists() {
            bail!(
                "file containing outgoing randomness '{}' must exist on disk.",
                outgoing_randomness_file.to_string_lossy()
            );
        }
        if !incoming_randomness_file.exists() {
            bail!(
                "file containing ingoing randomness '{}' must exist on disk.",
                incoming_randomness_file.to_string_lossy()
            );
        }

        let wallet_secret_file_locations = WalletSecretFileLocations {
            wallet_secret_path,
            incoming_randomness_file,
            outgoing_randomness_file,
        };

        Ok((wallet, wallet_secret_file_locations))
    }

    pub fn nth_generation_spending_key(&self, counter: u16) -> generation_address::SpendingKey {
        assert!(
            counter.is_zero(),
            "For now we only support one generation address per wallet"
        );
        self.nth_generation_spending_key_worker(counter)
    }

    fn nth_generation_spending_key_worker(&self, counter: u16) -> generation_address::SpendingKey {
        // We keep n between 0 and 2^16 as this makes it possible to scan all possible addresses
        // in case you don't know with what counter you made the address
        let key_seed = Hash::hash_varlen(
            &[
                self.secret_seed.0.encode(),
                vec![
                    generation_address::GENERATION_FLAG,
                    BFieldElement::new(counter.into()),
                ],
            ]
            .concat(),
        );
        generation_address::SpendingKey::derive_from_seed(key_seed)
    }

    /// Return the secret key that is used to deterministically generate commitment pseudo-randomness
    /// for the mutator set.
    pub fn generate_sender_randomness(
        &self,
        block_height: BlockHeight,
        receiver_digest: Digest,
    ) -> Digest {
        const SENDER_RANDOMNESS_FLAG: u64 = 0x5e116e1270u64;
        Hash::hash_varlen(
            &[
                self.secret_seed.0.encode(),
                vec![
                    BFieldElement::new(SENDER_RANDOMNESS_FLAG),
                    block_height.into(),
                ],
                receiver_digest.encode(),
            ]
            .concat(),
        )
    }

    /// Read Wallet from file as JSON
    pub fn read_from_file(wallet_file: &Path) -> Result<Self> {
        let wallet_file_content: String = fs::read_to_string(wallet_file).with_context(|| {
            format!(
                "Failed to read wallet from {}",
                wallet_file.to_string_lossy(),
            )
        })?;

        serde_json::from_str::<WalletSecret>(&wallet_file_content).with_context(|| {
            format!(
                "Failed to decode wallet from {}",
                wallet_file.to_string_lossy(),
            )
        })
    }

    /// Used to generate both the file for incoming and outgoing randomness
    fn create_empty_wallet_randomness_file(file_path: &Path) -> Result<()> {
        let init_value: String = String::default();

        #[cfg(unix)]
        {
            Self::create_wallet_file_unix(&file_path.to_path_buf(), init_value)
        }
        #[cfg(not(unix))]
        {
            Self::create_wallet_file_windows(&file_path.to_path_buf(), init_value)
        }
    }

    /// Save this wallet to disk. If necessary, create the file (with restrictive permissions).
    pub fn save_to_disk(&self, wallet_file: &Path) -> Result<()> {
        let wallet_secret_as_json: String = serde_json::to_string(self).unwrap();

        #[cfg(unix)]
        {
            Self::create_wallet_file_unix(&wallet_file.to_path_buf(), wallet_secret_as_json)
        }
        #[cfg(not(unix))]
        {
            Self::create_wallet_file_windows(&wallet_file.to_path_buf(), wallet_secret_as_json)
        }
    }

    #[cfg(unix)]
    /// Create a wallet file, and set restrictive permissions
    fn create_wallet_file_unix(path: &PathBuf, file_content: String) -> Result<()> {
        // On Unix/Linux we set the file permissions to 600, to disallow
        // other users on the same machine to access the secrets.
        // I don't think the `std::os::unix` library can be imported on a Windows machine,
        // so this function and the below import is only compiled on Unix machines.
        use std::os::unix::prelude::OpenOptionsExt;
        fs::OpenOptions::new()
            .create(true)
            .write(true)
            .mode(0o600)
            .open(path)
            .unwrap();
        fs::write(path.clone(), file_content).context("Failed to write wallet file to disk")
    }

    #[cfg(not(unix))]
    /// Create a wallet file, without setting restrictive UNIX permissions
    fn create_wallet_file_windows(path: &PathBuf, wallet_as_json: String) -> Result<()> {
        fs::OpenOptions::new()
            .create(true)
            .write(true)
            .open(path)
            .unwrap();
        fs::write(path.clone(), wallet_as_json).context("Failed to write wallet file to disk")
    }

    /// Convert the wallet secret into a BIP-39 phrase consisting of 18 words (for 192
    /// bits of entropy).
    pub fn to_phrase(&self) -> Vec<String> {
        let entropy = self
            .secret_seed
            .0
            .coefficients
            .iter()
            .flat_map(|bfe| bfe.value().to_le_bytes())
            .collect_vec();
        assert_eq!(
            entropy.len(),
            24,
            "Entropy for secret seed does not consist of 24 bytes."
        );
        let mnemonic = Mnemonic::from_entropy(&entropy, bip39::Language::English)
            .expect("Wrong entropy length (should be 24 bytes).");
        mnemonic
            .phrase()
            .split(' ')
            .map(|s| s.to_string())
            .collect_vec()
    }

    /// Convert a secret seed phrase (list of 18 valid BIP-39 words) to a WalletSecret
    pub fn from_phrase(phrase: &[String]) -> Result<Self> {
        let mnemonic = Mnemonic::from_phrase(&phrase.iter().join(" "), bip39::Language::English)?;
        let secret_seed: [u8; 24] = mnemonic.entropy().try_into().unwrap();
        let xfe = XFieldElement::new(
            secret_seed
                .chunks(8)
                .map(|ch| u64::from_le_bytes(ch.try_into().unwrap()))
                .map(BFieldElement::new)
                .collect_vec()
                .try_into()
                .unwrap(),
        );
        Ok(Self::new(SecretKeyMaterial(xfe)))
    }
}

#[cfg(test)]
mod wallet_tests {
    use itertools::Itertools;
    use num_traits::CheckedSub;
    use rand::random;
    use tracing_test::traced_test;
    use twenty_first::shared_math::tip5::DIGEST_LENGTH;
    use twenty_first::shared_math::x_field_element::EXTENSION_DEGREE;
    use twenty_first::util_types::storage_vec::StorageVec;

    use super::monitored_utxo::MonitoredUtxo;
    use super::wallet_state::WalletState;
    use super::*;
    use crate::config_models::network::Network;
    use crate::models::blockchain::block::block_height::BlockHeight;
    use crate::models::blockchain::block::Block;
    use crate::models::blockchain::shared::Hash;
    use crate::models::blockchain::transaction::amount::{Amount, AmountLike};
    use crate::models::blockchain::transaction::utxo::{LockScript, Utxo};
    use crate::models::blockchain::transaction::PubScript;
    use crate::models::state::wallet::utxo_notification_pool::UtxoNotifier;
    use crate::models::state::UtxoReceiverData;
    use crate::tests::shared::{
        add_block, get_mock_global_state, get_mock_wallet_state, make_mock_block,
        make_mock_transaction_with_generation_key,
    };
    use crate::util_types::mutator_set::mutator_set_trait::MutatorSet;

    async fn get_monitored_utxos(wallet_state: &WalletState) -> Vec<MonitoredUtxo> {
        let lock = wallet_state.wallet_db.lock().await;
        let num_monitored_utxos = lock.monitored_utxos.len();
        let mut monitored_utxos = vec![];
        for i in 0..num_monitored_utxos {
            monitored_utxos.push(lock.monitored_utxos.get(i));
        }
        monitored_utxos
    }

    #[tokio::test]
    async fn wallet_state_constructor_with_genesis_block_test() -> Result<()> {
        // This test is designed to verify that the genesis block is applied
        // to the wallet state at initialization.
        let network = Network::Testnet;
        let wallet_state_premine_recipient = get_mock_wallet_state(None, network).await;
        let monitored_utxos_premine_wallet =
            get_monitored_utxos(&wallet_state_premine_recipient).await;
        assert_eq!(
            1,
            monitored_utxos_premine_wallet.len(),
            "Monitored UTXO list must contain premined UTXO at init, for premine-wallet"
        );

        let premine_receiver_spending_key = wallet_state_premine_recipient
            .wallet_secret
            .nth_generation_spending_key(0);
        let premine_receiver_address = premine_receiver_spending_key.to_address();
        let expected_premine_utxo = Utxo {
            coins: Block::premine_distribution()[0].1.to_native_coins(),
            lock_script_hash: premine_receiver_address.lock_script().hash(),
        };
        assert_eq!(
            expected_premine_utxo, monitored_utxos_premine_wallet[0].utxo,
            "Auth wallet's monitored UTXO must match that from genesis block at initialization"
        );

        let random_wallet = WalletSecret::new_random();
        let wallet_state_other = get_mock_wallet_state(Some(random_wallet), network).await;
        let monitored_utxos_other = get_monitored_utxos(&wallet_state_other).await;
        assert!(
            monitored_utxos_other.is_empty(),
            "Monitored UTXO list must be empty at init if wallet is not premine-wallet"
        );

        // Add 12 blocks and verify that membership proofs are still valid
        let genesis_block = Block::genesis_block();
        let mut next_block = genesis_block.clone();
        let other_wallet_secret = WalletSecret::new_random();
        let other_receiver_address = other_wallet_secret
            .nth_generation_spending_key(0)
            .to_address();
        for _ in 0..12 {
            let previous_block = next_block;
            let (nb, _coinbase_utxo, _sender_randomness) =
                make_mock_block(&previous_block, None, other_receiver_address);
            next_block = nb;
            wallet_state_premine_recipient.update_wallet_state_with_new_block(
                &next_block,
                &mut wallet_state_premine_recipient.wallet_db.lock().await,
            )?;
        }

        let monitored_utxos = get_monitored_utxos(&wallet_state_premine_recipient).await;
        assert_eq!(
            1,
            monitored_utxos.len(),
            "monitored UTXOs must be 1 after applying N blocks not mined by wallet"
        );

        let genesis_block_output_utxo = monitored_utxos[0].utxo.clone();
        let ms_membership_proof = monitored_utxos[0]
            .get_membership_proof_for_block(next_block.hash)
            .unwrap();
        assert!(
            next_block
                .body
                .next_mutator_set_accumulator
                .verify(Hash::hash(&genesis_block_output_utxo), &ms_membership_proof),
            "Membership proof must be valid after updating wallet state with generated blocks"
        );

        Ok(())
    }

    #[tokio::test]
    async fn wallet_state_registration_of_monitored_utxos_test() -> Result<()> {
        let network = Network::Testnet;
        let own_wallet_secret = WalletSecret::new_random();
        let own_wallet_state =
            get_mock_wallet_state(Some(own_wallet_secret.clone()), network).await;
        let other_wallet_secret = WalletSecret::new_random();
        let other_recipient_address = other_wallet_secret
            .nth_generation_spending_key(0)
            .to_address();

        let mut monitored_utxos = get_monitored_utxos(&own_wallet_state).await;
        assert!(
            monitored_utxos.is_empty(),
            "Monitored UTXO list must be empty at init"
        );

        let genesis_block = Block::genesis_block();
        let own_spending_key = own_wallet_secret.nth_generation_spending_key(0);
        let own_recipient_address = own_spending_key.to_address();
        let (block_1, block_1_coinbase_utxo, block_1_coinbase_sender_randomness) =
            make_mock_block(&genesis_block, None, own_recipient_address);

        own_wallet_state
            .expected_utxos
            .write()
            .unwrap()
            .add_expected_utxo(
                block_1_coinbase_utxo.clone(),
                block_1_coinbase_sender_randomness,
                own_spending_key.privacy_preimage,
                UtxoNotifier::OwnMiner,
            )
            .unwrap();
        assert_eq!(
            1,
            own_wallet_state.expected_utxos.read().unwrap().len(),
            "Expected UTXO list must have length 1 before block registration"
        );
        own_wallet_state.update_wallet_state_with_new_block(
            &block_1,
            &mut own_wallet_state.wallet_db.lock().await,
        )?;
        assert_eq!(
            1,
            own_wallet_state.expected_utxos.read().unwrap().len(),
            "A: Expected UTXO list must have length 1 after block registration, due to potential reorganizations");
        let expected_utxos = own_wallet_state
            .expected_utxos
            .read()
            .unwrap()
            .get_all_expected_utxos();
        assert_eq!(1, expected_utxos.len(), "B: Expected UTXO list must have length 1 after block registration, due to potential reorganizations");
        assert_eq!(
            block_1.hash,
            expected_utxos[0].mined_in_block.unwrap().0,
            "Expected UTXO must be registered as being mined"
        );
        monitored_utxos = get_monitored_utxos(&own_wallet_state).await;
        assert_eq!(
            1,
            monitored_utxos.len(),
            "Monitored UTXO list be one after we mined a block"
        );

        // Ensure that the membership proof is valid
        {
            let block_1_tx_output_digest = Hash::hash(&block_1_coinbase_utxo);
            let ms_membership_proof = monitored_utxos[0]
                .get_membership_proof_for_block(block_1.hash)
                .unwrap();
            let membership_proof_is_valid = block_1
                .body
                .next_mutator_set_accumulator
                .verify(block_1_tx_output_digest, &ms_membership_proof);
            assert!(membership_proof_is_valid);
        }

        // Create new blocks, verify that the membership proofs are *not* valid
        // under this block as tip
        let (block_2, _, _) = make_mock_block(&block_1, None, other_recipient_address);
        let (block_3, _, _) = make_mock_block(&block_2, None, other_recipient_address);
        monitored_utxos = get_monitored_utxos(&own_wallet_state).await;
        {
            let block_1_tx_output_digest = Hash::hash(&block_1_coinbase_utxo);
            let ms_membership_proof = monitored_utxos[0]
                .get_membership_proof_for_block(block_1.hash)
                .unwrap();
            let membership_proof_is_valid = block_3
                .body
                .next_mutator_set_accumulator
                .verify(block_1_tx_output_digest, &ms_membership_proof);
            assert!(
                !membership_proof_is_valid,
                "membership proof must be invalid before updating wallet state"
            );
        }
        // Verify that the membership proof is valid *after* running the updater
        own_wallet_state.update_wallet_state_with_new_block(
            &block_2,
            &mut own_wallet_state.wallet_db.lock().await,
        )?;
        own_wallet_state.update_wallet_state_with_new_block(
            &block_3,
            &mut own_wallet_state.wallet_db.lock().await,
        )?;
        monitored_utxos = get_monitored_utxos(&own_wallet_state).await;

        {
            let block_1_tx_output_digest = Hash::hash(&block_1_coinbase_utxo);
            let ms_membership_proof = monitored_utxos[0]
                .get_membership_proof_for_block(block_3.hash)
                .unwrap();
            let membership_proof_is_valid = block_3
                .body
                .next_mutator_set_accumulator
                .verify(block_1_tx_output_digest, &ms_membership_proof);
            assert!(
                membership_proof_is_valid,
                "Membership proof must be valid after updating wallet state with generated blocks"
            );
        }

        Ok(())
    }

    #[traced_test]
    #[tokio::test]
    async fn allocate_sufficient_input_funds_test() -> Result<()> {
        let own_wallet_secret = WalletSecret::new_random();
        let network = Network::Testnet;
        let own_wallet_state = get_mock_wallet_state(Some(own_wallet_secret), network).await;
        let own_spending_key = own_wallet_state
            .wallet_secret
            .nth_generation_spending_key(0);
        let genesis_block = Block::genesis_block();
        let (block_1, cb_utxo, cb_output_randomness) =
            make_mock_block(&genesis_block, None, own_spending_key.to_address());
        let mining_reward = cb_utxo.get_native_coin_amount();

        // Add block to wallet state
        own_wallet_state
            .expected_utxos
            .write()
            .unwrap()
            .add_expected_utxo(
                cb_utxo,
                cb_output_randomness,
                own_spending_key.privacy_preimage,
                UtxoNotifier::OwnMiner,
            )
            .unwrap();
        own_wallet_state.update_wallet_state_with_new_block(
            &block_1,
            &mut own_wallet_state.wallet_db.lock().await,
        )?;

        // Verify that the allocater returns a sane amount
        assert_eq!(
            1,
            own_wallet_state
                .allocate_sufficient_input_funds(Amount::one(), block_1.hash)
                .await
                .unwrap()
                .len()
        );
        assert_eq!(
            1,
            own_wallet_state
                .allocate_sufficient_input_funds(
                    mining_reward.checked_sub(&Amount::one()).unwrap(),
                    block_1.hash
                )
                .await
                .unwrap()
                .len()
        );
        assert_eq!(
            1,
            own_wallet_state
                .allocate_sufficient_input_funds(mining_reward, block_1.hash)
                .await
                .unwrap()
                .len()
        );

        // Cannot allocate more than we have: `mining_reward`
        assert!(own_wallet_state
            .allocate_sufficient_input_funds(mining_reward + Amount::one(), block_1.hash)
            .await
            .is_err());

        // Mine 21 more blocks and verify that 22 * `mining_reward` worth of UTXOs can be allocated
        let mut next_block = block_1.clone();
        for _ in 0..21 {
            let previous_block = next_block;
            let (next_block_prime, cb_utxo_prime, cb_output_randomness_prime) =
                make_mock_block(&previous_block, None, own_spending_key.to_address());
            own_wallet_state
                .expected_utxos
                .write()
                .unwrap()
                .add_expected_utxo(
                    cb_utxo_prime,
                    cb_output_randomness_prime,
                    own_spending_key.privacy_preimage,
                    UtxoNotifier::OwnMiner,
                )
                .unwrap();
            own_wallet_state.update_wallet_state_with_new_block(
                &next_block_prime,
                &mut own_wallet_state.wallet_db.lock().await,
            )?;
            next_block = next_block_prime;
        }

        assert_eq!(
            5,
            own_wallet_state
                .allocate_sufficient_input_funds(mining_reward.scalar_mul(5), next_block.hash)
                .await
                .unwrap()
                .len()
        );
        assert_eq!(
            6,
            own_wallet_state
                .allocate_sufficient_input_funds(
                    mining_reward.scalar_mul(5) + Amount::one(),
                    next_block.hash
                )
                .await
                .unwrap()
                .len()
        );

        let expected_balance = mining_reward.scalar_mul(22);
        assert_eq!(
            22,
            own_wallet_state
                .allocate_sufficient_input_funds(expected_balance, next_block.hash)
                .await
                .unwrap()
                .len()
        );

        // Cannot allocate more than we have: 22 * mining reward
        assert!(own_wallet_state
            .allocate_sufficient_input_funds(expected_balance + Amount::one(), next_block.hash)
            .await
            .is_err());

        // Make a block that spends an input, then verify that this is reflected by
        // the allocator.
        let two_utxos = own_wallet_state
            .allocate_sufficient_input_funds(mining_reward.scalar_mul(2), next_block.hash)
            .await
            .unwrap();
        assert_eq!(
            2,
            two_utxos.len(),
            "Must use two UTXOs when sending 2 x mining reward"
        );

        // This block spends two UTXOs and gives us none, so the new balance
        // becomes 2000
        let other_wallet = WalletSecret::new_random();
        let other_wallet_recipient_address =
            other_wallet.nth_generation_spending_key(0).to_address();
        assert_eq!(Into::<BlockHeight>::into(22u64), next_block.header.height);
        (next_block, _, _) =
            make_mock_block(&next_block.clone(), None, own_spending_key.to_address());
        assert_eq!(Into::<BlockHeight>::into(23u64), next_block.header.height);
        let msa_tip_previous = next_block.body.previous_mutator_set_accumulator.clone();

        let receiver_data = vec![UtxoReceiverData {
            utxo: Utxo {
                lock_script_hash: LockScript::anyone_can_spend().hash(),
                coins: Into::<Amount>::into(200).to_native_coins(),
            },
            sender_randomness: random(),
            receiver_privacy_digest: other_wallet_recipient_address.privacy_digest,
            pubscript: PubScript::default(),
            pubscript_input: vec![],
        }];
        let input_utxos_mps_keys = two_utxos
            .into_iter()
            .map(|(utxo, _lock_script, mp)| (utxo, mp, own_spending_key))
            .collect_vec();
        let tx = make_mock_transaction_with_generation_key(
            input_utxos_mps_keys,
            receiver_data,
            Amount::zero(),
            msa_tip_previous,
        );
        next_block.accumulate_transaction(tx);

        own_wallet_state.update_wallet_state_with_new_block(
            &next_block,
            &mut own_wallet_state.wallet_db.lock().await,
        )?;

        assert_eq!(
            20,
            own_wallet_state
                .allocate_sufficient_input_funds(2000.into(), next_block.hash)
                .await
                .unwrap()
                .len()
        );

        // Cannot allocate more than we have: 2000
        assert!(own_wallet_state
            .allocate_sufficient_input_funds(2001.into(), next_block.hash)
            .await
            .is_err());

        Ok(())
    }

    #[traced_test]
    #[tokio::test]
    async fn wallet_state_maintanence_multiple_inputs_outputs_test() -> Result<()> {
        // An archival state is needed for how we currently add inputs to a transaction.
        // So it's just used to generate test data, not in any of the functions that are
        // actually tested.
        let network = Network::Alpha;
        let own_wallet_secret = WalletSecret::new_random();
        let own_wallet_state = get_mock_wallet_state(Some(own_wallet_secret), network).await;
        let own_spending_key = own_wallet_state
            .wallet_secret
            .nth_generation_spending_key(0);
        let own_address = own_spending_key.to_address();
        let genesis_block = Block::genesis_block();
        let premine_wallet = get_mock_wallet_state(None, network).await.wallet_secret;
        let premine_receiver_global_state =
            get_mock_global_state(Network::Alpha, 2, Some(premine_wallet)).await;
        let preminers_original_balance = premine_receiver_global_state
            .get_wallet_status_for_tip()
            .await
            .synced_unspent_amount;
        assert!(
            !preminers_original_balance.is_zero(),
            "Premine must have non-zero synced balance"
        );

        let (mut block_1, _, _) = make_mock_block(&genesis_block, None, own_address);

        let receiver_data_12_to_other = UtxoReceiverData {
            pubscript: PubScript::default(),
            pubscript_input: vec![],
            receiver_privacy_digest: own_address.privacy_digest,
            sender_randomness: premine_receiver_global_state
                .wallet_state
                .wallet_secret
                .generate_sender_randomness(
                    genesis_block.header.height,
                    own_address.privacy_digest,
                ),
            utxo: Utxo {
                coins: Into::<Amount>::into(12).to_native_coins(),
                lock_script_hash: own_address.lock_script().hash(),
            },
        };
        let receiver_data_one_to_other = UtxoReceiverData {
            pubscript: PubScript::default(),
            pubscript_input: vec![],
            receiver_privacy_digest: own_address.privacy_digest,
            sender_randomness: premine_receiver_global_state
                .wallet_state
                .wallet_secret
                .generate_sender_randomness(
                    genesis_block.header.height,
                    own_address.privacy_digest,
                ),
            utxo: Utxo {
                coins: Into::<Amount>::into(1).to_native_coins(),
                lock_script_hash: own_address.lock_script().hash(),
            },
        };
        let receiver_data_to_other = vec![receiver_data_12_to_other, receiver_data_one_to_other];
        let valid_tx = premine_receiver_global_state
            .create_transaction(receiver_data_to_other.clone(), Into::<Amount>::into(2))
            .await
            .unwrap();

        block_1.accumulate_transaction(valid_tx);

        // Verify the validity of the merged transaction and block
        assert!(block_1.is_valid(&genesis_block));

        // Update wallet state with block_1
        let mut monitored_utxos = get_monitored_utxos(&own_wallet_state).await;
        assert!(
            monitored_utxos.is_empty(),
            "List of monitored UTXOs must be empty prior to updating wallet state"
        );

        // Expect the UTXO outputs
        for receive_data in receiver_data_to_other {
            own_wallet_state
                .expected_utxos
                .write()
                .unwrap()
                .add_expected_utxo(
                    receive_data.utxo,
                    receive_data.sender_randomness,
                    own_spending_key.privacy_preimage,
                    UtxoNotifier::Cli,
                )
                .unwrap();
        }
        own_wallet_state.update_wallet_state_with_new_block(
            &block_1,
            &mut own_wallet_state.wallet_db.lock().await,
        )?;
        add_block(&premine_receiver_global_state, block_1.clone())
            .await
            .unwrap();
        premine_receiver_global_state
            .wallet_state
            .update_wallet_state_with_new_block(
                &block_1,
                &mut premine_receiver_global_state
                    .wallet_state
                    .wallet_db
                    .lock()
                    .await,
            )?;
        assert_eq!(
            preminers_original_balance
                .checked_sub(&Into::<Amount>::into(15))
                .unwrap(),
            premine_receiver_global_state
                .get_wallet_status_for_tip()
                .await
                .synced_unspent_amount,
            "Preminer must have spent 15: 12 + 1 for sent, 2 for fees"
        );

        // Verify that update added 4 UTXOs to list of monitored transactions:
        // three as regular outputs, and one as coinbase UTXO
        monitored_utxos = get_monitored_utxos(&own_wallet_state).await;
        assert_eq!(
            2,
            monitored_utxos.len(),
            "List of monitored UTXOs have length 4 after updating wallet state"
        );

        // Verify that all monitored UTXOs have valid membership proofs
        for monitored_utxo in monitored_utxos {
            assert!(
                block_1.body.next_mutator_set_accumulator.verify(
                    Hash::hash(&monitored_utxo.utxo),
                    &monitored_utxo
                        .get_membership_proof_for_block(block_1.hash)
                        .unwrap()
                ),
                "All membership proofs must be valid after block 1"
            )
        }

        // Add 17 blocks (mined by us)
        // and verify that all membership proofs are still valid
        let mut next_block = block_1.clone();
        for _ in 0..17 {
            let previous_block = next_block;
            let ret = make_mock_block(&previous_block, None, own_address);
            next_block = ret.0;
            own_wallet_state
                .expected_utxos
                .write()
                .unwrap()
                .add_expected_utxo(
                    ret.1,
                    ret.2,
                    own_spending_key.privacy_preimage,
                    UtxoNotifier::OwnMiner,
                )
                .unwrap();
            own_wallet_state.update_wallet_state_with_new_block(
                &next_block,
                &mut own_wallet_state.wallet_db.lock().await,
            )?;
            add_block(&premine_receiver_global_state, block_1.clone())
                .await
                .unwrap();
            premine_receiver_global_state
                .wallet_state
                .update_wallet_state_with_new_block(
                    &next_block,
                    &mut premine_receiver_global_state
                        .wallet_state
                        .wallet_db
                        .lock()
                        .await,
                )?;
        }

        let block_18 = next_block;
        monitored_utxos = get_monitored_utxos(&own_wallet_state).await;
        assert_eq!(
                2 + 17,
                monitored_utxos.len(),
                "List of monitored UTXOs have length 19 after updating wallet state and mining 17 blocks"
            );
        for monitored_utxo in monitored_utxos {
            assert!(
                block_18.body.next_mutator_set_accumulator.verify(
                    Hash::hash(&monitored_utxo.utxo),
                    &monitored_utxo
                        .get_membership_proof_for_block(block_18.hash)
                        .unwrap()
                ),
                "All membership proofs must be valid after block 18"
            )
        }

        // Sanity check
        assert_eq!(
            Into::<BlockHeight>::into(18u64),
            block_18.header.height,
            "Block height must be 18 after genesis and 18 blocks being mined"
        );

        // Check that `WalletStatus` is returned correctly
        let wallet_status = {
            let mut wallet_db_lock = own_wallet_state.wallet_db.lock().await;
            own_wallet_state.get_wallet_status_from_lock(&mut wallet_db_lock, block_18.hash)
        };
        assert_eq!(
            19,
            wallet_status.synced_unspent.len(),
            "Wallet must have 19 synced, unspent UTXOs"
        );
        assert!(
            wallet_status.synced_spent.is_empty(),
            "Wallet must have 0 synced, spent UTXOs"
        );
        assert!(
            wallet_status.unsynced_spent.is_empty(),
            "Wallet must have 0 unsynced spent UTXOs"
        );
        assert!(
            wallet_status.unsynced_unspent.is_empty(),
            "Wallet must have 0 unsynced unspent UTXOs"
        );

        // verify that membership proofs are valid after forks
        let premine_wallet_spending_key = premine_receiver_global_state
            .wallet_state
            .wallet_secret
            .nth_generation_spending_key(0);
        let (block_2_b, _, _) =
            make_mock_block(&block_1, None, premine_wallet_spending_key.to_address());
        own_wallet_state.update_wallet_state_with_new_block(
            &block_2_b,
            &mut own_wallet_state.wallet_db.lock().await,
        )?;
        add_block(&premine_receiver_global_state, block_2_b.clone())
            .await
            .unwrap();
        premine_receiver_global_state
            .wallet_state
            .update_wallet_state_with_new_block(
                &block_2_b,
                &mut premine_receiver_global_state
                    .wallet_state
                    .wallet_db
                    .lock()
                    .await,
            )
            .unwrap();
        let monitored_utxos_at_2b: Vec<_> = get_monitored_utxos(&own_wallet_state)
            .await
            .into_iter()
            .filter(|x| x.is_synced_to(block_2_b.hash))
            .collect();
        assert_eq!(
            2,
            monitored_utxos_at_2b.len(),
            "List of synced monitored UTXOs have length 2 after updating wallet state"
        );

        // Verify that all monitored UTXOs (with synced MPs) have valid membership proofs
        for monitored_utxo in monitored_utxos_at_2b.iter() {
            assert!(
                block_2_b.body.next_mutator_set_accumulator.verify(
                    Hash::hash(&monitored_utxo.utxo),
                    &monitored_utxo
                        .get_membership_proof_for_block(block_2_b.hash)
                        .unwrap()
                ),
                "All synced membership proofs must be valid after block 2b fork"
            )
        }

        // Fork back again to the long chain and verify that the membership proofs
        // all work again
        let (block_19, _, _) =
            make_mock_block(&block_18, None, premine_wallet_spending_key.to_address());
        own_wallet_state.update_wallet_state_with_new_block(
            &block_19,
            &mut own_wallet_state.wallet_db.lock().await,
        )?;
        let monitored_utxos_block_19: Vec<_> = get_monitored_utxos(&own_wallet_state)
            .await
            .into_iter()
            .filter(|monitored_utxo| monitored_utxo.is_synced_to(block_19.hash))
            .collect();
        assert_eq!(
            2 + 17,
            monitored_utxos_block_19.len(),
            "List of monitored UTXOs have length 19 after returning to good fork"
        );

        // Verify that all monitored UTXOs have valid membership proofs
        for monitored_utxo in monitored_utxos_block_19.iter() {
            assert!(
                block_19.body.next_mutator_set_accumulator.verify(
                    Hash::hash(&monitored_utxo.utxo),
                    &monitored_utxo
                        .get_membership_proof_for_block(block_19.hash)
                        .unwrap()
                ),
                "All membership proofs must be valid after block 19"
            )
        }

        // Fork back to the B-chain with `block_3b` which contains two outputs for `own_wallet`,
        // one coinbase UTXO and one other UTXO
        let (mut block_3_b, cb_utxo, cb_sender_randomness) =
            make_mock_block(&block_2_b, None, own_address);
        assert!(
            block_3_b.is_valid(&block_2_b),
            "Block must be valid before merging txs"
        );

        let receiver_data_six = UtxoReceiverData {
            pubscript: PubScript::default(),
            pubscript_input: vec![],
            receiver_privacy_digest: own_address.privacy_digest,
            utxo: Utxo {
                coins: Into::<Amount>::into(6).to_native_coins(),
                lock_script_hash: own_address.lock_script().hash(),
            },
            sender_randomness: random(),
        };
        let tx_from_preminer = premine_receiver_global_state
            .create_transaction(vec![receiver_data_six.clone()], Into::<Amount>::into(4))
            .await
            .unwrap();
        block_3_b.accumulate_transaction(tx_from_preminer);
        assert!(
            block_3_b.is_valid(&block_2_b),
            "Block must be valid after accumulating txs"
        );
        own_wallet_state
            .expected_utxos
            .write()
            .unwrap()
            .add_expected_utxo(
                cb_utxo,
                cb_sender_randomness,
                own_spending_key.privacy_preimage,
                UtxoNotifier::OwnMiner,
            )
            .unwrap();
        own_wallet_state
            .expected_utxos
            .write()
            .unwrap()
            .add_expected_utxo(
                receiver_data_six.utxo,
                receiver_data_six.sender_randomness,
                own_spending_key.privacy_preimage,
                UtxoNotifier::Cli,
            )
            .unwrap();
        own_wallet_state.update_wallet_state_with_new_block(
            &block_3_b,
            &mut own_wallet_state.wallet_db.lock().await,
        )?;

        let monitored_utxos_3b: Vec<_> = get_monitored_utxos(&own_wallet_state)
            .await
            .into_iter()
            .filter(|x| x.is_synced_to(block_3_b.hash))
            .collect();
        assert_eq!(
            4,
            monitored_utxos_3b.len(),
            "List of monitored and unspent UTXOs have length 4 after receiving two"
        );
        assert_eq!(
            0,
            monitored_utxos_3b
                .iter()
                .filter(|x| x.spent_in_block.is_some())
                .count(),
            "Zero monitored UTXO must be marked as spent"
        );

        // Verify that all unspent monitored UTXOs have valid membership proofs
        for monitored_utxo in monitored_utxos_3b {
            assert!(
                monitored_utxo.spent_in_block.is_some()
                    || block_3_b.body.next_mutator_set_accumulator.verify(
                        Hash::hash(&monitored_utxo.utxo),
                        &monitored_utxo
                            .get_membership_proof_for_block(block_3_b.hash)
                            .unwrap()
                    ),
                "All membership proofs of unspent UTXOs must be valid after block 3b"
            )
        }

        // Then fork back to A-chain
        let (block_20, _, _) =
            make_mock_block(&block_19, None, premine_wallet_spending_key.to_address());
        own_wallet_state.update_wallet_state_with_new_block(
            &block_20,
            &mut own_wallet_state.wallet_db.lock().await,
        )?;

        // Verify that we have two membership proofs of `forked_utxo`: one matching block20 and one matching block_3b
        let monitored_utxos_20: Vec<_> = get_monitored_utxos(&own_wallet_state)
            .await
            .into_iter()
            .filter(|x| x.is_synced_to(block_20.hash))
            .collect();
        assert_eq!(
                19,
                monitored_utxos_20.len(),
                "List of monitored UTXOs must be two higher than after block 19 after returning to bad fork"
            );
        for monitored_utxo in monitored_utxos_20.iter() {
            assert!(
                monitored_utxo.spent_in_block.is_some()
                    || block_20.body.next_mutator_set_accumulator.verify(
                        Hash::hash(&monitored_utxo.utxo),
                        &monitored_utxo
                            .get_membership_proof_for_block(block_20.hash)
                            .unwrap()
                    ),
                "All membership proofs of unspent UTXOs must be valid after block 20"
            )
        }

        Ok(())
    }

    #[tokio::test]
    async fn basic_wallet_secret_functionality_test() {
        let random_wallet_secret = WalletSecret::new_random();
        let spending_key = random_wallet_secret.nth_generation_spending_key(0);
        let _address = spending_key.to_address();
        let _sender_randomness = random_wallet_secret
            .generate_sender_randomness(BFieldElement::new(10).into(), random());
    }

    #[test]
    fn master_seed_is_not_sender_randomness() {
        let secret = thread_rng().gen::<XFieldElement>();
        let secret_as_digest = Digest::new(
            [
                secret.coefficients.to_vec(),
                vec![BFieldElement::new(0); DIGEST_LENGTH - EXTENSION_DEGREE],
            ]
            .concat()
            .try_into()
            .unwrap(),
        );
        let wallet = WalletSecret::new(SecretKeyMaterial(secret));
        assert_ne!(
            wallet.generate_sender_randomness(BlockHeight::genesis(), random()),
            secret_as_digest
        );
    }

    #[test]
    fn get_devnet_wallet_info() {
        // Helper function/test to print the public key associated with the authority signatures
        let devnet_wallet = WalletSecret::devnet_wallet();
        let spending_key = devnet_wallet.nth_generation_spending_key(0);
        let address = spending_key.to_address();
        println!(
            "_authority_wallet address: {}",
            address.to_bech32m(Network::Alpha).unwrap()
        );
        println!("_authority_wallet spending_lock: {}", address.spending_lock);
    }

    #[test]
    fn phrase_conversion_works() {
        let wallet_secret = WalletSecret::new_random();
        let phrase = wallet_secret.to_phrase();
        let wallet_again = WalletSecret::from_phrase(&phrase).unwrap();
        let phrase_again = wallet_again.to_phrase();

        assert_eq!(wallet_secret, wallet_again);
        assert_eq!(phrase, phrase_again);
    }

    #[test]
    fn bad_phrase_conversion_fails() {
        let wallet_secret = WalletSecret::new_random();
        let mut phrase = wallet_secret.to_phrase();
        phrase.push("blank".to_string());
        assert!(WalletSecret::from_phrase(&phrase).is_err());
        assert!(WalletSecret::from_phrase(&phrase[0..phrase.len() - 2]).is_err());
        phrase[0] = "bbb".to_string();
        assert!(WalletSecret::from_phrase(&phrase[0..phrase.len() - 1]).is_err());
    }

    mod generation_key_derivation {
        use super::*;

        // This test derives a set of generation keys and compares the derived
        // set against a "known-good" hard-coded set that were generated from
        // the alphanet branch.
        //
        // The test will fail if the key format or derivation method ever changes.
        #[test]
        fn verify_derived_generation_keys() {
            let devnet_wallet = WalletSecret::devnet_wallet();
            let indexes = worker::known_key_indexes();
            let known_keys = worker::known_keys();

            // verify indexes match
            assert_eq!(
                indexes.to_vec(),
                known_keys.iter().map(|(i, _)| *i).collect_vec()
            );

            for (index, key) in known_keys {
                assert_eq!(devnet_wallet.nth_generation_spending_key_worker(index), key);
            }
        }

        // This test derives a set of generation addresses and compares the derived
        // set against a "known-good" hard-coded set that were generated from
        // the alphanet branch.
        //
        // Both sets use the bech32m encoding for Network::Alpha.
        //
        // The test will fail if the address format or derivation method ever changes.
        #[test]
        fn verify_derived_generation_addrs() {
            let network = Network::Alpha;
            let devnet_wallet = WalletSecret::devnet_wallet();
            let indexes = worker::known_key_indexes();
            let known_addrs = worker::known_addrs();

            // verify indexes match
            assert_eq!(
                indexes.to_vec(),
                known_addrs.iter().map(|(i, _)| *i).collect_vec()
            );

            for (index, known_addr) in known_addrs {
                let derived_addr = devnet_wallet.nth_generation_spending_key_worker(index).to_address().to_bech32m(network).unwrap();

                assert_eq!(derived_addr, known_addr);
            }
        }

        // this is not really a test.  It just prints out json-serialized
        // spending keys.  The resulting serialized string is embedded in
        // json_serialized_known_keys.
        //
        // The test verify_derived_generation_keys() derives keys and compares
        // against the hard-coded keys.  Thus the test can detect if
        // key format or derivation ever changes.
        //
        // This fn is left here to:
        //  1. document how the hard-coded keys were generated
        //  2. in case we ever need to generate them again.
        #[test]
        fn print_json_serialized_generation_spending_keys() {
            let devnet_wallet = WalletSecret::devnet_wallet();
            let indexes = worker::known_key_indexes();

            let addrs = indexes
                .into_iter()
                .map(|i| (i, devnet_wallet.nth_generation_spending_key_worker(i)))
                .collect_vec();

            println!("{}", serde_json::to_string(&addrs).unwrap());
        }

        // this is not really a test.  It just prints out json-serialized
        // string containing pairs of (derivation_index, address) where
        // the address is bech32m-encoded for Network::Alpha.
        //
        // The resulting serialized string is embedded in
        // fn json_serialized_known_addrs().
        //
        // The test verify_derived_generation_addrs() derives addresses and compares
        // against the hard-coded addresses.  Thus the test can detect if
        // key format or encoding or derivation ever changes.
        //
        // This fn is left here to:
        //  1. document how the hard-coded addrs were generated
        //  2. in case we ever need to generate them again.
        #[test]
        fn print_json_serialized_generation_receiving_addrs() {
            let network = Network::Alpha;
            let devnet_wallet = WalletSecret::devnet_wallet();
            let indexes = worker::known_key_indexes();

            let addrs = indexes
                .into_iter()
                .map(|i| (i, devnet_wallet.nth_generation_spending_key_worker(i).to_address().to_bech32m(network).unwrap()))
                .collect_vec();

            println!("{}", serde_json::to_string(&addrs).unwrap());
        }

        mod worker {
            use super::*;

            // provides the set of indexes to derive keys at
            pub fn known_key_indexes() -> [u16; 13] {
                [
                    0,
                    1,
                    2,
                    3,
                    8,
                    16,
                    256,
                    512,
                    1024,
                    2048,
                    4096,
                    u16::MAX / 2,
                    u16::MAX,
                ]
            }

            // returns a vec of hard-coded bech32m addrs that were generated from alphanet branch,
            // note: Network::Alpha
            pub fn known_addrs() -> Vec<(u16, String)> {
                serde_json::from_str(json_serialized_known_addrs()).unwrap()
            }

            // returns a json-serialized string of generation bech32m-encoded addrs generated from alphanet branch.
            // note: Network::Alpha
            pub fn json_serialized_known_addrs() -> &'static str {
                r#"
[[0,"nolgam16n494axmrtxsxhftn2sgvn4uggt6tag8skztcfc8a2yrrn5l69n8gk5f8eenhf0pelr0rt5wxk82z46juq9ndpzx4377hv2ngns06x5hcchvtmr8wxtpvvykujq6tszt2w4mhdwmssknyfpjx59f6ywyz9crc2s4md0dksv0ayklk5rx3duz7p2gjtmtlc3sgz07urxljtf77a9yrwn6qy300f2e36z6humz3rehvphpaddj7vd02ucuktw9ux476njx9hn3sv92ay6up50ef4n3zh2a6jvdfcgsp6ed5k2q8e4lfpmc3p9uyrej7scarvkefe6e2muup8tyn5a4fvfsy48e3rcpxncfkz9wk0j5wss98rv3zq50uddhafh77z93ulleysdm839emh8v69y053v4hvpffr3s5x3lxmshkrq0087lyppqaj5xn52fhu6tkjxg2cvlxuarh85c58vnyatuaytcnshux4tk2qwgmu8kk0s6u34xv643aq59yrsnsvu5wskzvkwzjjrhlzjnhs5z5xg8fyvanv5806cpzjea277950vpt2npf9qr7qd96rttst05fdcxk5j0ut4c7qfpa03tq5rrk6n94m0fzu2km9hc4lez47e0v9yeu0xt6jkpjsvwcrwu0gxnuq4j2qjmcnkvafsuteyayhqvtwm3ypqyknr6zx7zcusp9h29970073wzwywp8gx0t3yh4usw23gvlhauctsfye8g3n4tvfg2xuhgrfr86q5z9u9kkec39krmyvzpewah3c36em0zskkns49jl9q4zynj8rymgaethxsrmjmakj0epe42xctu744ktwp9ms7xh2gumzexfhmgkqpypusx5sv7ag03pstlhqpp3s8vptaeqt4p57ejjlxl3v2zqdvsnvvcjv5twrcgv8s5a5hp27v8vqltxmfz9hd5m2l7yf9ux9d6rehz26gz0fuymnmcpru3500y689wq6k45xacwzhvlxwh0gk84090yxmeng8vwjvs35xkhnew7dn5zxky8g63kyfwhnz2m6vjwl655gwv9wgxynjsyga2h98hqz63eej8lpn6g3u4tar4j4d9ul2es2s28sdg9p0d4mawt7gryc69mvzadteyzxadg5h44vcm94gmynn9860fqxp2wgcwrm9df5vy32ylekjdvgmlxq63x2r3su50d25r3w90cl34pmmexhdeqj7g3kqphagvf5yff92am8f83s49c6y5mc972frs4qmmjew4f260e0tpsruqyxcyqm07kn6jmc7ptkfk0ky3hd3cv8xrjarrru69zqyxstn8d4lzjf2qsuvlktmqu037dfvmws8ejksrp66dqvezyndfz4sr6rcqsaexl8dk3kfyks6g8hdnspehnqcqw4pj8675ql4zwwsq74l0q8nu8a2ww38nz9psdf0t9cszvxk4y755pgfverlg4ew2u742sfv9f0ftwvcmu96p54gfd52hcsg6y0nlzmrdm2yqd08dvcdug9nyl27mchu2puhkmxdwylyuf26fdmya3tm6886sjx0v7ms50f0vsuyhsrewemkgykskldlfaqgfncaz0y60y0yw50t9mlshfezyp2sksn7zgpleuhqmqj5g5dh4ytqnt676ufr3g9aquq6dx8qdkqs5t5ptektsqlrpda9l4slycsy7hz5gyn4dzv45008e0fplhwwxwuarsjace2cr8qnzc55e7uwgs32juxfefflsv5942z5y33p2cgjm983npmw02v82jn9ktrmvadhvcraaz9avp3hpx6vd07pcwk76wml26zr2ew6e4uyjv0455uadvrldq4hev7fh3menu7hk9mvgl7yaez8afn5ysa95uvf4gwg4metjx78js4ssdqj4z5rk20ue0tl9d5k3x9cuefjyxzc6uu9mduke8k4wuz8hfj5wqpv35dzhj3je7g7phrcahrd9u24n04r2g5akupq05trs3h2r924rh53we5p6a3cresh73e9jy5ptr34a3fnhxlhg8gwn0uz5ra27lw5j392zhmjype9qtwmgrhm6y7whqqwkukmwthq40t2hd4j8ld2mckv3fyy97wcyf8dzjqnnqmcwvw0l4uwtl6e9z77w3mrenasvtdre320jhzq4phskk3q5r27avt6fa3k0j0evd8sanpgq6wtk0gssa3tlhstev2fuwpcf87h20v3apfvuglqj6kf4ra85x7zwks5g5tklfkxwswjlgheccypj4832wfu76gggxwvm9vzy5sxttns983z7qqul48ndp0u9268gj2l4qvxzv0r2xvzwc3nanc6sazwwgjc5fy6fg5vezsq93yft3znpwjpm4hqzu4mh2he84ru88gq43hrk4xld6et72m23cts8mlv0c9wc0shjg9jt7u5jn9le095jm32nt7dm257j6de0ym06ah3rpljfwf23gyf2ms8g2dj8hvyc59u7aquj35ajqhvjw55vhn8r2gdap607puwzvwlrvts0mrtsnkqnjq5u84pc39pf9x4pxhv0aghxwmj5wkqx9ynlcwh99wggeh2wk20f79anchlhe645lfzed3u82cf76yza8vhaz2lh42umza4hwfpz20mvjw6thm0vnj3etxvdeu73mjemv4mw27dhwku0726446tklc7memzet7ppdj9x3jvrmsstspt23zjnx8dl9w8akw88hyhlhtglal2gejqf9ktnuhng9p6paasth3sc5x7yjhtpvxr2ma435lfr4jceu6pn0na7n5h37qwuahtac9cpxuley7dvy0sslkq4n0nz0dwj9660pwkymmdj5e8mjnk5r2d2v8qgp4ymz83306teu3ge5rgjlmx0wnz5vpx6rtmgfhk2rphntwgmxd4c4m5cxt4q4y2lz03j0chqrrvjqycwe5kyr0tnhg954wntsx2fzgjx9pma7hq9qtfzweawps7j3mrzxkeg8eh9ve247tnyu4lqmx2f5wx2ql4slfkx2vd4a53n95ymt4frmr89jp8fx59c9jqevwzqzl5fz08clkn5wzawac8qhywryldnwvsjt8tz6s7w5qa85sqzyj23ep58lw2rl0ev8hez534cj2qyw34f8mue8g38s75pa2nju478qq6ylatguat4dam0r8vpepmslm3da3t8nrwm7gjwe968u0ps5nksp3d9svfptudr9sacqvxjcspru2vwzk099uq0pt38clvr3ezmuyq5cjg0ajvn70x8s7qfla3j5w8nwgrrssqfskcxk62zd9k6ssv26vm3p2g5n3lnvhd3dpv87l9hv5w0mvm4hl705t7cfm9tuc7ayxz5jux3u33vvlc84p5ewe5ruzcl7h2gy8u0ehqd38jz5335tme458ndm983gmhfqmcg38uukkzv8p9wz40a6e6"],[1,"nolgam1ctmven8wlv2qsz8dag50t09mk8wzvtyal7knz0ds7tvufgxwdlkt0c24c7ykxddl87rzw7q0jk8h2rmechagfyjylqy6tyd6vlrtepr5z02609n4qmtuu0whx6qpqydj6rtnkdj7rkp5a98ssvat6f8x5773kespk4un5z6lna9p0uwksuyux9uhkl2rqx2p3vvq7wurmuknj93d8zzlnx9zpasrn5mp0ayhzrlhewx99g5u98wpj34ue3260jcmr56enkd4xkwvkelkgaxkcswrcs924l6ssjxfvsudlz4ymq2hvqvr28jgrk8kze0ymuzrs6w30ce3wdxter00nclt2tjkl4tljy0w9zruclrv2ayx0uc80h6k3ut5866ljg3uz6dug9nrq9yk7av9ykxevmwljsyhyf6wmrp953k2ykte5q5qnrupcu58sa44rfn2cq5fx64fnzauhzr99qlzcfvyg5m6xhdrmxjtwlgwcflkux0f655k4srvm7fg8lrcm2ffgnklapdqmq8nj4mrx48zydz7m4lh04k8dhylentafzhl5kgny5gh4rwunvc5thv8k2h9nryl40tcawn7dvhhxl0ah4y4gfgd45f9tjyw56hmkmkmqqpxajxmny3dq9zrrapvrkcvkdyhtmpyzeyuxvafsk0jj8sa5k0z02tygqaupeeyjkt056hzzetvgw959yfj69e70vuu33fda35dyjjms2s30jrfhdmjlkwsmy79atyjwq5caqc3yaz5q97stavw0a55an8zutfgl4m3ytjw60f2xhunpr5kcvmem0rvr6wray38z3w82qyn6mnqft63qvsvcv7k55zcmuu6peje7l7srn0m892flee0degce7sgxk4qt665scmnjrjtj5536ntw67ytgg3rsdldsxhfrys964cuzrvtl8nccpd9jkjw50t8p5nahfp7yfp5alpwpqtc9h2vqm9m75kd8mvh5enfk48v45j4k25h4svmytqt83l6prprhwx0ck00054nfzar4nkry5khjn4sungre96p0kmuughd9apk6mu4t42l5x9z22ujmdad82vm6tt2zlwzgxrcn8m2yfqxvxrxtl2zxgauylfrrad8fc25n7ak4jgxds8708jtxcxjdsqv8ac6j7agvy2ceyw7e8u2dwm2wnacgdat5ka67u42xz57jefc84jfxr7cdpq5g45zlmq4jnzwpqqzd3eh3p463d8u3lhyn9hk7zerv8zclaag6xmaudsuzfhu8k96psd02u8nkk02nzefydfrmved23hw0cr4rcemwp7vqc0yzqw8fl69mpgfjgv9ngg2zue3e9spalmah85j8tlapv3s32r384c4a7gzqpz85l4p6akh5fsa45k62luma2dflfw4k46h2jtt255rpadkk2flefz984a0p4209utlfr0cxk9m45pg50fea4v55g93rwyf4sgfh7l0hs33vtkh69aleqwv4w48ky0t8eg6lnj5ct7nw77gxc4rg8cl5xyl73xl9kzqmhhsla8f989tq0u2m24lf5899hf56e0qnttvk4755y848rz84mmklwg2tc5mr03wmuhph5yeapxkmdw27u6hf2r26kstp6udu6p0lqg6leveaujsx0wj5tp3yq2cay0j6anshxkfexp7dx8xuxrme0n5mqqs5prlxe4skfgpkwv29neqwdw5l4c4w52k9pgrwatg8ed7whgsjrwtwmgc0255vpr9uhshldldd2j4lkchkwfv6k0r0e7rkv9r4gv06m2h7frufte87lg5752zy6c4wjx7w488lyx3v43n208ulseywaqfaq7388qaal893wz9gh7pzrxglq86hhur46jxmpf3fe0wtjdayhtmz33md8ur7q3x4h82vlgdqsrecvlktdh2vge4vl0rwf6mq8qw2ph5j739sknucyupt6fxdsj3s8cfar9xay0uky00ahmxc38778w4p0ajzncl3wmg8mde95m547tjjk2p2q2auyl2vzgeczcpuz020jmy76edmhjnp3r757jw70xvztt666vlkx90m94hqx6r4sved4fkuczphqg8t785xvf77gdhqm08z6g2r9pxaag96cksv0e4q2ssmq4yze8c2jecwsvmna2yu4nguy9ez75pfw2l3vwuh648xg99agdp99gmzmx56jgtyn5zakly72kldzzpuvn64lm49sdkvjcr0eas0h8apk2e9sh2s2zatlyur5yfvwncuzfwm63tvdyjcna6tyc5hdhs70ss94hz03uds4tynzk3898kgcd2m5qa4ewtrwp7zq92c3q7jmfvdrk99gjc95z6t7q695v8lqlu2cyjhmt7fk0wnf6x72gj7fppyk4qtna8y53ejpf3txxmhlj9k3ayauyfvxn40dv5g5pwhqwy2djfpwer9uvj3yqmts2xpmu85f2x3gwftmymau0hxgau5njyvx8hgrrmqw7424f6lzpdnj7qy2qp9dep3h4kqcncm6kd29n2thmlfkv22uta903js53vyaddtvake3y8mp8r72ft9yhp7pmec37my4qccc5yjq2qc8036ysj52cgujrctlaqgt8rj8qjt6uy9hlupnm605ftlpv2kjl6sqd6cudwxm7cv94cyhjwp6tnrup7vk0yl6k75eq36mtvaw8xwln9j3m5prvmcw2r6v0jt4x2vyrfy4p2smflyzujdevwt3perzg6z0lakk52k79n0fdzg4vgdnd8k75hg6qwyvqqj7k7sphuc0tgqhwd0q98ql647mqp43kk9qmfrx3mpuwmvtfmc3ragm2an8u05c6mrvpjfknd889v5d0um2h0dgyke7n060sna53d5fxyf9uyal88dujhr76na7jnhux0cxqq55nlpltm2mu57xw6dmud2vns87w5gqykwkqu54js94p9rudf95h4zhkves6u7ytrcag72j88jvwm09p7t2ml9u66utjgjljjpgzcpc6lfa9pwvlthslprecwhkgc3jn46ypp2zyav69wvv56jvvarf5d7n5e8ern4n5sj462d58y73x38ugs3y9m5d0kfqhekc0t495tdd94g4pz8tjst37z2lk9ahgeulgs3nfxap3vhy35fhg22rduqsfd2dqq0p26p6malul3zmkmq42xxpq2kn24gaysy4gpscpnz07x45d8dajqensfrgjn8rwg050xqcxz59hsxsrmvpwxwefy57pw52rk4lkh4yz5f2n4ckj299pn3wgz909rxjrkw9e74dxh5x5372te95hpwh7zxer2tawsxdet9ehc70qshydkvdj9qnfguldte5t3d2nf7n8v6625d6unt4xe94y3g7rs8c7pfjv520ukwm98c3lqash2wdt62wp5xdk2m9hzj9gxh2y9mfw2yl2r26s"],[2,"nolgam16puwz3zvruau6jj6m28c35g067qzdrcw7ks0zmlgls8ktrz86qv2ya9g96nh9eqea9s92vvcu2jnjz36cmfrrfcc85vxvg5r780xulyte4tpt9vz7kmf2xhjn7tz6u56hc8rn7sr9mdvf6cpxnmsh4k4h4xfw6szvwjmaylm35gv7n2xp5axjylwe27vrz8n5uf0elrn3ag3d5p7p4lsf4krves5g7zy7w5fuendpty0cj56p82hk9l8gkndmq7l24yuphkvtnyxfh2cr2p0mvcjyjknjh4gx83c83xhwlu5udrmfny7epcf3ff6uaqfvuslxcn8p9kg56y6tv2hn4mshnplhuwyfzme2dm8p540up5vmjv4pagpze8wptvgj7vejdpwzq2y5cdsfrjn9kxjvhka2mdruc0mg9xllqaskc0wd89lr043avyqutca2hkyvxzqyue7k3yp5m379ys22nu46cdwull8k9h329c95mnjp4386l8sdsvj7kxsqkmtt6gzq0wqdj2jkswlhyjvjdy0mq09ef0wfnvuhcutvhv2jj4f9jdg5usuhc0ygnwt9jrxhtn3u3eyzra2rwlmxycsu6gwjceu3h6hs9sgflxd3undz0meyf67pswhjetdzxkjl6g0t7eahpuymr7x5uyt42a96swwkdrml2wqvwtmcfqadx8gm5nmxx9w2qnnt2k55t94q5vxnae6nhtks2ge6dvqyh0ddrs0dkjty7fu8cfhpxk0kprlfhpgyaxtv3f4k4a7nmlvqqrvvg6m4j6xcfep25zfg90lfp4dnlw9ft58suwl3utvtk8ktur53tdrc295w4adz92vvhrcqwfutc3wc75qu2dzq3utvpk0tskk8hharwt4c0l7jnvrytt5vfutc638339vzpcxt9jld6ptgvn6jjdt7hh9nxpn4pc39wy528hk77jdzen2xkyjz2lz7rm297rfk4q2qcqnr3y2gcwcrp6320gv3tu5tc2quxcrzpyv9x2mr9cklvvh7vv4axyaqpk2szsxw8r6fes2fdl9tqf2h9ns8ruae8207nkx9zz5zv65s5f29h5d5grjxlpt8rr8snxauyzql7wwa0sfwz2a7z047a0v8er9przhgwhpnhrwcznu0q8c6m402napg7hd3ymnqk4dch72lumaqptpfmafugxgl29q2a5s69nv8qygvan42ffd4z46z5249gjgsqs62r9vczhm2fz7ey7yrh7c43fqd4kfl3gt57lv0m9rajvxctadny3er4n3e4jh0rwh4kwk2g9r26hkvwm09cm7zhqwg9ddma84fugkgval3qxmzzfyem0cudx3nmm9m8d8nvfadfhycv5hmhnxcpnae3jrma5sl7nh4dmkqw36a8pmw359xfj5vf5w9y2us76q7yt5hxxt8vcdsku8pvctxx9mcmnra0xfyr3g943tv7xd7yd4um0d8z98munchz4yjersl0czrjclkegsz8u25kysq4qq3fwlyhfu44rmp08n2wc7sw4tlgu9waqptp73xlm4gpmk6wszu3ttdc6vfs67n2vpvjvd5ck404sanzv7fl4qrchegmqcdjn0n6e7lsyerfd9duf02ulhnuvr2w62p35402x7xdgqelnyz45eh3t2mzz4ndjvyplwvhqf0p044s2kj44svvrnwyx70capd5762deg3tc3wfjn7vtt5fkehurys27jujlc97zxg3xwz8uf5lqpt0frmcf3pau05sfar25sqfeyql9wscrv60ja0sxcw6efu4cat9zvx7zlrevlm0pwl89t9q7qn0uenat09tzv75k6rrlwt5nrdjmhfk6ysucmudpzsnj7xg246q5kmpz25p40ms8jykxzx60ju7xuf274sgq97xs0wuexyhaujypx7a6ptyatu96cte8um5yyzhumljrjmdut6p92ljxg0e4qd65s7w22kdwpdsetp9jpvkap2ky8hfq2vdp0smujszsecn27pdkzuj9zx6e8j5wzqnfz8gjllw9kg47j34uam0ukzekc7l0nhjqm20xl6m8c48p42wye5mlyplkz4tk5q4wwn3f230w9whqgj3v3nfc0kfdujhf6nacmchetwp0aze8yjg6hmc2mzka7uh8fsl79t8498y3t6f0866gtv6qwmdrhsmzg3y63022m08eyw0rq3jpp5hrsamftsenztvr0mp9942qhf9k34x82u6l9mke0w6pye6sxj0sduw6n9xw2vtnrs6rhzs2v5dp250snjgt0nnmawqw5zskrephfh6xw0q8euv0pjvlujakjsftddeh8rsckh04qj4n6c3ctet7dfla0nrjscaqfwpmaya36ejsd07ycfqemh658qa97ah54lj4yelredtgl3sw7k5p9src675y8qzf8l6den7c2rz7u0chjvty3luyjgdjjzyutv8efhgj80lxqat93nmf0ggeg9xxmawsutznasjtxy0tc6e4stfq09jr3ne04gm2dqvlawxk34583r6chymj6yq2huks2mh7m3s6agy5l6gq5v7xc572l6wg9feuudpkdsd576qr39axn592ufgyqa8peem76p7d3jyc34r7r009kmlx5rl3euwkewgs69zr7z85rprgzmgxmf4pcv0wervksujdgzv5jp0vzgmye283xc99dktzl60cql337v9r3qapcgjryqgdltuw3j53g64s3qyjmsum0f8nnayrv3dts3lx2a87ufmdwnsc02jh6zrygqh560gjc0z3yhn8wfjszrht8nja3r5aq4nnlsndpue4x0pxxdkrahgc8sn6arrjpv9gnhtvx95duapk6rv95sjujax2rsqmuhejlucxfcjppgla6ru044swp6e9z2vqak2wn8v7xu8g5acmed0ed4tp02fd0ra3qyppd4yucms7jlvgsj8uc9ma7g72qm7uwy7kguv7h4cjtuqs4agut3cyuajyuhqls3vqtxwecmcnswcj8uac33djeqfemczzc0mlhwve4kq7t0tggyl80huj4295yjjcp58vng3drtr0ulm359vn3d498c7jywltss3kx53cyvu45r4d82rdn8eanyd5c0vz2d0rw3k6r5w85q3ls27lhcrp0mqntgp3uhwt8qkd9tgq8r59ugsmkzvwgse8wpf5hsk3u5c6rtkqzn0rkgrjuhfhtyeh9ynkv0wnf2y7yanpdvsjcuskw5nvfgzclhcjmr5ezwxnenm3anmq3py0hp96ulu803e2pu5snrc9peshxrperj7rhfg57vtadxdhd6tl6q4nsnwe2uscqz8lypdvp9mexwdwn7ztppcvxlv4pfpktsceh9d72w50lsu0j6ryhcuum5fwsp85qvev9gg9rk4af2s29jn882mfvf3nhxshyk2k9kml2kwdus9j"],[3,"nolgam1tzxe9cmey3xlr9ddlcg2cadq70nzm9cmzgcjhrjc0p5raquecylxe6aw8wev5vhf3c3ut6rxal32r6ngn04aln3crylkfpzsx823xfz79jcekn2z0myryaar9dp64zzj2tsy4k8zdfuhptwgrfaj9t67za9dc96clgy3q8t9z9q6u7zdenx2dp2aqcqp3ngf5uswa20u8mcz2098negrjksu3hud66n4220v6ss7xg7hg9ld6s9xzg2zlmm30rnwt900et9ygfy6xeaqly29aegku7yyfx0gg809lnc3tvu4qq3apjsntrr2huxrc9lc7pyt3gagdpwds2u6dmrm2yhmj2tkrhx8upfr2g06e7879ckzdh0e7sv7q2h2w9kcdnyzx0j0aa03s6rnjlnpuan9vjae9qugqj53gqenc6upyh604y6asgds4722uyjyqxxfn2pn7awtptd8ptky77qpgc7frxvnwavkscgh3tkxtdq0y46wff954m0rcfwxjlhzkuh9868nnxq3xenx4ywkkwuznxqup4ur844zh6q2vmntc6620vvu7crad6lnsgkfxeljez9sx2tdc7hrykdxetckfv6j95v84zkm0adgaqmhp29lkk96mu9w9n8nun5zk2y3a5nk8pd9krx0pf83j4qckvtylrt35yvu3l6u7qtxr3rewl34cexdr27znqyaqd3y46uhlmhnclk3ahegqctxz3nzmm2dk4427mj8c05ugvwqtvhrmjq8vx5lvd6qwpt4gf0gae7mj9u84u3650xxuuq4hzdtdrnarppnkckhlw3ka0x7j0lj4yx4upny57gwmrs42vej406wgz8ndw4geqtla888texscwvrzp03d6xhfu0yu0w03eym2qdzv6x92r27uzz3833k7ugz2zx7dfs00fhnnmx2q0cvrgpmphrlvq49n80rsauy57s4pn9pr0vh2ht9y92zx0xqe3xlncxjqcnlepatyhnu08wxq3j9h5qyazdge530u3rt8ft3c25u4anexflfst03efhklrkhap706mx8sfakrxjweua639fv636j0h2n075ud6p3ulxvzs042g9lwxqhl544us9lk6r42fpt7dax8fdr3c2w9lnx7uhxw5cw25gj32d8hah305283txpm52g59afcymdn5dw5ctv69krmkj42sff52wa2jgaqlrl6nszd5jurzhhxp840jngcnaq7nq9jgc0n8uejcmmkpe43frmtt4cn2gaj9sa5k29k2mqj05vn96s3f6eqyhdl7rlwu3vpxx8x8edg989nl9j99gcuppu7gpkuecw2jahcpx9cz08jm0xqm48uqadkkrkjmlfddwnk8xh3hc9wt3evhzjhpjkzu2yc8k4grxuwegt0eljrjflx9wlhdsa6geufzmjnf0rzyln5flxz2v6qhp53ccj42wdj4uqh7y44s5pacw4ew2pxvfugq54rjjt6r8rl7f6jm9zgkw8sgamc29awhc4p6f4leu8509ymyc4hz8yady5pd7jlfnjfyp7336hrsae7zevfjqwj0qc7zne5eu3l7vvrgyq7gu9y05nmyry0ns3phtttlx4dlf3j3nmhcfzs3aux4ct6dax93her7pte36nvcswgn0kajrglgk8fkyrttdgw83w66kkx3gcsaq2mj84huaghrrklnt30c7yh0trgmshh0v7gk0esxcgx4xscl6da0cvsw6lxcdh2wuv5ty84xkpfrcuhse9gxp28tg0fhqq8tk5y4udqv60vv44gyln69q7kv025yc2rxjpmmfsx8e69cpzk3p9rfvuz9ujawyla3a7appgm93v3xq7v9qlrn00rn2renuv7zwyhkdsg9yhh9a3sz4gn7fhrv08lgnyeyq47jmqr3syff58az0m79428us0knxd7y6rtxv4t5lducvhmgxqraxy0hrk52h50cxfpv8dgdaq260kgur24k8d5nefuvf3r3jg50z86lp8mhhh4qg6lyt09dpfysjzmacvzpka46tcljgnt3wyrtlyj3qf4nu9n4xza77laxm9v4alc9sfs78pqr8aw0gf9v7lfdvgxspqyja4p67qvzz0zj90vpyq4qzvzafmkr4hvdlcvmdkc59k3lkdmfyuhslk0uq364wygf76nhuhnxutlpn6sfu62nrqnfsfu4amdmn7quz9amwemuefcgvwn6crqqmsg5ph6dqv5a5cn66n80glp55zugawps68we0fjt9phjc8zp2pa6u7wp78uugmwklnm9wgy6clqvqdkpq6vasnjh4qqv8ksulyl8a5ns8r7uu7qlagh0fslmm84malkdqsq0yza0sd7ds6a94uxmrdj2zr8u8m9zwrh5f6ct3y7636s4pd67anz92g3szaq0huru7ckxu6yfq8da0dvjjxnj99jw7ussf6mf3cp6s2yj86v5gjtnsmq65zzu69x9r5eh2lmmnwmdxy0z2pa0v77w8f462p87apza8djcuw49g4chnmlfg2j6n0zsk5yw30f77y60ujhmg5x40yagh98l2tg208q7xdp67wawvfvwr0tqqcd9pwtpmfqh5fke83ehwcnu706wu4n4qmpegkf3se48zxygwzhyh9mpmucjqa4zcztfwzazhd8uhd7t90szkjpxhgu03xcyul30k42szgzwrf44ydksdrtxu90z5sc498lc0sgjhr7gwquq4w5s2kn8cx3xvfu936x6x92mdnj3vvz23dpzn3dzmqsavt8ymdr5yjd35gvm2tv8mqv6z754pfd2txr7kvrjfsljyfrx3xwef2l9ysq3njcydku5gljdtgmt3rh5qvnv63ll7vtmtqf44p0ft38jwam280g973kht6myfn4rms9vjg92lheqr7x7g2fqw5zev849u787c4ddvuapypv5r7x7455auep99cw2jh82u66k086cujkl5q5yax8rw3fjrzywr20ftth37le43xnmt8xhc07fvz2ufunn2nqyu46xwtypm4hlsnzxxhnxxtsw5ngddx5xcshhz9056fv279razejr29ttetuxyzrhcfuursecm9eumadd7je4c3vycqxzaclnslhf9rzndl8dddddw8xjm4ahyl9w8pvaekqr8p7urfcyzz2pudh2md78pc9p5jl90hz279l5tf9kc0pdzmnqqrx8p9prywvkvkve9rscseh4k6nl7wlxmcr34fn4ahmmwtstt3taqx8qhgdhv74xhap5zcwdycv6gvxlkf6354c88gvxauqpqgsa4dzj2ustqv8smuw69aw6a4rh9t2vf6uy786f3f85u4zkx6ufk6gvs45zcsm5qr92st2v2a4yutzcjckxc45qwtmprk5gjf3ffsqx6c7j96hxfmqnn076nvc25x9suz46np0tjp8mxpq4wqjqw"],[8,"nolgam16ypmlrghfdtfut2gm0j9x8u06dx73438rthdkm370jxru46hv6fp9ary7q0ry7rcgfpn7mf3wuznxhj5k4x3f3uuzc7myh0g2h2mtwqu06nk7f376ek9metdlyqcydaxumjgs29xfx68ld5aa6l54dtxadx8wwr9vd97wkq8g0ystf232d0kznj80hkq3n5xat97x3grn8enzlcw9w0k0qmmkuvnts7282p8sfn49dd45583gehtugjvq8zzjdza2qju67l2c4awml9jfxlhygvp2zfea5cza02dlpkkc0zsxnw2c3t53anmeav0tyeqq63r937lznxnl6vjsjcqpuy09zx84n46qhnewwmrtu742yunavysc0ywyuvvv2un2kkmlnay5vr9f7n78e0etsz99g5fkzzw4yj332m0eusr7khna9npp50n0ercx6n5t4n4kv6t54xy5pkanjuxx8uft0l9p8ttd0yuapx72rlt7cquuppx024utjuxd2t08nyve5cv8k508ghk0dzuynpuy09n4xqtnm6u3jfya7tn3l6q6je0wwhlct0jp3xfhkmnqph5kg2rp5uunuta68qanq65645cc9znxtzyluagggg33h6mcfkszvvnsk8w4pvzx6k0sz9gm46f4lrrxd6wxwgqewdu25uag9dq08wvmfd8uzfphxuqjxgg0c7zdek7ylx94yz7daesfj6x5yqzy9m9sslyt2cczhmkf7pc2rae0k6xrl76rt6tqja04fp5kyurkmertk5u385xrwnhpvj36s063xt4ta68hulv6htnjc6sx7ewvlrkws9a3hp7mcudydf0ktp0tp0umxrg5ayyk8fg5lavpxm0npvf5rjwf68gyu69pt8h4q7vwhtemve28waf5jhuxxtq5jjs73exy8f3ujmhpg3hf588t2ev6yz4apw7j0a3wlw4pgtfpgxfzp5wgqxse2a6dqd8ze7dugxpnyyprrtpw6gzetefq43wp243umn4ruh5fqv2lkpq3j9q6r33aglx8lr9dhnh7eaz8yahy6k8q6sykdlzz986jng9wxz40jxce0yf4jnuhw2ln5s5m5y972gds0w85j5m00nqwdugu33u77xw2snjmntwedldk46adkvmxeeat9ku5xqx3glql3emmj6ku9nsxu42rsv4c58gemvcexr2jarvum76e5crs2ychk5rppmsg0g84a8smvgxm8n65et502uwyjk7632epd04rwcejd5qfs495grgnsf3kmyek44hhen4ffr3sw4th70qjhu3yn5mqwluqc4dlqhdgx0l3yzdu3hk6jctfj3gmxfx2d9ah6gk86a2x7kdjwullq85p5xr2r9jgzwm3hfa8kp4y5j9d845ka85f58d0ymzethxrmf32r2mujzht8pt6ck78n4dj4msdlxynqnjadqjne6rtzw32eksrs38gxy24a553sddvsfy34e6d7kcjtvq9jpazc5z0wpl5sfxn95ua4psh0h0dks6asxjp2sr9vgz8yglnrpfq0fxme2qz8wytl508fmvrqkkw02nt94drzwmu867hxjdhg5kg33sdqvqt6287q7ml7085nqgmqkytuuhd59hz2szaptr099slzg7rys5uz62kgg6c3xa899f9h2f8l7080vjyz4462jmj2sxklz48sm7kj4ylt3800m6pmkuaqprsjqy2sr04lull6u3ng56u9t92qj4texf4u5r2fsd4tgy9ju263ft6y55dexzr8wwpv3g4gassk2avqm2rw0j5uv2alwxhwh04u333s5a98swdkxw2hv6rveuuzyqc4gq4w8sf8aa8euhxqh6xrm803p2au5vjcr5m024a79ma46z4x3qq2gfqhravnxqmdf7phsk9e94a7cpqwe5qp4tpgu322lddz6dp7lk5a9v6qwtzdflmhtj4fgpxdgkg80uws29zernuvl5vqk4auxk0ka9yh0s5pukg5ewly2dhu3hfrypdkhgu3lm9m4l7p5hfpa5hswh40sj4f44r87rk2wg7nduqxx6pu970eyhkxx3tvj94pw5tuq557zku8900n7vrkfu2m9k88ceq3t7cuclny6ckgqud0rgsrhar296qpulfh23gc2eajhyj958a34rg8tnfj6k25q0q6w02927hyvnarc4tmq5zalmt7ut86tldqge0f2lrx6vpacqr9yp5vu887p98nwpwl2ht9t32v2wuc5ylxvfwfuqw4t0uw84f850jgnhgz7uknf7m69pryqmm6jsfd5lk8up5yuc92s3cxlt6cutm0jhe26lvyz6ylcqzt7l9uu6zlhk5nzesmyvm3pncjgxfp6cm2xsch9uwyth9amrp2w838fguj9umy2a95hunvdy9dczkwa99aehug24ez3jq4jxt03w5j08k489mc87h0dq5enlk2f63w8qkws9s40yznsvj9m7z6lc4c2wkmddndryut8hy0xhjnnvjhghegz26pcfpyht8eygjw0nx23q6h5uhw20lglk6g3nln3djmcpqphtatax4yypzp8xgrk2g6l08wyg2pugq4333vmvmmecjfamqxzrx38xgs3p359rgazmt70csc49jl2qxenrefn5ds4kyvrmppzmewyga59kjq5ccr4v0r6l7rh3w082tx96jg7pl8vpdnafdajn9hshstxvu23n607p7x4yp9ak7lmq5x8rcd8js2tafes4lhqcjcptxnfhe6hyuzrsggz52wkvc3tpf3sznk0szsu0w5z6agkcugpre5uz3ad2e349cyf2mp26zp96fgt723danhqs09dfplfpfxa32mdm5rlk3r4e6ge3fffddv7t05tzcpazm5s8gjncughj0hkdnpjzks5fmfgs8znp3gy00c84f9q8cjmzjgtq8u9qpsyys9c6w02jcdqxnr2l8lvrswa0k2wzf9dqrwdzz3xjqw0n2tjn0h0rv7egwh6truf68xvnl3hlzspvnfglkgrs4ctndv64gccj9ta5qgr3fr030yuw90jrq88cn7wmhmykvttsdsrqh3a847ff8mw0tet5qkaepx5sxepe70lhr4j6cdexugaqstzgpmjcj6tfftplrl6s42gjvyyw50cpu69tq089j0x6nj08xkrc9ztsga55wwgal95dewgkc8lf5daqnhrp2z58rmff3f5fu6cerg08auvh593m005wxcm8qcq2fdez6nc00tadrsq99u2ccfzsqswgm6tp4enc2gnl4hc8kqpy85d60gflv2k4vcf68xyj8wcv0x8588k6qmc9gjqkvcn7mwqkwtvhqa722psdrsvcp0c07wxnhf0k2wwdm8s3f698kqt77em0sua8sevgpvfwwu7e92v3k2m5uzud2h0mgp998t0g5fs303cn68yzkhq87nu"],[16,"nolgam19phze7nqq74s3yjzm5ajmr47tmrc8yafex86fzy7f088rq6uafnq06lmeyx8p70na07dx9wm6p3utmlpjq97cyya3guxg0tvxmtwhz5at9hytyw8ysmdkwal9y3grf7phd20heu73ac0fawc8c04d0yzxpm623ekq6t0qea78e37ea6582uvtrenw7jh8jzul6ejhmeqxpf5xr6vpqu3s72qumq3cpax35wzhunjp4ftxe9ahw0yzjyjm74jp5rf693ftfugffw7pt2jh2dqd3ggv3hvq2g0urr0czj7t38pxuml30uds23wqeq3w8weht9rz8ktejetdx4zce3kaefgxzd7tmkyafq0nku62u2yrf69uzccnjcc0ydkxdy2cvrpvwhesg5678ka3dr47flg2w4g9d54v5y74ujtg9uyq9arhhc07meljec76qgsrdzc99pq4thas0cp79g3ww40dsx9xll06lfvr682nn39zyr3fzkvqaumvazz7thhzcjd8m74tgezj7nk80xj0a7cp7vggggjapxhm8dnsahpddvmsr30xrevn5zywz3v0xvhr8rjydg7y2nt8x3jraq8medgyxtru3htu8e4g87vn3jurk67p0w45n74qlau6qxlxekk4wdj47rt6thq790e6tszqpn3v7rdz4rh9gkjhsf3j9uz966ufxx7zamwp8ryzvw794sqrt0sza9sy65vr6ev9nn7apajcyjhzhqev50scwjhm0n5lemau30ac0vwnje9ktn5jn2m209jxlg2kyax8addqgyq00l83yme0rud90d7edps7kj90we40qwpuhny9kp9hn7gtvghcyx8lpg5tqe8yyj87qmte3l0f0rcml23u4tcqtdt4mhumdhcetk60dnf43jjgjvc70j4psl0xhdw4wrrtysk8t9rkxwt7jtyggge3w2dqxed70cms0fnqgf3h58wu25d4dn8xdaeyequ6n2e0xg0ffgu74amdhl98pqpqaxk3kkt7ujyw7a00qxla5lnr3gfxl93ad0tfcl20ut8cyl5s5ghuje57v662nxld0ljmdhyvw4kjrpjr9shzt8m37099ug59u3r7qt74s8vredcll9r5jdsyjqdjhgr799nsf8386ld7ak9x4hxkqwxy9qwu3wpxqtsu9lvv2muczsh8tkd4xuuwajt55vavxpauycl839kvl645kjx8thkf5lmvrtlajwnwqc6tz8sz8nz5n8msm72yf8rqxd2029a6dpflp6qr8gln44xu7jrtkr4pfwk8h064n8qhakczl064rm59a5pty0nsgd7lqxsj8ph0mnmrl6pfpa0dtf64ezulh6pvae4mx38d6n6ks4q3pnslazd4675f00kvzwanvg0qzgqp8jyrelwsyueg0g2enhhffhpdhl829efgua9099vzw680s4xuy4fgyfwq3caac4ys9prtug3rvfevxd8gfuap4pcpv968uzn8h6525ljg4pw7fc97tpyuteracfltx2dd725upzz5gyxuewxcydaeguu5rw3szxwdnul3rt27nvqjwxymy8ksf0esh3d9cmdt2ch4y87fqwdzujz9k9jq0mwk0lq52gtrg4ddgchqfv3h2lshhzc5g99zpjjnqmql8crzgtwherlu2caf6dh303etpn9a2c3k2zkv2gtfhps3lf24087ws9zpqnanhdecnkcw4797dfe3rmaaf639qqzsdy9mf26ar05c7zashuccy35r3n6hanl352x0deca3dsxdd9f38xxc390dw9rfx3gufv7c4ty5ht5756jc2zwmmxe9u48wgnrvzre3u0gz2ngj36pmg8wm8s0j5rk5qxvvndwr34j6plxhpe87gkzq67xu5mtr42cxtqeg33a43un7me8s7zendd62h3mxjuzsffnuhjdgzvpy76rhf5m05dnmrkvu7rj8w68j8kdcnh5q6lqgqrx8gqrw4mrlpjz2vdz6ez3rj4y5e0xdhts4pqed8nyvdcs0pmg3nkys5t6ym80ujaf7pqd33cnw4nz8x5q8w06v7xg68mgdq5r9da8tn4fywyqg4rlg0345u2t7qh58azcv7pa075zdrc2maqgmyc55j45ehjv375wekqrnv8rqdthuxzrnleryyaql3vhrwdup3e680jvffgk0ex7w2n76kranwwhpk9pxkaatkk55wfwn069j8endfksy3ammmar55jkn2nzj5kwua2e79ldlza4a5vhkdsjychg2c7x6qerkalq8juldh9yldagtgwpx6u7s4f2hrppl80r26se8ulymreswtztf4rqd73xqf78wm6f0ky5nwhk5n86jkdyvv6ezsx0rnxer87st8vwgagaz677wt593j0vegpherfdcyl4uvhfw5mdm86645rr435328lzvx8eshrjs0vkr0pxyxp3kt0r59qhxyx3n43zgm7s6lyrf4jc9x96llgr4t4whd09nu7yswslzcls8n9sfm8q44gvl8g2qad8s8g6xdyrtrme45y44x72f0dzjedc42thm0qrwc9wxa6lsm7t9nz47dafhm45430d9n8ve8fq3age5lcgk2uctzyntkqutd0rrhhec7vdzyh3k83pddxt0tass05r02pg0rp2960pj0rl6me2zvxlax5axc49rxl2p3mqvwe4a7ts995mtxypqgsdy6cgguath0cv5jh2mddpje64ggnh0akrffykgd93c7kj5lm32fjspmutzwj64ck57qk94wj8nllnutf7g5mlc9c4guu7k95lwz6zkzq3fq8zmfgejpyrzchd6s6v5f3pw07dy7x6pu6x44mdk6p4ym7e4q08w69xjskjwxkt6p8sr45avjxcuwvwuxtanp7madfnda7c76cgdrqwjrmpzh5vn2yd4qsxw26u4d5mua6h6z0ksy6jw5tllqqpwddyets4m3ea7gsm5vmr3tq4lr2ep8h9vnnn6x69gqukzg0ejxzurguwah6zjp0vqnvn23ga8p6wvwt5ugsgpx5qh0jw7wyheaqr6w9tgxsqhcyfc8js0yah8zpd2d28v7q55uenxxdgug58um3rd9uj43zsn8kanj2hzcgka5nx6ejndqwps0hpghkugng3sxq05243q7zz695e6j4f35jy32yh4skh8v39ls267cp94tcwyx0cmzh95r7r2auzpac98u4ktwrxg4f9a8vpzsjatcsnhnnqf88j7kqaqlxfns260le9qurm8p5z6wvx3rpg4xa73xnhm0v4yjz3x632vere005u5syhkljdre9960dd9humjm2edz8dpc0g0cdt9ljj7u3dtzqyx4t9h0vx4mqshykzkf8m3rr3ynmy9vmply8drucdjhsly3n2xgu4zwr433hpwjvtjcwn4msm4t9amnf7r9vsyx"],[256,"nolgam1qnpl0flsnmtr7gshld0mrvpq2fwfv4vdka8uhqgt32pd5evxhgmfhnvgtfv8xnktjp03yn9z4avtx5pf6dyglkr7h52ssjrrnsq94eful4pxkwj9y7e3mvzcg3aepuwdp7fu8yznp05ua292tq87m3xn97c33pamf89efnqtud0z30uhlxx0q03jvyh7fxr5ydt3lh5l90l59gvj2yp30gcmen6dzsxwtxqxc903yfyn63g2n8shq9hf5py0lftd5vj0txmt65c25pyde474u4y3hh9l3kusv9azl3q5gp5ks2w2czvgnlw7rj7t7mc9vufsqtd6kx6knrp9t9jcleanht0egtwpp97xtjhnnxsqxfm2wrhx36663tuch8avu4c9putq7vqn8se8r9xmu08lwzwe3ktgn4np4vk80ndsfpt8pwnnl37r9kkfw5uwm8zu68ac2n498ra577tuksqc68hk9dkh4dauqkxnk02cxu3g89z3a06qkf88hyku0rtle40r5fzujxrxtrr0c34lfkqpuaxwqe7x09wtj0mrt9hh6velv2tttqlq7wt5hp3txyhyvysw6ntrxcq4vv64jwfs5xz7v92xzwehmyara40knc0sxslamdwq0226355rf32sv5e39eqjs9hzmgf5erk2av68eu4k9lr0kwqlksdscssq8unaddz8smj67s5esy3h9jjm58wraupjzy83c2mrrfuyl3ucadgaehekm7khvrq2q6cfnu9mxdmemltl7ru88wyuymf505e52ccdcmc5efg6nmfxy8scj4esh09vfzydw9lpjvthgxds7v57kmrznqr28guzeh0pxdduq744cnrc58myeay4pfz8kdmw78r3h8l3efqemyhj7awk46qllsw9s6h3pe2vex4jd25nekw0sp5nwv4ahgu5d3qcegtv27j80lcxwerdkjfvnhp5v4vm4qymggt3jarmu2h5w9naw97h8cyyc3lqw6mcfgssm8skszmddstgavkcjnqkkyharqkaygvdmfy873wu6c0m6uqzcf4fqumsq06z7vuw8ped3hhax6c4tqza37mmjjz509nf5x9uj4y07jjex652reppwnt7uupr5lrvl6slc25hhkt2shg5r4awtq78x2w3j8wyq404s9hswpwwwuhgq7n647xw5z7smww2p5rlsq3u00ctu6tsu8sv9axft3eull47cneyen67ng5azqg2ddvrh0957mgfxe8q72tnyrt4nxm4y6qeg2a400v6hwmcdcq53p8xdgt45xqsgy779j4vd2ttvemd4ujnmyqls0puv22rmj0trhmyv4cvg2tglru2m6lhzzsdhjycn47mrr2zdymwhahjfst2eqhp9zmhfuksq4xuem2hgswjghffwr8hdq0ytlel2duc6sucg9fm5y92x22j6htxyjltpc6vszw6nhqfk5jthvqnjcup606cksd5g3h0lrqwsest5dlkel3cz2j3ynar28aknzlzw2awzltpdks5gfuufenwvfyv5k7ywep4dc5rxn03mw08kdwp9gllm6ada3uy4ap522p59am9c5hpt3nwmzhk2t3u0geja7wnptf5ukrukldl08gk9rp4yk2c9aumdnexn694n3cy06vpxh39ru65rjr89wuu4wvnd92akv3de7kdg7g6443cnhrcp9nq9x0g3rfnw6l94kqlcklwkqqvp27v00nm4qjdg82n26zvrhtyw4ms36h2xfahrgmks4eael53c2sh4fj3szewk0kdj3l9nvhtrc86njxt3az5anvfzxnfr88q7cljhfx5jwpd7cgysw0ryvpwsr3c39kus6ded0r8sw95hf6nl0z9r28vcn7rj7w8qgyuj7c4cf4yufkftqjx6hwffxfudmmcjkle88ymf9kpz3hyufses2a4a7fwprn9lqgshu2pn3p4vlczf2rh9rvsw6la72zd2k7a4jrwkxldeqdd3w6ua3yr2v0nj0s2zufsp8a48n2q2kexxvs52w0a90s3fa78wt2tctflfcr4xmprnfqck4uvdu4rfm7f7lewhldy2sjdmdepqzyz4m798sge0wfyclju9xewq6mqmpadusrcukeevya7fse03luqr0cssu2spwfkvmjhepgnxucw6825dxh23jy4plx4v8rsdudztkn9du93j7qyw9clzpuuyjzeekjumknuy8ckx6dxaa205nmydcqgx2wzkwflg4z47zkl7ermacas77jvs75zlgm34wtdystrslgcuw9jrafsua5cye9fp8djg0yunejqep32yjqf4520y38ylfmvhrs0m0v8kjl6tcwuyl070xallnzeu0hscn938462837hmxr50539f48u8v0tmnwjgdu5ahn48zyalpda0wg0dsq9le72ttrla83uc3cdtdz05w4v65ete35ppeurr6523zqgazu269hjrnm3yzusy8y5t2tyjy2wrl9qe22gd9ltjm7s2dt7ymhaw96k8498se3nzcjdf0aws2gf9zrymjtfg66fgzmmrkk3d4h6hmx5wmf2z3xwapz9z2v2kenpc8ms0zlednhwrph3fezwjcy890nul0ejer2hg98gew3jnu5ecjp3zt0ngn4ll63gpevr9h8a82s764hqxgr47rlh4y8y8ne27jgsuev4w95679w4kfzqrmuwrfaqhq3fu02r6n6wp30lz0wmxffwr92eht0tqf9lt77wlwhjz6s86lyjs8e02ghwq2p9t509f443zt0pzmhtyqgymqc8z9cu3glmp449qw26vlvt5rtnfrgjzpf7dyqfqz58zgyyr2n5xa652s4t7w63p6ayxrhmd0hau6vsczd5qf0zlee5g7l7etqs7faesp4nac6wchnzs2a0pgjm4ee6njt2wqyjjgpz83ma8cq23n2r7jcquy7a5n0dh3ljqtz6ls3u8xlny55vz82qevdtntvhgyp3y95np2gga25jkcd7zfhgl3z0q4sghfae42zy63yqdhwp2yjyrknegjtxkkr3z5temuwj689826p267drkzz6dynn8yw5jm8a4ml0058a3p84fpf8q0ljsl8etqf40jqmuwng8709q4sts2c5q4zpk6trtxtck3wlusyzggu6wlazheufuzvdrkcj0rtv3y06l8s3e4fcvzmxcx8w9q5kw8p9m60pqh4jalputnklhmfgpslagwazaeszdajj9juv9xd5df8uf3s8p830wj2smy0u4zs5z52p3tdmus0l4yk7fqj7xnstax56rxrf8qgu3j7dlan3q929sk69m2f0md90sxqf06kv6le5cflg67zu46wxlqq75gycqu3lyt5a5xjlr62kvlcmaz2rlysr4sz58stv4lmkwfjfdau2mnmd2hwrgz7dqfnfmuc97llhyfrgjze"],[512,"nolgam1ulnvnx365d6pdcuqy89rqz9sgkkfk7svxent666eze88az2mfndd3hpj90ndhupvfcycj95uzsrw7gl3mwp2et6gvww0m7ffls9fvak4ssccqzdldyp4jltwqhfl5eaczy8dshqx89044wp08f70688yyy0gxu54ksz4f46sdcxdf6lzu748xueh0xrkmf3qvvrg9nkz3xdvgscezzckv7d724yl3aatspy3hzpzjdpymj6vgphh9n5tt386hsga3lz4ev7a3rv66whkzk3vgdxdy7sxutxgaruncu4u74k4c5lxxcjyw9g2whu47ch320md3d8u52d08w2a7pjqjkgprxta7mk7996ymnd4d29qwjy877fr3q5hdzn8ypwwxe82erpvz294d8l3h760pk729ycj5224uh2rzmyyjk820s09kw0fphxgjqhw3gzypp6ahcwutpdswua7pxgy5gapw77vzymqj55p8k0jqx5pvxwegdauk335kgxva9n2k8t0dune6uvayteq26tyn5773emdhhpt7njanjgcanms7hf7z9mphcghw3lzffgt92zcyxyxr2dycf6yrudn62pdv6yulue8las8dd2v00p3f6hzc920dg3je87pvwp5u54rcnr63777eq4cxqevntucjjath8qpfl4vqu3zps7d7mh7dfpclcxp62gwtw43vny57kndv67kl0adejzcdhdes2s6aptn9kj56y4yv3072uc8qlgn9gk0yf539xgrm4zsw0savys6vy6r8al5vm2d7jzaxywuvl2gdzycykrmc3xmrzwwq7vh4dvxd20vjfg3dhsvedd85pqydjtydzw94lxclfea9vale63qdhh5dwm2y8jzv9csymq07t5gde4qlzjmn3jz4un2g5pytmhkhmplwldumrq8sehpqspl622yd0atfwly2uv69rfecr9lasyydxm9d8zzt9zsnzeu696zakwa8hvmhuv0xmf6ws4mnwzueacqlkzy4zgxzh3ucm3lnlvhlxvff8neqnxmjeyujzt9fsxqr0vcs6384t3mve8spw9lx7yk87utmwcr4qp5rpluhx8atmhgqny6uzclx3lk6dy8yva2t65uh67mfvy865hhmkdwmlrl378vqrlsa6acg5n0uj4vwzyezvwt56w79hc56x2vuz0grsumwxwcyuujk7x4fzw9x7z2awrrmvq5h7lu2y9j3hvffvyrv09avzq42hfez3l0seygxhfs6ar0cn3f2q4wqjr4s7yhrp4c2cnnrr35q9a45t9zm6cntarcfagauztzxmsdwpw48d53k3rgzt4kflz50rccjalk02kxwu507eyzhrzm5vu95lgc68xvv2c66qmjzac9ctz0vr720h8f8j3lnucvhqagx7z7wqsve0yn2764rzewcexq2q7zwykj0l8ul9yftz294nzna4xk7jsquyzkd49gx3ydqv9dzuax3squ8pljtna80kjhcj0vu638w3a0nud7a0taaz5p0tvk86xpqszelr4x5knt7vytv4m2yvg3gj0n2w7ehgsm84k8n7e52czrw4ckj6wklwuycthe24fuc9jvnaupcvk85dwc5dh84cghvhusnmld728cjw2dezr570vqc7jknzvshahdldq6wfevzpkk9xvjwlhnzt5hn7u5f8z2u8d2cnmyr8ahl2murfrt6y93j0huk5e57ywh76w3r9x445x33ppe6vy9fplnju2g0hnulf2xltyf486xjvplfcqc2lve2hyh43e8llla6qhpmpavj0cgqrpt5gh3gxunczhh7jys9v7ecqlkrkcekjp9xlfa702g2axcthrhkldrfgmuzrgdacawe3wzf973whamemt5wtx27uznhqhxdways2e3xgxpwqcc80395zhf0l5p6lcu7g2t3f47kknaavp97262rc9a6tzawrftjlpzyc4s845apd8m0qdh6d9760xy9urt5rz6v5a4x9j057r2ykunzuuze6u8c60scdsy06wwud3l6dc54ka0agqjms9uhnyf567eel0zqugqpp69f98tcf28crvl6jtmyf69aqhvl2f9scydyhfjvtq7gej5hseg4ctqhwpduhl2upnkwqh5aahtj3cgzrdlu6nayjsxvgle2ljvfzgrn9u04slnxm7ee6xglqpehjssvskjx706e5h9wejq4f76yv7m43llwstpjrfn8vy07hjjr9lare6fjtww2kzgjgdnxr5slfzm67frfyp82g77rsc7cv9q2snzdcvfwqxy02207esrmcvxs25lx6nrd2d3g38vej0kfc7rrclhkxqcvwzt6dg89642yjcthc69r7ljsj4d4v64hlczrhwlnejrsswn9mv4pu9xwzuzs7xee5vvjkmt327ujtaa2u2yk00r3mmawtnl0clpltwztlyym75r4hmc6egef0uvsst5jn6pasnk2rvhx9v6d8upxvul903f4nefmnwwlj669w5w5fn4a306fvgxckdtyxwjhnahc3urvl8sg7djhdu3gjl08dj3vzhke4x2e070ndfn967qtxn29thqz5yys5e3ujsrczw90k4rmm7knnvkeqlj9v4myty9qnp630zsk5vpcpsvl8l95h04ceknxhy9xmce287hnxyjymnqrdynkejyl876epyv5fwmz9xthg9rf80ygxzujncy9a7nzt94rjr4rpnn7aacv5hcxut8e5rg347n5urapg6c0w6sfszehetpszng4p0uj4kl0jglmqm43vfd7zuseml0csxck80rsx709gj9zzh5nflz9nxc3ezedyexcm93h2pl4p8k9vn3whra3a7tq6ft0x9grav72d4pdc0pawlz42meka03x80p8rtgn8yutqsckap9n9p22zkxsrkudtgu7gymws00dqjrms62dpveu2jantvmkkyxl2zcmaa50xx0dgsqlmhsztyj7dnm9wz56tyvl0x4qg3stqw4yuczvxq432zzt26x8c0nkj55pjvmj2j8kqruvv00hq9kyhumap00nmeqr9xg3g6gupeqemj78xf2j44q4xvjyhzvfcv74r7v4n0sfcev50ym3clhqqxgrh6rkdw9u6fzctv4r50zxmgye4ct9ajzqdjvxlu4rn6tdj48cwnmzz2edpfe7pe6q8gv58e72gt2yn7jy8etwkw0yu2dm47rmxhf9q424uzvknw9836hddr5nxkq0ykkjg72clptfz9udlrprx0g9tazj93gl6264pusdd3vtx43hc69te86vjzc8s0f990jmkgty70e0tllszn4k2x5uu903vagqqzj00jew60aymrg0v48s6xg9u7vzz9phgje8aemsl3xg3tw73ls05car03e30rlxm4p3gw8ucvsngaahntplyp308enzj0gvk8kt9e"],[1024,"nolgam1vtxnrhcwje6zm8lu3vw8zsp9tcjhr5svx8h8vrztn6xuxvycwttupuk4e7lehmmt3e3frwhg52e6asq36cjnvncg0tyl0pep8lm9chw0daw22nl6fz856pg6vglr0dp374g6yxcrsvxt4sc0wz47gzlec3rha6zn23e4n7v4kxygvx0l8qyav8r3qg5wfhneaux8pweq8hmk07ds5329tgfym6f77ptlzk32jfm880r63x8gu32dafrvrfhvskym6vfuffvzghfyh3rnmvd5y08vmy990mdeszu26vpznuuumklrewvmfrktplkev88mqjvg3lzxjq9c5xejrk7th750trl83wavwf0fck7effuclalqenfcsv8yrsut6f6pg55amplw89dlvexrgyrpd3s0mq8rpryputhpqn8gk9x3h4dc7ft97yp3z8edhjazymfvzytl4trp5y9nnwmm2h3jq337yfznnrcnr9r7qyp5sp5mu7hsxlyxslnae3wnnzwq3h0v8zprd5cjm67alh2g9zm9wsw8dz2fuzlv2rfj4mddz670m0slda636c3srew4fchtg0cpgldp4pu7228tdeyu9uh2rnmpd8gqpxsjtwg3z2qrxhkcjxsuy4g4vfdusx07cy84e9l7zayhw5ddwkklssy4xxdxshehqjrd0v225y2y0vpqjmzemdgrmc8larf4aknpe0hkdv7272y2s95z9ud64we760ltvm79ndwguqrgweyf4h7w465kxaufmyknhtmq3qgg28csapjzku83j9aan0rxzvkn5ac2zf6z9k4czj2n8s7fdvfg6w5ptq8dtdlq24lnlddsumyxhec8c8xjvezx8wj05r3djpnck56yf7ma8kwr0r9s2qavfmq9lxy68c6sj608vl6n3jf9s5n5jva776vnkmyvvrrk9rtdn3axcekkuauh7kyxgrua6vgst044wm3hx503eq0f9yggvtvgj63r5mldzxkejtspup4m2egx7tkn0uk7zdeuv0ktuz0m8nsv5m8d7qkxfjvdau4hc9c5yduzsfxzac0f2u0u27fjt4la300vyxedgjmfrhr2uzty69ver93d9ettxdmtrlp9csqlrd9ckny2n3g870rn0xzn8pphjehvr4353xzj4lfypgxt36phdyv2h96p05ckeug8ahcq3qvme7mw8xe55d0zlm9r8yd3y3xsg4g006ch32arz85zvgw0gahuvq2g5npcszmd7x8ypxatqxnxegkgfe2v94f2qq089efrtrr8c5v44y45llytu89uylzm955a4jhmffchujm6e3spl8ftsv25t754rfpjl3uxqvwdu32acxsn6vdjwd9jswgnx0qqls8060q8zekc78qm7wxxnpmk39uuppg3wfzqw55aq3k6dxmc2jzdav70j5czkysjpzar3twherdpftvkceh75jmlsakdcvkw8vmjf3xmtugfqelf63xt6dd6wwa2ce7xkl3ettzuhywvy2p05tdne9cxxyws4y37wxlpuqp63p68zgxjyx5q080qfnj6aa3mqzqy8q24345fmpps9py89d30un4d6kxsmrztu7ah38xk8paavhnmlj6tprsf5zqr7z5mg687ax9q2tac89pts9eatm2wv5q60enz0h9yk2gx0xxu3d579e2plwza4u082j72yd726vm6yk7kfctz5nf6w67dyqa2hvgkk9dy6yppn97vz42gkq0tu69hwv7hrpfxwrtef9948ewxg90qdtvuk00nu3hy6hmgr445m8j9ke3kzqu27zdkhzzdr5psk7gcaz04nuj4lqmq685f09djl6h3gu5r80car6j3qq08dfj2vjj0kfx7jxqgl32wwws3s4fsvsw6t20p57p5e0xnd7guxpk66sx76pvltq0398meshvq2wqd5dx8w72g7m2hwyy4zjgsep7mrtdvamqqpcvkxfyek7l839mxfd3shl9ua84390z0crzmrzd9g5pm4knk7hrfygk07zvsqm8qjerjshc2656ntczzdereqj65hznhkyyux92r77q53qf647kdyjn3pzl9xfdq2hcw0wryjsysz3amgkk3tdu335kyngvev5hu7a8u24wpnzqr838tyqqk459tm42vql6cmlj6en84vjuqu9ukdpxv0dzlupf4dkax94fvjkdh7l623w8k7nj2qpmctcufwgxushz4a94848qpct9zvmt7nwmp87c94xavc7gn4ud2jghllcumaye0pk8n9ywrhgydz0rl35nls7z3y95p0pg4pffd0yu4w2wucjp53hkq7gersl4d2h7upqj5dm2ggqhrakqvs35k9ejc2scrdgn65mk3pulg9vhtu5q5h74l25qee2skjsscuqym3xpts2q3vpt9wvffm9qrkq7s5r95ahd08hy8ktg338rjgu6wcwstr3jynu59h5vgshm6m6xdaadn3peqrgtywn824r3ja0zxvm799uha3j07nnz7chr7km6z9r6mqs3mph46uaz852h633979qm7kwn0js88eysmk7krfv7w8rygyy9e7xxhp68nkttw7x868hrypq96t72ajqytnkrsfjgwtamcnmuews8jlgmypqmm3lywlzf6krmsr5waqh03tqw5e04v7hyku2gkclefade7uknexxe6se5a59798j8a5p39d2jyc0kx40mp3u6djcy4tx9njys9e9nrcy2eannk4nxg36d6rdfffqv97ves7422yff456smv0jhv0g5gdnapp3s89s5d4trsh5cvr0960aky5kxjppv2md35ee9c62m5e9q6s56ep8szpgu2u3d83mk8avf0lrmac6gpd6adsuk74gkdzs3ph8wu4uqwvc6d766ee5qx32lr5enwpc8cgxev2rs03dxw7p9cmjzw0u3d290glrewafgd2rpmfzcygdhvqs6j7s0dztvtu37pqwkf2rm38mefjgmwgp2lu08gcv3csgzw6mppu5ne2j3wpjgtg4ak0hv8328wt54x093hvm8ewszlpva62f09zqq0hmydkpvksp798p5gujq8zgf4s75z4ukpmx82djytp7hm8dzrcrd4ft5dfmf4ams9udsw2uryfc8va2mzm55ftkrp0kspjd4ks5lfhg205pge7ng8x4cggct32yhcfju6t7m0v7c80c2ug7medm3874gexelvftvwhmzmfcacmqmjnswrgu5duwspm4e75sna357t5t2yqxkckcpyhxvslnnj3flp53ppfezz8t9xr63d6hm3rk6rm3vt4wduglsx63270muuycdrc6sds0m0uqs2jr7fddmyxfat9t7gs2ytk4fw3zvth8gzd7u360uy7wmjh4aprnftheuv3sdarzcmhcfgdwp248tg5cwh658x5lprn70yk4958y"],[2048,"nolgam15ghkqlqh654udkqw3xcr48tk804gcg26x62w8yvfj8jnet4n4254an9e5mzjvkhd4a0rrese6gs3cf2f3qhzyrugjxgst4ng2t3hf9p3ksf5j8d2pltwe0feqt5kthz3pa7kdt24he8974ad7nqrm6lj3yefxpg7vkf0hp6jfphhvnedmrx4tu3j8qkvmmzra2rgaz9mua40yqzsy89qgkkp724gkdw3568h4mar0kehv9tgptyvkqtwyxkdux863zmc6u26u9d6wh5zz5n0m34vr6ak78k2ls9h698vde2ghxn5cjmrste84fh8v8fsnzwz30z555hz8kquvd9awv7ja2e07dqcshd74dcfp9ulyrc8g50zxt0a7ftp3txacu3emz6csv0apzmvazm8uh3j59mgphcxmgshl32af2e5ygzwgnqzcdwvr669dz9qs5k8at6qssvz7jrqw7l4e4puk3t74rcxt7ke7k9ufjx7vg3zgm6jzjr8hlhkazss0wvpajx6zv7jspexze76chr60c38ntc25w3nal3nhsm9agw6p5xj9vsqsp90elrt62lh6frejz5nucu4fhxww3rsktrjlkl6ehh2xjklcnr2n47x0leayf5t9mf3fcvlxfgc4amwfspv2a8cvs3capcqmaeclmt56uu3z8w5cu8z9xsne6y9ddlvcym6p83gcta2hmew696nl07r2gp9rnpdap8z80lhxgctndjlklpnthwz56c4v0ju440yfw5a9k6eu44gfmpv8klmqmhd8ntruayz0v5gldn9vdwx3ke2kws89mdnl7tal3qstxzc208ztsd3dq0dlzrxezy99ctx6vdqucwa3qr3w9946lrnz32zzndyk9qn449fvlz3u75fcq50c37fcpexm6duhz456yp0ukulhr88dt84gr404p3mjduczpesk4j6sgthajpt0egtvsnwdgc07yqtwh5gm42x98pxhvekpcuyy2sh3j0detkrvz0f2v4auz5r5axll46e3wc4wgtezk80596lxan3wseuuxm65yr9a7cnyvg9wl95z9xf0a99yqc5s0xzeem745phfs8mq2ccuh4005pskhg5xayqd0zqayuwn8t92snn63jh6urm5gfesdla5l2r94zf0gcxhpalqjcu72e3szywvqnk6x84l4uj6hy0ft2scprpuc4u9840wc3k44qhw45h47q46yzxvjadeqx5g3zn47zpnm0yge5gx2ranptc4hcggf0fyd2kszxyhvujdscp50vagxsrdcdek3psdpatlqca8x6pl5re3cr3yxvst52uamfyrdguq5nm9y2ffdde00u2z8hpqlftp80kpy26v36q79s4el0z5y60zl9af76s3e834lyp4e0e38je4jav0zsznkfj6uucju778sns5083e5v50qp8ulntvaxhucmywaxvr4nzmzdqv03z9yufkwh9engqgfux7y3ljfymwn9v2wpaaearcqdkfckvrhjcd708qcax2p6ulxc6kwdde440k0shrql2mh73ta30m0dxvg0j8svre4eh0pmhv0hnq04rfrlx6r83ufq3htnk95l6whm95t046w3ayt7e9m2vd79vhckx5jl5afzrx97rhjdhfudrkn4fuxe92tgqccuqykcys30zp9vdnwwkztwhe2w6rscr2samg2wujzgwe35hjst29nv5eq6qqrrn5vc3wye9j73hfy0vfcspd0rttalsdtmgmycu3yvya82kfhk3tmkddnqy5kh5ucfu07ju8s026gtug5m4pspevssvp7e9pyq69wl4lkkvuke800cmt4jtaejcpx8lyghprg2sk7p3v7rwrtzy2ctdfa7lr499gxm9nxdkwapw6t7qmf2hl9c7j5c26tjuuzgvelwc44xgye4t4w79ett2ceh5tnzv32e3pxnwexelx3ktera4fmrsp4sa4655e2zw4erxvjjzjlvqpwl8kth8hl256vvgftm4vj86cvku4vp2c3jp3dnh6ekst7xjzsjec8qd36ruahx77q4hdp9tz2upkjs467ta5s6tl2tp7fkp96p22673yscwz9lktwy44e2su7cr8pq3dmfrgmv2g56f0nd4d78pvgx8wa44ckgmmd3ccevgdrwvwe0wxptld3jkknsdalw3spwk2ckds64mmpx7y8vzrrjh2t5tdr6kkdtdfsgvp7dk9pk58r7yc2vypzjps946223u9yjfujscau86hdl8x7j33gr97cpkesxg397deeqnvlhm5hqfw85k4y90gda9lv0puhdrx4wl7ajuqfxpmh8sckefsj7ax5u9nn67jz02xpxr66w7yvcxpz3fjd6e3uzs5kyjw08ycux2zdvcwtnvntlsuu67at3rl0n2hsyr2c2hk2345v60hknp5s7u5l4pl4n6dcfya2pdx8xcftxarpan3hw6hsfdepq6vday0mfy383r67g6qhv6wuvhpmvuvkv85xevgwaxlpjxkqltsyudlkh9y9wvwva4gkv96kl2zke8x30h3jj3rwzr8kpk3pgzrgcpk6awe9n5gmx5k4d782agd3355mvcekvkzhh8x6n5qvewgpn27ae4ku6agl4l0v9wx5cncz0xhfa2ec5jqn7x7l4mqx0yc4x2f9jecwqdvruk5hjsgzx48a2xxg7jg9zejhpatt4c639y6qys6rmylzkc3akadw8x4xxt5ugwskk43dwfuph2dmdxgw5eq2h4kfdxa9vvkys72f2puut3fqgqpnsxxfnll93zyez9gwp0hn77w9p5f202hr5gr9x9wycjr3cw3sce0ak67f3evvkzenaq4kvsjj5sn7amm0xpxnu4w4eyct8a34m9tghxxj9j604lwsj0gtt95t5pw3746jjg2tzwelj5s3ldj42l8v90ftp99726are39rnhjnp27txk8dvvdzh7a4hlhnaas87a8959vhrzztakszhl25f83cn2yldlexqfkjeatqjtaldjke8y6yw43ut4dnsdgk4w53g5ca2py7suvg693s43zqrnkcc7kgg4fc9ks0ftaee5gse9z3vauymw76q6lu7rh64jak40rtxdl0m0qv2ay7n9ztmnv040428svkjxj2h3ugcwld58g8spw2zu2t5t7pkp6fcuupr4rzlu9spvjq0xd65gnqucf9k0f89v7nlk28fxkw8m9xu60d5xgh2qaf4x997py9y6q9d9wjmu99wldxs9gsj5ykw37jywg0y6ax68l0fw2q86ku3muj7508ypsdmuuj5zjnk8yhdgmdzuf06vcsx0svtvks997wvvw5vwrs352w7unppdphj8tsj29m0v3zn3lkggjd0ma79xq9qn08jsmyqapdysnk952pde6g09t79r4t7ptaxlhu92yv5gmlctqjgh7y0"],[4096,"nolgam16697p5c462kpyhf3p2mc8ur74r9uh38lwqsyyxfz7exv5khtfaahpcc4gwaq0vku63xj45k3mg3uujmr6yl5p0pnsf2yw45xumh2r83rl7w2lj38eymadv8gqv8fuyr85q7yeservyrk7epshqjrrxw9ezy9gwmnqlxwq8araaanfa256k5yvlqeccdw0h5r6t46rgcwgk0505cwxa07ctsxe0vhqc9ua85d8fuwa23k9cxt740cfhnwx6qrpnnmarvphnfhm3jt49wfyetx02zaav3j3tfgghahgwapq8h0jvgck8sjm5gn6zfxal7rt7ls79u68tx4zrnkr3ycuutrzrcef5wusyhvuj544nl096r76krvlftf2tf9rxcr5mytx42eq96k93nnsa2tyf67j9d86c0z7mn8e3aswtnd5tdx0z6x2x57jxt5nu5dg66w406uwn27l80hmcq44nq4u82dzm9g3dgynl2uf3468s8n4gj0h4lu6q0wn07cdacaslplzvcpccnmdt4lv9rjz5g65he5arjnxvrtwlvlyehtlcptu7vvnqyef59m3dnjh9v2ss7uqkexvxjhxs28tcs406axu9k2jlljhhqmc26pwt9z6j6ht72m587nle5y0pwsnz4ezvzmuwttmje7yssdnyax98nx2jq8d3nrrh75xesjkdpqp5qq87ygefczqgc8yywtn0zn7v9q7r8zvpsdgdvzdrk7wmnaqhpmz46cg0l2eruw3pn7m6f4etgsramq7gulf55w0e6qff759cymg5s2rzljvhrqu287dh4cgpqz4d5m03huzfl6zjj4sgd7suq2y62t424jkw8px5utlu275rlm5fkywd5s4tdz3xtga3mmhhjqu4l7e0mvvm86l07scr5e33qcf8gj423ld67lgmqadaeh63qfw2f5ng79eaesvy59wc8qgt63laf0h6l5jf40ucp78u0km445g4ew6vz0s0twh000wuk4gysxh89np0zaxrzxwxaglaj7yevftf0cj7rrrqdpktxrkz5hszga7l7yl5hc944qc30lctql8qttelyhmkj0zaqgns59xhjxwuahukxjr0rvhdj45p7dulw32c9v5ady0kg09d370j4utc5y6vh5g8zzs72l7a9rewzg2kx6rf079e9ys2d38e2eqk49f807p6lqun4xuq2emu8599xr0hdmx7avf3kza6uagzct5frv8v85rl2w78nrakfpnr5wdgetunyk8t33g7wnr3g7ag6ms6xnzprcx6f8grzqw0g5ppz9rvphgw7g4600yr7xnkdhmstmlrhrssv2vs3gfv962dnzdwewpas66vt32usmsn97cen2pamhynpqmdpg7f79dfuk9v2z4g4jxepl6m0rcl75tu3dtl4h7f9amctdg7n5ne3ky9fmdqxrn39v2z9tmn0sjhkp48q6q70dxgxzgumc5887kmwukpgrkn09vuqvwvw6yg5x9tgvjy8gfxf00c4ujgfmfw28jxnvgvwceec94q5c6gnl37zsnklegdwn7n9csr49psrlgrrfcre8qadrqc9h89s8qjrgakrrtuy3xuv2fpdvnsty2tuvrldxgjphng7px8ag88yyett3ym2ldna48hy9jp5p8hp7lzkd9wv3ul5mxf2ajyn0eul5theqyf528g40lu53x4l05ghsd8ka3xth43y60haa85tcakv275grwrdxs837nhwhp3j3stj3xu72pxj3eq3l3dgs892jsetq0m3vxcqymqzh8am890s2es3yc28ttsyjv0hfm62vrr0qk6k9mr0h4tnumultg8zxjx9m4n6uh9g27mglpad7n2j9d0f7rs8z5dft40s5zlkwrq26rnvt94f6gqs22zkdphzzj0ju944veftj3kpwaxdhrhaaqw8y3guur76xe9r302865cpkgr0cham554vpcjc7907dvr2jtyh66axasjdp2clcz2yd9f83zyh25m9jv0x2jmnnhaaqpt5l6qhsxc6e23lhevuzmmxz5hgn5pr5e4j45kwdzuuek3l89e5lra4wprx9zjmmf5h5t2h7ty2knva2cqn8pc8k692gmrg43nj4u6wzvln4lru0hnzg7z9jrm0gul3p7er7vq2qpkp36e887r59za53np48dnszf62qyxgdl8fl6vsapqzxmxx7nkswzs8un3dy5wzs3vqj94a2j2gx2gpkmuxk6yuxrj76cttk3fv0uew0g7wmyvc8y0f7pfc6a7x9yh6wn0l75smf62l5lg683x3g0uw5tah9c9r8qtr4cse8az20cvqv44wzl0ns4h88fx65gkxp35qpcuv6m4pyyskzmp3w6heq26e9ruvlatht8qr9gp8usxdpngvqdf8xccglwm3suw9zhjkdpjnyvw4x8lug3tq6vwel4xth68ue2ex3w6jeuux75q9cprt4ex9gevvjawnsq9096nxtydr7leyv06c2ckr7udp8zderymh2tg5gj9r9k5l2fyld4wl8flwje2zctfwzalg56drhpnrvh5quf7kpfdvxysp920r7dt7wkztdkj9cwwav6w52re87ywmt3hpvnu2r62l75tduh0wjylp95d0vcgq9grx4gm4nc53vkjsmvcrua4vk045a9h9dcz6r6g86apg2sf8qfxrkzld02nal7h7gq88ulctsp4cfqtmgwzzqt0n5zckqtc3s27werhml64d3u34jmt6tnqjm7s4hnn799fdp9wpwzmaqxa74kngvxy7u4humr3l50gr9zny5wcw2cl0pk0er3wugspzjvmsv2k2dyhg5duvmfgksg6pdy8h9nuh82zwp2pfmslntary7e4uqws0vme5uf2fd5ruwjk0ay6ste09dhp92604mtn2lk8mvk2jdt79gvv8az3gds6gxy095v57pd9vmjtlh6f4t7capmhqf9ershyzg9dzf9wftvuk3jkkc3fh4pf2z9rjfah9ece6rpqwpnn2uxdtd7nmdckjstarf3ftx7fn028jdcrj8m6qgrmufzdkdjup3npq3g3g8vu7gyedm69fn3tmyphxly4wu6af9mxhr5mv8yc8ekqlcp9mamnkusr96cm59ew3lcnvarlk0f3ns6vsqudgputfmru6j79re2e3h2szk9ecf85dc8n7tptt3zqlq4d2ugd0xsydurg73nrwp3j2qxa9g969xr9qaags8cjvqg8ur0f2zyq5ftn5vsjh3lfaqvx9zqn3hnwzwpr6304fzd94v55nqjzwkhxg6n686t9tsdzn2wfe6sfy6l2q6zeevvtyj8sfggdnjgqugycwtg24arnmdttssnaqep6nzw5h55xurhavurgdpzkjkc5js9zchwwgs2ls4qvkd7thpv20kqcxzwzfk9qc96hkrtags2atev55"],[32767,"nolgam12r24kxyf02qnf863xl49ex98q4emq2rf9thng5gxx0kxtv5782lnmg5m47fx3fk885hte30a7efxzly98tatejcl3t2w6clztmtz43l6xamd6eq6d5uyaurshqpy0awhph3wshxer6knxw6ygvkngtdz28va2qmedmwxd89np2xjj0vul8snwld932ucnkcwkya5yftr32mdhvn8x85wmne0a7w8ytssn40pgftp820y266ukf6t7qrm0nts54lxn8u2lm64clqzv6sxz3vfjdrhgrfumpm6af994jqkfw4z65gmkrwwemvql2yffgex9ammezqd0k7velgusmraszrjegqfxxd9vssks6ev5l2ty470y5aw64eckye0frvezghzvn7ws2n9sdnthd3hqas6gwg54us008lthlhvjkqrmws2zzgpmdlh3rurmgwhlglkmdldq7009laum7dyfajz7s0urgr9z7kkqgyeutn7qdmgz03jnrmz75nz233enlm70lh8yc9j940ffd2myd4a4fapc8ma57hqnc6rcvn7s9fw72aks782d55uclxjpw4ymx8mrgs4zkxvgazgpy7a8fwvmjurvyn6xssfqc5z9r3aye0xvj3qwkgxucllntecjzzvsq59fn2f2f3ph7mkjhy6hskhq3rrg5jlsm4r4sdy9t9x4ay7t8x98ltjq4zu2dhg3xsqmrsnh2wlrdvp2849h5rjszghtfx6564an34975qcaz4j430fkzxtsv7f6jhhf5jgaem2wkrf7mp7hetys0jdxfw5yzac0umsu0jqxswq0ezavs8kd6dhd0wu3ldtc0anfj6g6p27mcz4xrr4cj58xvzct0tegf7l5lh25wy7fr79h6c9u0mc40ul39vx24h2c2k0ndw0axje9fspzwsswypv3x09j3ykf3qnf0agcdyxz3dv5sg3uje4ph58h0sq5nwprlkydy22vqv73tl2zvp7zgjmj67k5lsw0urmw3rtn7j5nwhq7gyvtfwvzp8fh2397jzvnuf2g6y86ku74ev4pte49pvsqf5kyw7h9sj34xnfp4mmvpx8wsxtmf6ryteg9ympklrq7302sm80u35kgt6kc43nwxtdkjj5m5glevur45yqkcrfhm0hmzchnr3fes9he3tks4tp594ef320mmk9n9ye98q784v6wj2zjf8lp2fz3cg9x64gx0rch3f6qda4ulj8f8mr83y6yua6m60jczhkjz98ssfe66hpgw40nrqe0mn8wpdxzvpk9nwgsr8kamkhfa2xt0se4yxxlj9x0exzvq79f5m0msfsam2nyc4wx7u3w9hjuq4ht8f5c79evc89jt82wpfkys3kcdw0e9qepvfgu2lxauhd34k7ch0ey0dqxvcgkyy2tlkc0ha8sm0vuwnv6nhamz7xv2ff08h7t76q82y4p7fm4m9s6nswk3h4ferlpdlcrs3d37lc98p7m6uxgah79t5rxznfhmx0yus7jqtqm334vzh8y25jgggzl5vssc73f0gxjgnqnzh2xqtq0f0wav9e0r0gsl0rqtceh0n9wlqxpza08vlls96el8axmx77pvzhav34mfkvnwgd6grzy96fmzjkcyc60uy7afla9lvj96cvrfendczhsd2jqay8lg7yzele3qtcs0ecmuatm6qkehqam2dcyfssaacsfkywd6detgt78lrl2hp2qa0xjlm9ml4prlwvcnshgr376pcdrp623k0rpgxu8pnql2rx8ncpmkgx0f2q8epvuv4fpwk9xxhxsjwxaaa592k6wq00fv32c3sx0gdwj40cjx5fd3dm99wvl3nate0880l56eddp626hmdltdc57afprdqf9vh6mjrta8vmqwg8r266lt39zjjhj4clxg633hx3ylutysc00vr2jw5ww4gge99k47le8cccttv2yve05u8dy3k0znj9flvsysvzhweu9cnl66k5n7vzeszx59ttf698fmv392uh8dlajg3l36eu822235xmpvaumns526u89qhl9anmlferuk4vrtdp2xucjzgd42n5vydq4szsgv5tqcuw28jwvg9c5lydvp9r8h8xg4tftymty3zmtfaey80n38s7d6gvgfjzv5ahy4mfd3lxua4e4uuntd49svrcvpc6pgj2fp683nl7ukq2pzlrr7hvywscfqpd2evy2ppcmp6lkcq500gfttz9v66nh07wfs76m4a5lpwds3tq2zpyepakaqaqk5mjywhlc3t0v28hzwht2r09szc9gu5nrp6xg9zjd5z3xdp9mxv6a7jq9r3svsgh4gualfukdqw8qn32jf3rdapms2eazhpgscgzw5va6uqwnr98ydxqeqn3t8e3ap5demts4tf9zt2zhcv5lzxzpta2mvfu7j9zlwl2e4t6930jclecyrs2kcpmljl9qmd7jf70p7z0ffcy6ej93kwz3uaj2lpfc59mmhsr9gv6tn9c2andxfxw62m8ypjvmvyysx6cnm3djsttwfesyjzcxtw6f2fc6wqp3hyg6gf54f0g39r3kcgy6u8r8aamj7sjg0u06smqvhhqkjkfms784lcm3gqtzkrudgx7xxw0xqmekyd2krldcls2clc7ckqvrumfp98sc835d0rv28dm7edhpg4q0uymzs4ze30a9m28cq5l6r5rmxfccz6rx3l8e4zq2mm5gua8hwu3sc9eagpqswhsrk3jmyyfpp8sh07ldzmt6thgg4xe0gr7sunngxs9t53a6xtqvl0sk9eh78euy8fupc3ask4pl6yq8zldmcqgm0lhzcz36ku8uc3p04lkzxn69eh5rwchu0wxg2cumecvhevsmwa9ngugk9r7dljp0dt9nsee4hq3sz03audss5rwtn0lhzvws33dj6k220svwxd47yn0ytr44qptzwq6ll5syw4wv6xm7vngcemtpmcq4seu3jdfednm5y2tzfqdf5y4q20lespfxta0y9k3jt75k2u8tp04qltskrkwhykrj6z85md8kgaz3r322rrejztuzj63tmh383v7wlux5387t3ka6s87te9a2t3fyfcxw8z8ra2u9wssv8tq7dfsw5zpav4vm9dn7mt345yf0c9rlxr9r3jutc3kxkhyx08a42pc7dhdryjsea0hu50mrzzyzhlpn247p2vdgcpssdyevn0te7g9lh8av4j2zvw5hh6hhndutkz9y0thpt75csuqzd7puh2gcjxrs062v249q0vdg9lfzns8vd8ecgt7a03hnkh3f0xjvtrt73fy47gpf72vayjpawpfxhazgh50kn3pj07xeql2vu408jy3mu2szz928ska04x43dhj585v2qrxp2vrs0ayt9hvnqrjdn8n87r90tsk985y88ycuxkmj9zkgjqkld47nq73geslgztlwl4"],[65535,"nolgam14e9pwpz9ps4gsl5h37hfu527sfjkt5tcdx09sta4yafvj5ky7y4w96kj5k3rq9hrne940dtrpqd48vhj935vxx85h7p4tsx92hz4cgzqfx9wrvlanzu4fqhpqmk0lw4pd4nuea7uuhuuwh39s3ah52rp5j2tz6a85lw9j3z0tsgehk59trxhuw9dt822hez0p3qu3h3q432u4cj2pgeth3qcc9rpycjgzfg85g74cwj972zm9npxu2ccd9ueuqu6z7asfg6qy5pyeu58u6nkmj25d7mxfqn60stacfvu5pxszm2dcqp95t64sslgp0k4ec0vkshw2fpsmj7a7k52jp84spmzmnl2nsk46mcw20nukaghw0m44r60apmtrptctxuce08h58j78qlvnx0td36vycknz38shga44pgffpa3hs20n34rtdlpyh5cepqjq3q70vf3q7mhtsxmyszvnc5ca6xzt2c9ejnuvhv24y2mf37t7x6f5nt5glfhqcx42nac3qhd2suxyr95e4mfqtsqhvj7dsjh7nedvcczy0u6fp90td0nt556gaz4x0whyq3lhqusgccqy2nfjq74ka9w73vw0q6amw69r0860nxf4z7vduzkn444qjmdwayqzrc82l6huzzhzy84kumzntvsdyc0yeyzvz4mcds20ujry6lud0wgeqfajehrz7zd3aqjgvqu6jrtgpslw5n27qpskshvglgmh8djahry0stej68z0vkeygu3vzsxc24w3r5qu7m9thele9xwqfk568l3t7njpgrz37rd0ymsvpanjyu84x9ynm58anj7unzkk0rlnr32akcgdr98md24gg9txaxt2xrerthxf7w6z6ltp8rzvxqfdpaa29t2s9rdffrx4gt20rsunk4cjyl05l3k49s2yntp2hd2gsym5zl84qdqcfk7zfg9elpt6vwxplwzzgtvnl326s8vca5ls950mctvk5vy3d0wqhu4dv2kmqxu9tnmfnxc4asfsawqr9nrm974heezd6t7cde525fxlfsjyt8xwx3vpvvp3kdnlhglgqajv0v2yw73tek2rs2prdasjkxf7gsn9ulms45kd39swpkvup7sue7jjr79y0cnlqvhrz3tnrk8xjsr4gwet9d888lu0w3mamdg75lkn9d6npsvyl49uc2r4fln7hgtq8gwqgk66j8za5yg55k3prjzjcqvmdwhk4fvfsnrmhj0k97leecuy3uw39awe8vyg8tt33krpmzq285d7jzsnntxj2n09ghq46pm52rgdtythamzhudxhm2tsez40juv0lz8r3wev5w704dekxat7wxdgrhjha8xm9wdaply57ljssr0mk0vs03h2r2q9naw75vzrj5h65jml3vd7pt3kdk7qtt28vmyksc3hu5dpgahdpp2v05cl423axgseykcw7yclulw7m7ardkxemaggss0fn7rzk92uac5qdwd7hmjt33vpjra0yzq5vtf6p2akrh0fk7kcjed76xsf6425uv6zx6l4xrg3qnwp5xlmat8ajptlz206j6wj47d0xlwegkk6v6z43yf2jyrhure3x00j20r6qh328aahqshf2q8dwrpfu4l79g4dtq53ngzpd2qemrvv05rvnxkhzrlsmfpvzr9ysj78v6sanmcve5sdaj9uqgfa9gen3gxwg3jt232ynftq5l5yt45g3ahesxxv50rv4meh9f3l0nfswwcmzf87nq56l7tl375whd00ap7mmnxmuthq6j7l0hjsztur3xfyn2z4cmqj62998dpggdmm4ehhxphewuspx807gk60q0g2ll6gfnujy5kw35y4v70672fle67pf4nl5qvw6frjq9la6chgswy69jm92gyt7meh7ed05aee6qjzju7fl6z9xq5rym9h9eqy72zaqd8hv5mgzecym0rw04rj8jfeav5prayaqcjq6m2h6m6hxl53t78dvtv4n6gvu732yfhn7q2mrz4sdwnvtegaj4wp8c4ccu9u8344lfz7m5kts7fe458a2xwha8fu4pxrclf90m8w7u5kdfa9fhf4augd3n4s340q5trvnwr87uk36q6pgllzs569gyyf2g3fzpjmmuzy7v55a8v3sh4tzz9nu3xgfe9rq7pwqdq3gr22uvrpd7nfhklsmxp6kzn2ujx6td6uchc4yj2ulkx8t4hmj9vh7gxyw2srsn7tuyu8xkvh7z69sd5kmydzgcfu9qqrj3u7wyh0znmrrgp50lzyrmpvnhz4hs8al7zpx7z7hmfdannyq0sz2qkrsqsv9ezyqlu4vt07zcmqmu03pmuzyta9n7w8uhxcs2d5xwepdjhrswkk3c0fyza8epk5sxsskaujgzjjdt0smtj9pr4txkfa9ym576ltmz8kehnlp0v6v4dtcea3jf3rtj69uydqxjexehq5kpu594767vnpa6r7rqnukklt20lp33ps9j5x965qry0tl3n4ftfgvufwvv4qh0chdeaju0ry2s2qtk8avntten4hjgfwte2g30fg4xufmc45zndqht7cp7ynjtc9pvumnaee2mpyjgtjfm0kupus0suypyfehzf4au7udq5f7dsxsupqxcpfz2f529jcz6gml388jepn4mvg5sekvhnfz32q5s70kvuszfp7jgmme57ragumfsp8ec5jsafpdul23yk97gjd4aulc7z40ah4ye53gd0vjfwcw5ryn62phkgdhxkw3rmxwc4ek54u0vqce8hs2zlrh8r5m5878qjupnzz2438eu7f3gyt2nq8smsj7xxnedz3nfpwh8rrw0gx0enwgd4je429mjva5zm2fujfq8gk84vcqcxgd0gyvpzr6xwcdnu6dea4p8j7az9enyualn9s9pzf03cyh70xpwmfenecezasge8mtj4885d7t58s3m5gwakvu864pe87a50zss9cdw30x200zcujh8lgl6gaqvqer350y3wll4qe2rjhdw045vq759usw7arvm8trjhmcvy59z33m7dz80qr5hut7fmqx37kszzsjx4r8vvq4sln0pxvfjj73uh6ynqzvh98s35gd9r5apxq2e728efs3hp56eeruxcpsj427jgx3p8txvw3gzlvueqwfzwrhwkhukulskrh5a6cv3g82gx4wvledpsshrh5u3y2h8lc4juce48ms8kx7f6y48nqln59tqvv5tdwk7zcrsswc2zra4lfnauskug0zmncfp0vpayekr2evdp3p6ksesv3esjaamm3jghg4faekxakn2rwqpweq47meu6c20w8vkq88kc953xz6spg5yhqsk7x2650cz0ezlshsra206aemwl6rydmnuw2xnvlmr3u9097tr4ulhht2r9zqlfvpuh3wu2lx0pupwwgf0udms090kv506xjwl4vxu6c4h"]]
"#
            }


            // returns a vec of hard-coded keys that were generated from alphanet branch.
            pub fn known_keys() -> Vec<(u16, generation_address::SpendingKey)> {
                serde_json::from_str(json_serialized_known_keys()).unwrap()
            }

            // returns a json-serialized string of generation spending keys generated from alphanet branch.
            pub fn json_serialized_known_keys() -> &'static str {
                r#"
[[0,{"receiver_identifier":273904684339227348,"decryption_key":{"key":[13,8,84,207,25,131,62,162,67,125,29,73,164,47,43,47,55,144,100,48,151,144,43,91,204,91,130,24,30,208,125,234],"seed":[93,43,154,160,134,78,188,66,23,165,245,7,133,132,188,39,7,234,136,49,206,159,209,102,116,90,137,62,115,59,165,225]},"privacy_preimage":[16668675123578690059,18341806128591050102,1246742487899939509,8374509992006160184,15155321970247435705],"unlock_key":[15660138539970665810,2768208773744898019,9776494292307028494,5704907116912976402,10397876263944324645],"seed":[10642852542774074632,6030819547344830910,10633164815444652215,16253442335210725520,2360709231089825227]}],[1,{"receiver_identifier":582367254894409410,"decryption_key":{"key":[192,21,154,188,155,10,230,210,105,174,38,235,95,47,139,247,20,180,169,52,207,74,155,50,208,229,181,254,14,75,172,208],"seed":[8,237,234,40,245,188,187,177,220,38,44,157,255,173,49,61,176,242,217,196,160,206,111,236,183,225,85,199,137,99,53,191]},"privacy_preimage":[16072197254615499390,1341497956397543824,215126691660328374,11244170797436127523,7737220741132978740],"unlock_key":[14201556380720162266,15445351501201002606,211873681988348803,1822504044019930995,6306483395564950300],"seed":[2245910115176371003,9250431508474045989,9884686889294796137,13060398710753816061,11614512322999604640]}],[2,{"receiver_identifier":14788448213834758352,"decryption_key":{"key":[144,21,54,7,115,127,86,243,1,200,123,28,80,137,239,19,171,36,221,10,26,203,239,35,158,168,59,197,238,127,176,124],"seed":[74,90,218,143,136,209,15,215,128,38,143,14,245,160,241,111,232,252,15,101,140,71,208,24,162,116,168,46,167,114,228,25]},"privacy_preimage":[9612414495056300103,12033213927341075139,16253774228024816948,8042094109050213223,15406231770307767976],"unlock_key":[9962086481642866259,16217427397075880714,14402144115940208220,9738079569791634510,12334712979075723108],"seed":[10563202088645784360,18047411003706852966,11838689555566383561,11786724597725893086,1005293496769719269]}],[3,{"receiver_identifier":17387593842275028312,"decryption_key":{"key":[151,184,221,94,107,67,88,68,61,244,162,103,65,123,11,189,157,169,19,78,230,122,115,196,125,7,5,138,102,133,28,220],"seed":[149,173,254,16,172,117,160,243,230,45,151,27,18,49,43,142,88,120,104,62,131,153,193,62,108,235,174,59,178,202,50,233]},"privacy_preimage":[1630183294869365289,238113279661772043,520882888752056255,16847380582607118546,9852615318542485418],"unlock_key":[11632918823272446659,9443163243805321590,11665266807194560198,8475285105192628118,12089783969356215464],"seed":[9354543330576022855,14240737961211213057,2826628799684484616,6229835507149903048,17632291508964821087]}],[8,{"receiver_identifier":11409389270524167121,"decryption_key":{"key":[222,97,79,137,229,131,17,64,126,75,88,145,217,150,140,94,155,44,13,154,211,248,172,32,115,60,39,112,79,204,186,152],"seed":[45,72,219,228,83,31,143,211,77,232,214,39,26,238,219,110,62,124,140,62,87,87,102,146,18,244,100,240,30,50,120,120]},"privacy_preimage":[1434136946318461325,7261537935243178505,12800639952893549478,4382720082682859819,6719658405285344169],"unlock_key":[1664785851039397997,9260908295040230292,340678748975837043,14643941754583126001,4692380173588220910],"seed":[2677984163902157287,14323326924032851704,6758402509867752671,6718719500741091834,13295244432580800527]}],[16,{"receiver_identifier":624601086416416296,"decryption_key":{"key":[39,144,135,214,203,161,95,2,45,0,166,150,137,176,251,40,116,125,52,58,133,94,255,64,53,159,3,68,51,131,176,251],"seed":[146,66,221,59,45,142,190,94,199,131,147,169,201,143,164,136,158,75,206,113,131,92,234,102,7,235,251,201,12,112,249,243]},"privacy_preimage":[10850438342207107546,11404359657874156214,14941564338963407980,13239966383071676661,14788671197432907210],"unlock_key":[14103819235758255899,11796566718490419292,16441037706476319086,9852188795416442517,8776811481996096510],"seed":[15308243757016084685,718954801482632930,17697711766318084140,13998488933359953434,17984313484414971454]}],[256,{"receiver_identifier":4600038825852912388,"decryption_key":{"key":[78,24,11,253,218,72,116,237,187,197,171,230,144,34,34,59,244,162,170,253,241,13,251,100,196,41,157,6,3,4,49,5],"seed":[34,23,251,95,177,176,32,82,92,150,85,141,183,79,203,129,11,138,130,218,101,134,186,54,155,205,136,90,88,115,78,203]},"privacy_preimage":[15859162726193391069,10188544645673733363,11127625489886884377,3590334856109684894,13792973554074938748],"unlock_key":[6583537617295293030,3135804882123093417,4780309936460076836,590362263600766138,10922085139310488336],"seed":[8801874994872184445,17366990964505227856,4691840595559661985,1193148791917031265,6239643872384101934]}],[512,{"receiver_identifier":1618097638233204455,"decryption_key":{"key":[145,208,55,245,194,33,240,229,24,117,191,230,204,245,109,92,226,75,106,17,144,245,174,82,93,98,218,130,141,52,51,235],"seed":[227,128,33,202,48,8,176,69,172,155,122,12,54,102,189,107,89,22,78,126,137,91,76,218,216,220,50,43,230,219,240,44]},"privacy_preimage":[15897011588427740387,9684119893777951269,10584783532660942146,4551012108753889871,6351672086655204363],"unlock_key":[14091560630791917152,10093906361117673454,16477722165263664588,5655423783186994373,11412650348275934007],"seed":[5205664798712452878,10344510265634719745,9266105337161533415,9084386681879105435,16987230812128061909]}],[1024,{"receiver_identifier":3275407819623484770,"decryption_key":{"key":[133,52,124,127,2,148,200,223,227,70,71,7,185,222,2,65,189,219,70,50,162,100,137,171,243,168,81,87,80,129,119,183],"seed":[159,252,139,28,113,64,37,94,37,113,210,12,49,238,118,12,75,158,141,195,48,152,114,215,192,242,213,207,191,155,239,107]},"privacy_preimage":[3231502284138442815,3835863106157948432,5314039198010405249,7685660965729028837,10251658509646763517],"unlock_key":[10929830829418637775,16976300707128570796,15289423475144408537,1672498997618699098,12586667729509991876],"seed":[14521052207598645294,12387301719919355336,733993275851230918,3664063803918613401,10141114429453128399]}],[2048,{"receiver_identifier":14279741340355932066,"decryption_key":{"key":[79,54,18,38,133,249,141,170,137,106,163,19,181,3,185,112,214,219,38,143,94,91,231,70,76,190,36,222,162,7,121,87],"seed":[216,14,137,176,58,157,118,59,234,140,33,90,54,148,227,145,137,145,229,60,174,179,170,169,94,204,185,166,197,38,90,237]},"privacy_preimage":[154506351936219726,8681823448832561019,5143039451798493755,7663456240680657341,8770798044946808513],"unlock_key":[15576850314116203612,11000883362639892844,6082423871802091287,10149057265555220585,17806075860918111696],"seed":[12184773253696362769,7871559909312953421,6274311328951699531,8475403641542156896,16892759356993298775]}],[4096,{"receiver_identifier":1345681379867790294,"decryption_key":{"key":[156,30,129,182,242,118,154,35,200,158,172,241,126,35,96,126,17,101,235,10,207,48,67,40,146,160,190,170,93,117,242,42],"seed":[93,49,10,183,131,240,126,168,203,203,196,255,112,32,66,25,34,246,76,202,90,235,79,123,112,227,21,67,186,7,178,220]},"privacy_preimage":[4241757553656424794,11285689273644415582,7074568824318010500,10670185677383837662,4768155401846473576],"unlock_key":[8404358555156876671,11660012544449164001,3389282576847629864,5654429992919377988,17074250581958537675],"seed":[7778355330264861288,10967201848914548896,14636081467279311995,15247001707615007549,5050389500446245936]}],[32767,{"receiver_identifier":3783439891205707088,"decryption_key":{"key":[201,33,254,234,124,76,225,44,197,160,93,187,165,86,131,52,188,224,167,185,40,205,78,138,183,229,209,214,32,110,172,91],"seed":[159,81,55,234,92,152,167,5,115,176,40,105,42,239,52,81,6,51,236,101,178,158,58,191,61,162,155,175,146,104,166,199]},"privacy_preimage":[5329285513085370576,15417381536741326795,14328980918327036845,950512400958545829,11821730963736457587],"unlock_key":[2194281212432158346,14157892823925972113,6059267181542780069,7678302851212547965,12656055589550383753],"seed":[6796736216323740975,2696630545623589749,5452483991254704190,18062839443905637548,9423387729061483502]}],[65535,{"receiver_identifier":9811668228740958894,"decryption_key":{"key":[68,70,10,57,129,223,213,122,199,124,36,239,221,47,83,62,198,215,50,221,250,230,141,219,132,70,210,51,180,143,226,36],"seed":[126,151,143,174,158,81,94,130,101,101,209,120,105,158,88,47,181,39,82,201,82,196,241,42,226,234,210,165,162,48,22,227]},"privacy_preimage":[1255169973904000487,13053305603516563885,9706032528971708080,8473394876739135854,8341341419593938599],"unlock_key":[13242140641413128678,16591890723323192045,6888562562443922823,6068379561171930886,10224211500672236693],"seed":[17544079709103350546,968884705042225828,9354605114213694151,8515737674561592408,14122109919564875209]}]]
"#
            }
        }
    }
}
