//! provides an abstraction over key and address types.

use anyhow::bail;
use anyhow::Result;
#[cfg(any(test, feature = "arbitrary-impls"))]
use arbitrary::Arbitrary;
use serde::Deserialize;
use serde::Serialize;
use tasm_lib::triton_vm::prelude::Digest;
use tracing::warn;

use super::common;
use super::generation_address;
use super::hash_lock_key;
use super::symmetric_key;
use crate::config_models::network::Network;
use crate::models::blockchain::transaction::lock_script::LockScript;
use crate::models::blockchain::transaction::lock_script::LockScriptAndWitness;
use crate::models::blockchain::transaction::transaction_kernel::TransactionKernel;
use crate::models::blockchain::transaction::utxo::Utxo;
use crate::models::blockchain::transaction::PublicAnnouncement;
use crate::models::state::wallet::incoming_utxo::IncomingUtxo;
use crate::models::state::wallet::utxo_notification::UtxoNotificationPayload;
use crate::BFieldElement;

// note: assigning the flags to `KeyType` variants as discriminants has bonus
// that we get a compiler verification that values do not conflict.  which is
// nice since they are (presently) defined in separate files.
//
// anyway it is a desirable property that KeyType variants match the values
// actually stored in PublicAnnouncement.

/// Enumerates available cryptographic key implementations for sending funds.
///
/// In most (but not all) cases there is a matching address.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum KeyType {
    /// To unlock, prove knowledge of the preimage.
    RawHashLock = hash_lock_key::RAW_HASH_LOCK_KEY_FLAG_U8,

    /// [generation_address] built on [crate::prelude::twenty_first::math::lattice::kem]
    ///
    /// wraps a symmetric key built on aes-256-gcm
    Generation = generation_address::GENERATION_FLAG_U8,

    /// [symmetric_key] built on aes-256-gcm
    Symmetric = symmetric_key::SYMMETRIC_KEY_FLAG_U8,
}

impl std::fmt::Display for KeyType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RawHashLock => write!(f, "Raw Hash Lock"),
            Self::Generation => write!(f, "Generation"),
            Self::Symmetric => write!(f, "Symmetric"),
        }
    }
}

impl From<&ReceivingAddress> for KeyType {
    fn from(addr: &ReceivingAddress) -> Self {
        match addr {
            ReceivingAddress::Generation(_) => Self::Generation,
            ReceivingAddress::Symmetric(_) => Self::Symmetric,
        }
    }
}

impl From<&SpendingKey> for KeyType {
    fn from(addr: &SpendingKey) -> Self {
        match addr {
            SpendingKey::Generation(_) => Self::Generation,
            SpendingKey::Symmetric(_) => Self::Symmetric,
            SpendingKey::RawHashLock { .. } => Self::RawHashLock,
        }
    }
}

impl From<KeyType> for BFieldElement {
    fn from(key_type: KeyType) -> Self {
        (key_type as u8).into()
    }
}

impl TryFrom<&PublicAnnouncement> for KeyType {
    type Error = anyhow::Error;

    fn try_from(pa: &PublicAnnouncement) -> Result<Self> {
        match common::key_type_from_public_announcement(pa) {
            Ok(kt) if kt == Self::Generation.into() => Ok(Self::Generation),
            Ok(kt) if kt == Self::Symmetric.into() => Ok(Self::Symmetric),
            _ => bail!("encountered PublicAnnouncement of unknown type"),
        }
    }
}

impl KeyType {
    /// returns all available `KeyType`
    pub fn all_types() -> Vec<KeyType> {
        vec![Self::RawHashLock, Self::Generation, Self::Symmetric]
    }

    /// Returns only those key types that can receive UTXOs, i.e. the key types
    /// that other people can send to and that have an associated address.
    #[cfg(test)]
    pub(crate) fn all_types_for_receiving() -> Vec<KeyType> {
        vec![Self::Generation, Self::Symmetric]
    }
}

/// Represents any type of Neptune receiving Address.
///
/// This enum provides an abstraction API for Address types, so that
/// a method or struct may simply accept a `ReceivingAddress` and be
/// forward-compatible with new types of Address as they are implemented.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(any(test, feature = "arbitrary-impls"), derive(Arbitrary))]
pub enum ReceivingAddress {
    /// a [generation_address]
    Generation(Box<generation_address::GenerationReceivingAddress>),

    /// a [symmetric_key] acting as an address.
    Symmetric(symmetric_key::SymmetricKey),
}

impl From<generation_address::GenerationReceivingAddress> for ReceivingAddress {
    fn from(a: generation_address::GenerationReceivingAddress) -> Self {
        Self::Generation(Box::new(a))
    }
}

impl From<&generation_address::GenerationReceivingAddress> for ReceivingAddress {
    fn from(a: &generation_address::GenerationReceivingAddress) -> Self {
        Self::Generation(Box::new(*a))
    }
}

impl From<symmetric_key::SymmetricKey> for ReceivingAddress {
    fn from(k: symmetric_key::SymmetricKey) -> Self {
        Self::Symmetric(k)
    }
}

impl From<&symmetric_key::SymmetricKey> for ReceivingAddress {
    fn from(k: &symmetric_key::SymmetricKey) -> Self {
        Self::Symmetric(*k)
    }
}

impl TryFrom<ReceivingAddress> for generation_address::GenerationReceivingAddress {
    type Error = anyhow::Error;

    fn try_from(a: ReceivingAddress) -> Result<Self> {
        match a {
            ReceivingAddress::Generation(a) => Ok(*a),
            _ => bail!("not a generation address"),
        }
    }
}

impl ReceivingAddress {
    /// returns `receiver_identifier`
    pub fn receiver_identifier(&self) -> BFieldElement {
        match self {
            Self::Generation(a) => a.receiver_identifier(),
            Self::Symmetric(a) => a.receiver_identifier(),
        }
    }

    /// generates a [PublicAnnouncement] for an output Utxo
    ///
    /// The public announcement contains a [`Vec<BFieldElement>`] with fields:
    ///   0    --> type flag.  (flag of key type)
    ///   1    --> receiver_identifier  (fingerprint derived from seed)
    ///   2..n --> ciphertext (encrypted utxo + sender_randomness)
    ///
    /// Fields |0,1| enable the receiver to determine the ciphertext
    /// is intended for them and decryption should be attempted.
    pub(crate) fn generate_public_announcement(
        &self,
        utxo_notification_payload: UtxoNotificationPayload,
    ) -> PublicAnnouncement {
        match self {
            ReceivingAddress::Generation(generation_receiving_address) => {
                generation_receiving_address
                    .generate_public_announcement(&utxo_notification_payload)
            }
            ReceivingAddress::Symmetric(symmetric_key) => {
                symmetric_key.generate_public_announcement(&utxo_notification_payload)
            }
        }
    }

    pub(crate) fn private_notification(
        &self,
        utxo_notification_payload: UtxoNotificationPayload,
        network: Network,
    ) -> String {
        match self {
            ReceivingAddress::Generation(generation_receiving_address) => {
                generation_receiving_address
                    .private_utxo_notification(&utxo_notification_payload, network)
            }
            ReceivingAddress::Symmetric(symmetric_key) => {
                symmetric_key.private_utxo_notification(&utxo_notification_payload, network)
            }
        }
    }

    /// returns the `spending_lock`
    pub fn spending_lock(&self) -> Digest {
        match self {
            Self::Generation(a) => a.spending_lock(),
            Self::Symmetric(k) => k.lock_after_image(),
        }
    }

    /// returns a privacy digest which corresponds to the.privacy_preimage(),
    /// of the matching [SpendingKey]
    pub fn privacy_digest(&self) -> Digest {
        match self {
            Self::Generation(a) => a.privacy_digest(),
            Self::Symmetric(k) => k.privacy_digest(),
        }
    }

    /// encrypts a [Utxo] and `sender_randomness` secret for purpose of transferring to payment recipient
    #[cfg(test)]
    pub(crate) fn encrypt(
        &self,
        utxo_notification_payload: &UtxoNotificationPayload,
    ) -> Vec<BFieldElement> {
        match self {
            Self::Generation(a) => a.encrypt(utxo_notification_payload),
            Self::Symmetric(a) => a.encrypt(utxo_notification_payload),
        }
    }

    /// encodes this address as bech32m
    ///
    /// For any key-type, the resulting bech32m can be provided as input to
    /// Self::from_bech32m() and will generate the original ReceivingAddress.
    ///
    /// Security: for key-type==Symmetric the resulting string exposes
    /// the secret-key.  As such, great care must be taken and it should
    /// never be used for display purposes.
    ///
    /// For most uses, prefer [Self::to_display_bech32m()] instead.
    pub fn to_bech32m(&self, network: Network) -> Result<String> {
        match self {
            Self::Generation(k) => k.to_bech32m(network),
            Self::Symmetric(k) => k.to_bech32m(network),
        }
    }

    /// returns an abbreviated bech32m encoded address.
    ///
    /// This method *may* reveal secret-key information for some key-types.  For
    /// general display purposes, prefer
    /// [Self::to_display_bech32m_abbreviated()].
    ///
    /// The idea is that this suitable for human recognition purposes
    ///
    /// ```text
    /// format:  <hrp><start>...<end>
    ///
    ///   [4 or 6] human readable prefix. 4 for symmetric-key, 6 for generation.
    ///   12 start of address.
    ///   12 end of address.
    /// ```
    pub fn to_bech32m_abbreviated(&self, network: Network) -> Result<String> {
        self.bech32m_abbreviate(self.to_bech32m(network)?, network)
    }

    /// returns a bech32m string suitable for display purposes.
    ///
    /// This method does not reveal secret-key information for any key-type.
    ///
    /// The resulting bech32m string is not guaranteed to result in the same
    /// [ReceivingAddress] if provided as input to [Self::from_bech32m()].  For
    /// that, [Self::to_bech32m()] should be used instead.
    ///
    /// For [Self::Generation] keys, this is equivalent to calling [Self::to_bech32m()].
    /// For [Self::Symmetric] keys, this returns the privacy_preimage hash bech32m encoded
    /// instead of the key itself.
    pub fn to_display_bech32m(&self, network: Network) -> anyhow::Result<String> {
        match self {
            Self::Generation(k) => k.to_bech32m(network),
            Self::Symmetric(k) => k.to_display_bech32m(network),
        }
    }

    /// returns an abbreviated address suitable for display purposes.
    ///
    /// This method does not reveal secret-key information for any key-type.
    ///
    /// The idea is that this suitable for human recognition purposes
    ///
    /// ```text
    /// format:  <hrp><start>...<end>
    ///
    ///   [4 or 6] human readable prefix. 4 for symmetric-key, 6 for generation.
    ///   12 start of address.
    ///   12 end of address.
    /// ```
    pub fn to_display_bech32m_abbreviated(&self, network: Network) -> Result<String> {
        self.bech32m_abbreviate(self.to_display_bech32m(network)?, network)
    }

    fn bech32m_abbreviate(&self, bech32m: String, network: Network) -> Result<String> {
        let first_len = self.get_hrp(network).len() + 12usize;
        let last_len = 12usize;

        assert!(bech32m.len() > first_len + last_len);

        let (first, _) = bech32m.split_at(first_len);
        let (_, last) = bech32m.split_at(bech32m.len() - last_len);

        Ok(format!("{}...{}", first, last))
    }

    /// parses an address from its bech32m encoding
    pub fn from_bech32m(encoded: &str, network: Network) -> Result<Self> {
        if let Ok(addr) =
            generation_address::GenerationReceivingAddress::from_bech32m(encoded, network)
        {
            return Ok(addr.into());
        }

        let key = symmetric_key::SymmetricKey::from_bech32m(encoded, network)?;
        Ok(key.into())

        // when future addr types are supported, we would attempt each type in
        // turn.
    }

    /// returns human-readable-prefix (hrp) for a given network
    pub fn get_hrp(&self, network: Network) -> String {
        match self {
            Self::Generation(_) => generation_address::GenerationReceivingAddress::get_hrp(network),
            Self::Symmetric(_) => symmetric_key::SymmetricKey::get_hrp(network).to_string(),
        }
    }

    /// generates a lock script from the spending lock.
    ///
    /// Satisfaction of this lock script establishes the UTXO owner's assent to
    /// the transaction.
    pub fn lock_script(&self) -> LockScript {
        match self {
            Self::Generation(k) => k.lock_script(),
            Self::Symmetric(k) => k.lock_script(),
        }
    }

    /// returns true if the [PublicAnnouncement] has a type-flag that matches the type of this address.
    pub fn matches_public_announcement_key_type(&self, pa: &PublicAnnouncement) -> bool {
        matches!(KeyType::try_from(pa), Ok(kt) if kt == KeyType::from(self))
    }
}

/// Represents cryptographic data necessary for spending funds (or, more
/// specifically, for unlocking UTXOs).
///
/// This enum provides an abstraction API for spending key types, so that a
/// method or struct may simply accept a `SpendingKey` and be
/// forward-compatible with new types of spending key as they are implemented.
///
/// Note that not all spending keys have associated receiving addresses. In
/// particular, the `HashLock` variant has no associated address.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpendingKey {
    RawHashLock(hash_lock_key::HashLockKey),

    /// a key from [generation_address]
    Generation(generation_address::GenerationSpendingKey),

    /// a [symmetric_key]
    Symmetric(symmetric_key::SymmetricKey),
}

impl std::hash::Hash for SpendingKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::hash::Hash::hash(&self.privacy_preimage(), state)
    }
}

impl From<generation_address::GenerationSpendingKey> for SpendingKey {
    fn from(key: generation_address::GenerationSpendingKey) -> Self {
        Self::Generation(key)
    }
}

impl From<symmetric_key::SymmetricKey> for SpendingKey {
    fn from(key: symmetric_key::SymmetricKey) -> Self {
        Self::Symmetric(key)
    }
}

impl SpendingKey {
    /// returns the address that corresponds to this spending key.
    pub fn to_address(&self) -> Option<ReceivingAddress> {
        match self {
            Self::Generation(k) => Some(k.to_address().into()),
            Self::Symmetric(k) => Some((*k).into()),
            Self::RawHashLock(_) => None,
        }
    }

    /// Return the lock script and its witness
    pub(crate) fn lock_script_and_witness(&self) -> LockScriptAndWitness {
        match self {
            SpendingKey::Generation(generation_spending_key) => {
                generation_spending_key.lock_script_and_witness()
            }
            SpendingKey::Symmetric(symmetric_key) => symmetric_key.lock_script_and_witness(),
            SpendingKey::RawHashLock(raw_hash_lock) => raw_hash_lock.lock_script_and_witness(),
        }
    }

    pub(crate) fn lock_script(&self) -> LockScript {
        LockScript {
            program: self.lock_script_and_witness().program,
        }
    }

    pub(crate) fn lock_script_hash(&self) -> Digest {
        self.lock_script().hash()
    }

    /// Return the privacy preimage if this spending key has a corresponding
    /// receiving address.
    ///
    /// note: The hash of the preimage is available in the receiving address
    /// as the privacy_digest
    pub fn privacy_preimage(&self) -> Option<Digest> {
        match self {
            Self::Generation(k) => Some(k.privacy_preimage()),
            Self::Symmetric(k) => Some(k.privacy_preimage()),
            Self::RawHashLock { .. } => None,
        }
    }

    /// Return the receiver_identifier if this spending key has a corresponding
    /// receiving address.
    ///
    /// The receiver identifier is a public (=readably by anyone) fingerprint of
    /// the beneficiary's receiving address. It is used to efficiently scan
    /// incoming blocks for new UTXOs that are destined for this key.
    ///
    /// However, the fingerprint can *also* be used to link different payments
    /// to the same address as payments to the same person. Users who want to
    /// avoid this linkability must generate a new address. Down the line we
    /// expect to support address formats that do not come with fingerprints,
    /// and users can enable them for better privacy in exchange for the
    /// increased workload associated with detecting incoming UTXOs.
    pub fn receiver_identifier(&self) -> Option<BFieldElement> {
        match self {
            Self::Generation(k) => Some(k.receiver_identifier()),
            Self::Symmetric(k) => Some(k.receiver_identifier()),
            Self::RawHashLock { .. } => None,
        }
    }

    /// Decrypt a slice of BFieldElement into a [Utxo] and [Digest] representing
    /// `sender_randomness`, if this spending key has a corresponding receiving
    /// address.
    ///
    /// # Return Value
    ///
    ///  - `None` if this spending key has no associated receiving address.
    ///  - `Some(Err(..))` if decryption failed.
    ///  - `Some(Ok(..))` if decryption succeeds.
    pub fn decrypt(&self, ciphertext_bfes: &[BFieldElement]) -> Option<Result<(Utxo, Digest)>> {
        let result = match self {
            Self::Generation(k) => k.decrypt(ciphertext_bfes),
            Self::Symmetric(k) => k.decrypt(ciphertext_bfes).map_err(anyhow::Error::new),
            Self::RawHashLock { .. } => {
                return None;
            }
        };
        Some(result)
    }

    /// Scans all public announcements in a `Transaction` and return all
    /// UTXOs that are recognized by this spending key.
    ///
    /// Note that a single `Transaction` may represent an entire block.
    ///
    /// # Side Effects
    ///
    ///  - Logs a warning for any announcement targeted at this key that cannot
    ///    be decrypted.
    pub(crate) fn scan_for_announced_utxos(
        &self,
        tx_kernel: &TransactionKernel,
    ) -> Vec<IncomingUtxo> {
        // pre-compute some fields, and early-abort if key cannot receive.
        let Some(receiver_identifier) = self.receiver_identifier() else {
            return vec![];
        };
        let Some(receiver_preimage) = self.privacy_preimage() else {
            return vec![];
        };

        // for all public announcements
        tx_kernel
            .public_announcements
            .iter()

            // ... that are marked as encrypted to our key type
            .filter(|pa| self.matches_public_announcement_key_type(pa))

            // ... that match the receiver_id of this key
            .filter(move |pa| {
                matches!(common::receiver_identifier_from_public_announcement(pa), Ok(r) if r == receiver_identifier)
            })

            // ... that have a ciphertext field
            .filter_map(|pa| self.ok_warn(common::ciphertext_from_public_announcement(pa)))

            // ... which can be decrypted with this key
            .filter_map(|c| self.ok_warn(self.decrypt(&c).expect("non-hash-lock key should have decryption option")))

            // ... map to IncomingUtxo
            .map(move |(utxo, sender_randomness)| {
                // and join those with the receiver digest to get a commitment
                // Note: the commitment is computed in the same way as in the mutator set.
                IncomingUtxo {
                    utxo,
                    sender_randomness,
                    receiver_preimage,
                }
            }).collect()
    }

    /// converts a result into an Option and logs a warning on any error
    fn ok_warn<T>(&self, result: Result<T>) -> Option<T> {
        let Some(receiver_identifier) = self.receiver_identifier() else {
            panic!("Cannot call `ok_warn` unless the spending key has an associated address.");
        };
        match result {
            Ok(v) => Some(v),
            Err(e) => {
                warn!("possible loss of funds! skipping public announcement for {:?} key with receiver_identifier: {}.  error: {}", KeyType::from(self), receiver_identifier, e.to_string());
                None
            }
        }
    }

    /// returns true if the [PublicAnnouncement] has a type-flag that matches the type of this key
    fn matches_public_announcement_key_type(&self, pa: &PublicAnnouncement) -> bool {
        matches!(KeyType::try_from(pa), Ok(kt) if kt == KeyType::from(self))
    }
}

#[cfg(test)]
mod test {
    use generation_address::GenerationReceivingAddress;
    use generation_address::GenerationSpendingKey;
    use proptest_arbitrary_interop::arb;
    use rand::random;
    use rand::Rng;
    use symmetric_key::SymmetricKey;
    use test_strategy::proptest;

    use super::*;
    use crate::models::blockchain::type_scripts::native_currency_amount::NativeCurrencyAmount;
    use crate::tests::shared::make_mock_transaction;

    /// tests scanning for announced utxos with a symmetric key
    #[proptest]
    fn scan_for_announced_utxos_symmetric(#[strategy(arb())] seed: Digest) {
        worker::scan_for_announced_utxos(SymmetricKey::from_seed(seed).into())
    }

    /// tests scanning for announced utxos with an asymmetric (generation) key
    #[proptest]
    fn scan_for_announced_utxos_generation(#[strategy(arb())] seed: Digest) {
        worker::scan_for_announced_utxos(GenerationSpendingKey::derive_from_seed(seed).into())
    }

    /// tests encrypting and decrypting with a symmetric key
    #[proptest]
    fn test_encrypt_decrypt_symmetric(#[strategy(arb())] seed: Digest) {
        worker::test_encrypt_decrypt(SymmetricKey::from_seed(seed).into())
    }

    /// tests encrypting and decrypting with an asymmetric (generation) key
    #[proptest]
    fn test_encrypt_decrypt_generation(#[strategy(arb())] seed: Digest) {
        worker::test_encrypt_decrypt(GenerationSpendingKey::derive_from_seed(seed).into())
    }

    /// tests keygen, sign, and verify with a symmetric key
    #[proptest]
    fn test_keygen_sign_verify_symmetric(#[strategy(arb())] seed: Digest) {
        worker::test_keypair_validity(
            SymmetricKey::from_seed(seed).into(),
            SymmetricKey::from_seed(seed).into(),
        );
    }

    /// tests keygen, sign, and verify with an asymmetric (generation) key
    #[proptest]
    fn test_keygen_sign_verify_generation(#[strategy(arb())] seed: Digest) {
        worker::test_keypair_validity(
            GenerationSpendingKey::derive_from_seed(seed).into(),
            GenerationReceivingAddress::derive_from_seed(seed).into(),
        );
    }

    /// tests bech32m serialize, deserialize with a symmetric key
    #[proptest]
    fn test_bech32m_conversion_symmetric(#[strategy(arb())] seed: Digest) {
        worker::test_bech32m_conversion(SymmetricKey::from_seed(seed).into());
    }

    /// tests bech32m serialize, deserialize with an asymmetric (generation) key
    #[proptest]
    fn test_bech32m_conversion_generation(#[strategy(arb())] seed: Digest) {
        worker::test_bech32m_conversion(GenerationReceivingAddress::derive_from_seed(seed).into());
    }

    mod worker {
        use super::*;
        use crate::models::blockchain::transaction::transaction_kernel::TransactionKernelModifier;
        use crate::prelude::twenty_first::prelude::Tip5;
        use crate::util_types::mutator_set::commit;

        /// this tests the generate_public_announcement() and
        /// scan_for_announced_utxos() methods with a [SpendingKey]
        ///
        /// a PublicAnnouncement is created with generate_public_announcement() and
        /// added to a Tx.  It is then found by scanning for announced_utoxs.  Then
        /// we verify that the data matches the original/expected values.
        pub fn scan_for_announced_utxos(key: SpendingKey) {
            // 1. generate a utxo with amount = 10
            let utxo = Utxo::new_native_currency(
                key.to_address().unwrap().lock_script(),
                NativeCurrencyAmount::coins(10),
            );

            // 2. generate sender randomness
            let sender_randomness: Digest = random();

            // 3. create an addition record to verify against later.
            let expected_addition_record = commit(
                Tip5::hash(&utxo),
                sender_randomness,
                key.to_address().unwrap().privacy_digest(),
            );

            // 4. create a mock tx with no inputs or outputs
            let mut mock_tx = make_mock_transaction(vec![], vec![]);

            // 5. verify that no announced utxos exist for this key
            assert!(key.scan_for_announced_utxos(&mock_tx.kernel).is_empty());

            // 6. generate a public announcement for this address
            let utxo_notification_payload =
                UtxoNotificationPayload::new(utxo.clone(), sender_randomness);
            let public_announcement = key
                .to_address()
                .unwrap()
                .generate_public_announcement(utxo_notification_payload);

            // 7. verify that the public_announcement is marked as our key type.
            assert!(key.matches_public_announcement_key_type(&public_announcement));

            // 8. add the public announcement to the mock tx.
            let mut new_public_announcements = mock_tx.kernel.public_announcements.clone();
            new_public_announcements.push(public_announcement);

            mock_tx.kernel = TransactionKernelModifier::default()
                .public_announcements(new_public_announcements)
                .modify(mock_tx.kernel);

            // 9. scan tx public announcements for announced utxos
            let announced_utxos = key.scan_for_announced_utxos(&mock_tx.kernel);

            // 10. verify there is exactly 1 announced_utxo and obtain it.
            assert_eq!(1, announced_utxos.len());
            let announced_utxo = announced_utxos.into_iter().next().unwrap();

            // 11. verify each field of the announced_utxo matches original values.
            assert_eq!(utxo, announced_utxo.utxo);
            assert_eq!(expected_addition_record, announced_utxo.addition_record());
            assert_eq!(sender_randomness, announced_utxo.sender_randomness);
            assert_eq!(
                key.privacy_preimage().unwrap(),
                announced_utxo.receiver_preimage
            );
        }

        /// This tests encrypting and decrypting with a [SpendingKey]
        pub fn test_encrypt_decrypt(key: SpendingKey) {
            let mut rng = rand::rng();

            // 1. create utxo with random amount
            let amount = NativeCurrencyAmount::coins(rng.random_range(0..42000000));
            let utxo = Utxo::new_native_currency(key.to_address().unwrap().lock_script(), amount);

            // 2. generate sender randomness
            let sender_randomness: Digest = random();

            // 3. encrypt secrets (utxo, sender_randomness)
            let notification_payload =
                UtxoNotificationPayload::new(utxo.clone(), sender_randomness);
            let ciphertext = key.to_address().unwrap().encrypt(&notification_payload);
            println!("ciphertext.get_size() = {}", ciphertext.len() * 8);

            // 4. decrypt secrets
            let (utxo_again, sender_randomness_again) = key.decrypt(&ciphertext).unwrap().unwrap();

            // 5. verify that decrypted secrets match original secrets
            assert_eq!(utxo, utxo_again);
            assert_eq!(sender_randomness, sender_randomness_again);
        }

        /// tests key generation, signing, and decrypting with a [SpendingKey]
        ///
        /// note: key generation is performed by the caller. Both the
        /// spending_key and receiving_address must be independently derived from
        /// the same seed.
        pub fn test_keypair_validity(
            spending_key: SpendingKey,
            receiving_address: ReceivingAddress,
        ) {
            // 1. prepare a (random) message and witness data.
            let msg: Digest = random();
            let l_and_s = spending_key.lock_script_and_witness();

            // 2. perform proof verification
            assert!(l_and_s.halts_gracefully(msg.values().to_vec().into()));

            // 3. convert spending key to an address.
            let receiving_address_again = spending_key.to_address().unwrap();

            // 4. verify that both address match.
            assert_eq!(receiving_address, receiving_address_again);
        }

        /// tests bech32m serialize, deserialize for [ReceivingAddress]
        pub fn test_bech32m_conversion(receiving_address: ReceivingAddress) {
            // 1. serialize address to bech32m
            let encoded = receiving_address.to_bech32m(Network::Testnet).unwrap();

            // 2. deserialize bech32m back into an address
            let receiving_address_again =
                ReceivingAddress::from_bech32m(&encoded, Network::Testnet).unwrap();

            // 3. verify both addresses match
            assert_eq!(receiving_address, receiving_address_again);
        }
    }
}
