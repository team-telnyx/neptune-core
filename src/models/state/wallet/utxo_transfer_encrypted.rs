use anyhow::bail;
use anyhow::Result;
use bech32::ToBase32;
use serde::Deserialize;
use serde::Serialize;
use tasm_lib::triton_vm::prelude::BFieldElement;

use crate::config_models::network::Network;
use crate::models::blockchain::transaction::transaction_output::UtxoNotificationPayload;

use super::address::SpendingKey;

/// an encrypted wrapper for UtxoTransfer.
///
/// This type is intended to be serialized and actually transferred between
/// parties.
///
/// note: bech32m encoding of this type is considered standard and is
/// recommended over serde serialization.
///
/// the receiver_identifier enables the receiver to find the matching
/// `SpendingKey` in their wallet.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct UtxoTransferEncrypted {
    /// contains encrypted UtxoTransfer
    pub ciphertext: Vec<BFieldElement>,

    /// enables the receiver to find the matching `SpendingKey` in their wallet.
    pub receiver_identifier: BFieldElement,
}

impl UtxoTransferEncrypted {
    /// decrypts into a [UtxoTransfer]
    pub fn decrypt_with_spending_key(
        &self,
        spending_key: &SpendingKey,
    ) -> Result<UtxoNotificationPayload> {
        let (utxo, sender_randomness) = spending_key.decrypt(&self.ciphertext)?;

        Ok(UtxoNotificationPayload::new(utxo, sender_randomness))
    }

    /// encodes into a bech32m string for the given network
    pub fn to_bech32m(&self, network: Network) -> Result<String> {
        let hrp = Self::get_hrp(network);
        let payload = bincode::serialize(self)?;
        let variant = bech32::Variant::Bech32m;
        match bech32::encode(&hrp, payload.to_base32(), variant) {
            Ok(enc) => Ok(enc),
            Err(e) => bail!("Could not encode UtxoTransferEncrypted as bech32m because error: {e}"),
        }
    }

    /// decodes from a bech32m string and verifies it matches `network`
    pub fn from_bech32m(encoded: &str, network: Network) -> Result<Self> {
        let (hrp, data, variant) = bech32::decode(encoded)?;

        if variant != bech32::Variant::Bech32m {
            bail!("Can only decode bech32m addresses.");
        }

        if hrp != *Self::get_hrp(network) {
            bail!("Could not decode bech32m address because of invalid prefix");
        }

        let payload = Vec::<u8>::from_base32(&data)?;

        match bincode::deserialize(&payload) {
            Ok(ra) => Ok(ra),
            Err(e) => bail!("Could not decode bech32m because of error: {e}"),
        }
    }

    /// returns human readable prefix (hrp) of a utxo-transfer-encrypted, specific to `network`
    pub fn get_hrp(network: Network) -> String {
        format!("utxo{}", super::address::common::network_hrp_char(network))
    }
}
