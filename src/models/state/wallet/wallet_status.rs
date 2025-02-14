use std::fmt::Display;

use itertools::Itertools;
use serde::Deserialize;
use serde::Serialize;
use strum::EnumIter;

use crate::models::blockchain::transaction::utxo::Utxo;
use crate::models::blockchain::type_scripts::native_currency_amount::NativeCurrencyAmount;
use crate::models::proof_abstractions::timestamp::Timestamp;
use crate::util_types::mutator_set::ms_membership_proof::MsMembershipProof;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct WalletStatusElement {
    pub aocl_leaf_index: u64,
    pub utxo: Utxo,
}

impl WalletStatusElement {
    pub fn new(aocl_leaf_index: u64, utxo: Utxo) -> Self {
        Self {
            aocl_leaf_index,
            utxo,
        }
    }
}

impl Display for WalletStatusElement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let string: String = format!("({}, {:?})", self.aocl_leaf_index, self.utxo);
        write!(f, "{}", string)
    }
}

/// Represents a snapshot of monitored-utxos in the wallet at a given point in time.
///
/// monitored-utxos are those utxos that the wallet has a reason to track.
///
/// The utxos are divided into two primary groups:
///
/// 1. synced utxos are those which have a membership-proof for the current tip.
///
/// 2. unsynced utxos are those which have a membership-proof for a block other
///    than the current tip.  (unsynced_utxos are only used by unit tests.)
///
/// Each group is further divided into:
///
/// 1. unspent are those which have not been used as an input in a confirmed
///    block.
///
/// 2. spent are those which have been used as an input in a confirmed block.
///
/// For unspent utxos, a further distinction is made:
///
/// 1. available utxos are those which are spendable now.  (not timelocked)
///
/// 2. timelocked utxos are those which are not spendable until a certain time.
///    (timelocked)
///
/// note: WalletStatus is generated by WalletState::get_wallet_status().
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct WalletStatus {
    /// UTXOs that have a synced and valid membership proof
    pub synced_unspent: Vec<(WalletStatusElement, MsMembershipProof)>,

    /// UTXOs that have a synced membership proof but it is invalid (probably
    /// because it was spent)
    pub synced_spent: Vec<WalletStatusElement>,

    /// UTXOs that do not have a synced membership proof
    ///
    /// note: this field is presently only used by:
    ///  a) unit test(s)
    ///  b) indirectly the neptune-cli `wallet-status` command when
    ///     it json serializes `WalletStatus` to stdout.
    pub unsynced: Vec<WalletStatusElement>,
}

impl WalletStatus {
    /// synced, total balance (includes timelocked utxos)
    pub fn synced_unspent_total_amount(&self) -> NativeCurrencyAmount {
        self.synced_unspent
            .iter()
            .map(|(wse, _msmp)| &wse.utxo)
            .map(|utxo| utxo.get_native_currency_amount())
            .sum::<NativeCurrencyAmount>()
    }

    /// synced, available balance (excludes timelocked utxos)
    pub fn synced_unspent_available_amount(&self, timestamp: Timestamp) -> NativeCurrencyAmount {
        self.synced_unspent
            .iter()
            .map(|(wse, _msmp)| &wse.utxo)
            .filter(|utxo| utxo.can_spend_at(timestamp))
            .map(|utxo| utxo.get_native_currency_amount())
            .sum::<NativeCurrencyAmount>()
    }

    /// sum of synced, timelocked funds (only)
    pub fn synced_unspent_timelocked_amount(&self, timestamp: Timestamp) -> NativeCurrencyAmount {
        self.synced_unspent
            .iter()
            .map(|(wse, _msmp)| &wse.utxo)
            .filter(|utxo| utxo.is_timelocked_but_otherwise_spendable_at(timestamp))
            .map(|utxo| utxo.get_native_currency_amount())
            .sum::<NativeCurrencyAmount>()
    }
}

#[derive(
    Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq, Default, EnumIter, strum::Display,
)]
#[strum(serialize_all = "lowercase")]
pub enum WalletStatusExportFormat {
    #[default]
    Json,
    Table,
}

impl WalletStatusExportFormat {
    pub fn export(&self, wallet_status: &WalletStatus) -> String {
        match self {
            Self::Json => match serde_json::to_string_pretty(&wallet_status) {
                Ok(pretty_string) => pretty_string,
                Err(e) => format!("JSON format error: {e:?}"),
            },
            Self::Table => {
                fn row(wse: &WalletStatusElement) -> String {
                    let utxo = &wse.utxo;
                    let native_currency_amount = if utxo.has_native_currency() {
                        utxo.get_native_currency_amount().display_lossless()
                    } else {
                        "-".to_string()
                    };
                    let release_date = utxo
                        .release_date()
                        .map_or("-".to_string(), |t| t.standard_format());
                    format!(
                        "| {:>7} | {:>44} | {:^29} |",
                        wse.aocl_leaf_index, native_currency_amount, release_date
                    )
                }

                let header = format!(
                    "\
                    | aocl li | {:^44} | {:^29} |\n\
                    |:-------:|-{}:|:{}-|",
                    "native currency amount (coins)",
                    "release date",
                    (0..44).map(|_| "-").join(""),
                    (0..29).map(|_| "-").join("")
                );

                format!(
                    "\n\
                    **Synced Unspent**\n\
                    \n\
                    {header}\n\
                    {}\n\n\
                    Total:      {:>44} \n\
                    \n\
                    **Synced Spent**\n\
                    \n\
                    {header}\n\
                    {}\n\n\
                    Total:      {:>44} \n\
                    \n\
                    **Unsynced**\n\
                    \n\
                    {header}\n\
                    {}\n\n\
                    Total:      {:>44} \n",
                    wallet_status
                        .synced_unspent
                        .iter()
                        .map(|(wse, _)| wse)
                        .map(row)
                        .join("\n"),
                    wallet_status
                        .synced_unspent
                        .iter()
                        .map(|(wse, _)| wse.utxo.get_native_currency_amount())
                        .sum::<NativeCurrencyAmount>()
                        .display_lossless(),
                    wallet_status.synced_spent.iter().map(row).join("\n"),
                    wallet_status
                        .synced_spent
                        .iter()
                        .map(|wse| wse.utxo.get_native_currency_amount())
                        .sum::<NativeCurrencyAmount>()
                        .display_lossless(),
                    wallet_status.unsynced.iter().map(row).join("\n"),
                    wallet_status
                        .unsynced
                        .iter()
                        .map(|wse| wse.utxo.get_native_currency_amount())
                        .sum::<NativeCurrencyAmount>()
                        .display_lossless(),
                )
            }
        }
    }
}
