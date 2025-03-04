use std::collections::HashSet;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::fmt::Result;
use std::sync::OnceLock;

use crate::job_queue::triton_vm::TritonVmJobQueue;
use crate::models::blockchain::transaction::utxo::Utxo;

use super::transaction_details::TransactionDetails;
use super::tx_proving_capability::TxProvingCapability;
use super::wallet::address::SpendingKey;
use super::wallet::transaction_output::TxOutput;
use super::wallet::utxo_notification::UtxoNotificationMedium;
use super::wallet::wallet_state::StrongUtxoKey;

/// Custom trait capturing the closure for selecting UTXOs.
trait UtxoSelector: Fn(&Utxo) -> bool + Send + 'static {}
impl<T> UtxoSelector for T where T: Fn(&Utxo) -> bool + Send + 'static {}

/// Wrapper around the closure type for selecting UTXOs. Purpose: allow
/// `derive(Debug)` and `derive(Clone)` on structs that have this closure as a
/// field. (Note that these derive macros don't work for raw closure types.)
struct DebuggableUtxoSelector(Box<dyn UtxoSelector>);
impl Debug for DebuggableUtxoSelector {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "DebuggableUtxoSelector")
    }
}

impl Clone for DebuggableUtxoSelector {
    fn clone(&self) -> Self {
        panic!("Cloning not supported for DebuggableUtxoSelector");
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ChangeKeyAndMedium {
    pub(crate) key: SpendingKey,
    pub(crate) medium: UtxoNotificationMedium,
}

/// Options and configuration settings for initiating transactions
#[derive(Debug, Clone, Default)]
pub(crate) struct TxInitiationConfig<'a> {
    change: Option<(ChangeKeyAndMedium, OnceLock<TxOutput>)>,
    prover_capability: TxProvingCapability,
    triton_vm_job_queue: Option<&'a TritonVmJobQueue>,
    select_utxos: Option<DebuggableUtxoSelector>,
    track_selection: bool,
    selection: OnceLock<HashSet<StrongUtxoKey>>,
    transaction_details: OnceLock<TransactionDetails>,
}

impl<'a> TxInitiationConfig<'a> {
    /// Enable change-recovery and configure which key and notification medium
    /// to use for that purpose.
    pub(crate) fn recover_change(
        mut self,
        change_key: SpendingKey,
        notification_medium: UtxoNotificationMedium,
    ) -> Self {
        let change_key_and_medium = ChangeKeyAndMedium {
            key: change_key,
            medium: notification_medium,
        };
        self.change = Some((change_key_and_medium, OnceLock::default()));
        self
    }

    /// Configure the proving capacity.
    pub(crate) fn with_prover_capability(mut self, prover_capability: TxProvingCapability) -> Self {
        self.prover_capability = prover_capability;
        self
    }

    /// Configure which job queue to use.
    pub(crate) fn use_job_queue(mut self, job_queue: &'a TritonVmJobQueue) -> Self {
        self.triton_vm_job_queue = Some(job_queue);
        self
    }

    /// When selecting UTXOs, filter them through the given closure.
    pub(crate) fn select_utxos<F>(mut self, selector: F) -> Self
    where
        F: Fn(&Utxo) -> bool + Send + 'static,
    {
        self.select_utxos = Some(DebuggableUtxoSelector(Box::new(selector)));
        self
    }

    /// Enable selection-tracking.
    ///
    /// When enabled, the a hash set of `StrongUtxoKey`s is stored, indicating
    /// which UTXOs were selected for the transaction.
    pub(crate) fn track_selection(mut self) -> Self {
        self.track_selection = true;
        self
    }

    /// Get the change key and notification medium, if any.
    pub(crate) fn change(&self) -> Option<ChangeKeyAndMedium> {
        self.change
            .as_ref()
            .map(|(change_key_and_medium, _)| change_key_and_medium)
            .cloned()
    }

    /// Get the transaction proving capability.
    pub(crate) fn prover_capability(&self) -> TxProvingCapability {
        self.prover_capability
    }

    /// Get the job queue, if set.
    pub(crate) fn job_queue(&self) -> Option<&'a TritonVmJobQueue> {
        self.triton_vm_job_queue
    }

    /// Get the closure with which to filter out unsuitable UTXOs during UTXO
    /// selection.
    pub(crate) fn utxo_selector(&self) -> Option<&Box<dyn UtxoSelector>> {
        self.select_utxos.as_ref().map(|dus| &dus.0)
    }

    /// Get the set of strong UTXO keys of selected UTXOs, if selection-tracking
    /// is enabled.
    pub(crate) fn selected_utxos(&self) -> Option<&HashSet<StrongUtxoKey>> {
        self.selection.get()
    }

    /// Get the transaction details corresponding to the produced transaction.
    pub(crate) fn transaction_details(&self) -> Option<&TransactionDetails> {
        self.transaction_details.get()
    }

    /// Get the change output, if any.
    pub(crate) fn change_output(&self) -> Option<TxOutput> {
        self.change
            .as_ref()
            .and_then(|(_, output)| output.get())
            .cloned()
    }

    /// Set the change output, if change-recovery is enabled and if it is not
    /// already set. In case of failure, this function returns the given
    /// `TxOutput` object, wrapped in the `Err` variant.
    pub(crate) fn set_change_output(
        &self,
        output: TxOutput,
    ) -> std::result::Result<(), Box<TxOutput>> {
        let Some((_, change_output)) = &self.change else {
            return std::result::Result::Err(Box::new(output));
        };
        change_output.set(output).map_err(Box::new)
    }

    /// Set the selection of UTXOs that are used as inputs in this transaction.
    /// If the selection was already set, it will not be overwritten and this
    /// function returns the given argument, wrapped in the `Err` variant.
    pub(crate) fn set_selected_utxos(
        &self,
        selected_utxos: HashSet<StrongUtxoKey>,
    ) -> std::result::Result<(), HashSet<StrongUtxoKey>> {
        self.selection.set(selected_utxos)
    }

    /// Set the transaction details corresponding to the produced transaction.
    /// If the transaction details were already set, they will not be
    /// overwritten and this function returns the given argument, wrapped in the
    /// `Err` variant.
    pub(crate) fn set_transaction_details(
        &self,
        transaction_details: TransactionDetails,
    ) -> std::result::Result<(), Box<TransactionDetails>> {
        self.transaction_details
            .set(transaction_details)
            .map_err(Box::new)
    }
}
