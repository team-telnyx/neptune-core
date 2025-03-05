use std::fmt::Debug;
use std::fmt::Formatter;
use std::fmt::Result;

use crate::job_queue::triton_vm::TritonVmJobQueue;
use crate::models::blockchain::transaction::utxo::Utxo;

use super::tx_proving_capability::TxProvingCapability;
use super::wallet::address::SpendingKey;
use super::wallet::utxo_notification::UtxoNotificationMedium;

/// Custom trait capturing the closure for selecting UTXOs.
trait UtxoSelector: Fn(&Utxo) -> bool + Send + Sync + 'static {}
impl<T> UtxoSelector for T where T: Fn(&Utxo) -> bool + Send + Sync + 'static {}

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

/// Options and configuration settings for creating transactions
#[derive(Debug, Clone, Default)]
pub(crate) struct TxCreationConfig<'a> {
    change: Option<ChangeKeyAndMedium>,
    prover_capability: TxProvingCapability,
    triton_vm_job_queue: Option<&'a TritonVmJobQueue>,
    select_utxos: Option<DebuggableUtxoSelector>,
    track_selection: bool,
    record_details: bool,
}

impl<'a> TxCreationConfig<'a> {
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
        self.change = Some(change_key_and_medium);
        self
    }

    /// Enable change-recovery with the given key, and set the medium to
    /// `OnChain`.
    pub(crate) fn recover_change_on_chain(self, change_key: SpendingKey) -> Self {
        self.recover_change(change_key, UtxoNotificationMedium::OnChain)
    }

    /// Enable change-recovery with the given key, and set the medium to
    /// `OffChain`.
    pub(crate) fn recover_change_off_chain(self, change_key: SpendingKey) -> Self {
        self.recover_change(change_key, UtxoNotificationMedium::OffChain)
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
        F: Fn(&Utxo) -> bool + Send + Sync + 'static,
    {
        self.select_utxos = Some(DebuggableUtxoSelector(Box::new(selector)));
        self
    }

    /// Produce a [`TransactionDetails`] object along with the other artifacts.
    pub(crate) fn record_details(mut self) -> Self {
        self.record_details = true;
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

    pub(crate) fn details_are_recorded(&self) -> bool {
        self.record_details
    }

    pub(crate) fn selection_is_tracked(&self) -> bool {
        self.track_selection
    }

    /// Get the change key and notification medium, if any.
    pub(crate) fn change(&self) -> Option<ChangeKeyAndMedium> {
        self.change.clone()
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
}
