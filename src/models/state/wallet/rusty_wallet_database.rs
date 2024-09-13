use twenty_first::math::tip5::Digest;

use super::monitored_utxo::MonitoredUtxo;
use crate::database::storage::storage_schema::traits::*;
use crate::database::storage::storage_schema::DbtSingleton;
use crate::database::storage::storage_schema::DbtVec;
use crate::database::storage::storage_schema::RustyKey;
use crate::database::storage::storage_schema::RustyValue;
use crate::database::storage::storage_schema::SimpleRustyStorage;
use crate::database::NeptuneLevelDb;
use crate::prelude::twenty_first;

pub struct RustyWalletDatabase {
    storage: SimpleRustyStorage,

    monitored_utxos: DbtVec<MonitoredUtxo>,

    // records which block the database is synced to
    sync_label: DbtSingleton<Digest>,

    // counts the number of output UTXOs generated by this wallet
    counter: DbtSingleton<u64>,
}

impl RustyWalletDatabase {
    pub async fn connect(db: NeptuneLevelDb<RustyKey, RustyValue>) -> Self {
        let mut storage = SimpleRustyStorage::new_with_callback(
            db,
            "RustyWalletDatabase-Schema",
            crate::LOG_LOCK_EVENT_CB,
        );

        let monitored_utxos_storage = storage
            .schema
            .new_vec::<MonitoredUtxo>("monitored_utxos")
            .await;
        let sync_label_storage = storage.schema.new_singleton::<Digest>("sync_label").await;
        let counter_storage = storage.schema.new_singleton::<u64>("counter").await;

        Self {
            storage,
            monitored_utxos: monitored_utxos_storage,
            sync_label: sync_label_storage,
            counter: counter_storage,
        }
    }

    /// get monitored_utxos.
    pub fn monitored_utxos(&self) -> &DbtVec<MonitoredUtxo> {
        &self.monitored_utxos
    }

    /// get mutable monitored_utxos.
    pub fn monitored_utxos_mut(&mut self) -> &mut DbtVec<MonitoredUtxo> {
        &mut self.monitored_utxos
    }

    /// Get the hash of the block to which this database is synced.
    pub async fn get_sync_label(&self) -> Digest {
        self.sync_label.get().await
    }

    pub async fn set_sync_label(&mut self, sync_label: Digest) {
        self.sync_label.set(sync_label).await;
    }

    pub async fn get_counter(&self) -> u64 {
        self.counter.get().await
    }

    pub async fn set_counter(&mut self, counter: u64) {
        self.counter.set(counter).await;
    }
}

impl StorageWriter for RustyWalletDatabase {
    async fn persist(&mut self) {
        self.storage.persist().await
    }
}
