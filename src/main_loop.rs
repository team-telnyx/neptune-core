use crate::connect_to_peers::{answer_peer, call_peer_wrapper};
use crate::database::leveldb::LevelDB;
use crate::models::blockchain::block::block_header::BlockHeader;
use crate::models::blockchain::block::block_height::BlockHeight;
use crate::models::database::BlockDatabases;
use crate::models::peer::{PeerInfo, PeerSynchronizationState};
use crate::models::state::State;
use anyhow::Result;
use rand::prelude::IteratorRandom;
use rand::thread_rng;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::{Duration, SystemTime};
use tokio::net::TcpListener;
use tokio::sync::{broadcast, mpsc, watch};
use tokio::{select, time};
use tracing::{debug, error, info, warn};

use crate::models::channel::{MainToMiner, MainToPeerThread, MinerToMain, PeerThreadToMain};
use crate::models::peer::HandshakeData;

const PEER_DISCOVERY_INTERVAL_IN_SECONDS: u64 = 10;
const POTENTIAL_PEER_MAX_COUNT_AS_A_FACTOR_OF_MAX_PEERS: usize = 20;

struct SynchronizationState {
    peer_sync_states: HashMap<SocketAddr, PeerSynchronizationState>,
    last_sync_request: Option<(SystemTime, BlockHeight)>,
}

impl SynchronizationState {
    fn default() -> Self {
        Self {
            peer_sync_states: HashMap::new(),
            last_sync_request: None,
        }
    }
}

struct PotentialPeerInfo {
    reported: SystemTime,
    reported_by: SocketAddr,
    instance_id: u128,
    distance: u8,
}

impl PotentialPeerInfo {
    fn new(reported_by: SocketAddr, instance_id: u128, distance: u8) -> Self {
        Self {
            reported: SystemTime::now(),
            reported_by,
            instance_id,
            distance,
        }
    }
}

struct PotentialPeersState {
    potential_peers: HashMap<SocketAddr, PotentialPeerInfo>,
}

impl PotentialPeersState {
    fn default() -> Self {
        Self {
            potential_peers: HashMap::new(),
        }
    }

    fn add(
        &mut self,
        reported_by: SocketAddr,
        potential_peer: (SocketAddr, u128),
        max_peers: usize,
        distance: u8,
    ) {
        let potential_peer_socket_address = potential_peer.0;
        let potential_peer_instance_id = potential_peer.1;

        if self
            .potential_peers
            .contains_key(&potential_peer_socket_address)
        {
            return;
        }

        // If this data structure is full, remove a random entry. Then add this.
        if self.potential_peers.len()
            > max_peers as usize * POTENTIAL_PEER_MAX_COUNT_AS_A_FACTOR_OF_MAX_PEERS
        {
            let mut rng = rand::thread_rng();
            let random_potential_peer = self
                .potential_peers
                .keys()
                .choose(&mut rng)
                .unwrap()
                .to_owned();
            self.potential_peers.remove(&random_potential_peer);
        }

        let insert_value =
            PotentialPeerInfo::new(reported_by, potential_peer_instance_id, distance);
        self.potential_peers
            .insert(potential_peer_socket_address, insert_value);
    }

    /// Return a random peer from the potential peer list that we aren't connected to
    /// and that isn't our own address. Returns (socket address, peer distance)
    fn get_distant_candidate(
        &self,
        // connected_clients: &[(SocketAddr, u128)],
        connected_clients: &[PeerInfo],
        own_listen_socket: Option<SocketAddr>,
        own_instance_id: u128,
    ) -> Option<(SocketAddr, u8)> {
        let peers_instance_ids: Vec<u128> =
            connected_clients.iter().map(|x| x.instance_id).collect();
        let peers_listen_addresses: Vec<SocketAddr> = connected_clients
            .into_iter()
            .map(|x| x.address_for_incoming_connections)
            .filter_map(|x| x)
            .collect();

        // Find the appropriate candidates
        let not_connected_peers = self
            .potential_peers
            .iter()
            // Prevent connecting to self
            .filter(|pp| pp.1.instance_id != own_instance_id)
            .filter(|pp| own_listen_socket.is_some() && *pp.0 != own_listen_socket.unwrap())
            // Prevent connecting to peer we already are connected to
            .filter(|potential_peer| !peers_instance_ids.contains(&potential_peer.1.instance_id))
            .filter(|potential_peer| !peers_listen_addresses.contains(&potential_peer.0))
            .collect::<Vec<_>>();

        // Get the candidate list with the highest distance
        let max_distance_candidates = not_connected_peers.iter().max_by_key(|pp| pp.1.distance);

        // Pick a random candidate from the appropriate candidates
        let mut rng = rand::thread_rng();
        max_distance_candidates
            .iter()
            .choose(&mut rng)
            .map(|x| (x.0.to_owned(), x.1.distance))
    }
}

async fn handle_miner_thread_message(
    msg: MinerToMain,
    main_to_peer_broadcast_tx: &broadcast::Sender<MainToPeerThread>,
    state: State,
) -> Result<()> {
    match msg {
        MinerToMain::NewBlock(block) => {
            // When receiving a block from the miner thread, we assume it is valid
            // and we assume it is the longest chain even though we could have received
            // a block from a peer thread before this event is triggered.
            // info!("Miner found new block: {}", block.height);
            info!("Miner found new block: {}", block.header.height);
            main_to_peer_broadcast_tx
                .send(MainToPeerThread::BlockFromMiner(block.clone()))
                .expect(
                    "Peer handler broadcast channel prematurely closed. This should never happen.",
                );

            // Store block in database
            state.update_latest_block(block).await?;
        }
    }

    Ok(())
}

fn enter_sync_mode(
    own_block_tip_header: BlockHeader,
    peer_synchronization_state: PeerSynchronizationState,
    max_number_of_blocks_before_syncing: usize,
) -> bool {
    own_block_tip_header.proof_of_work_family < peer_synchronization_state.claimed_max_pow_family
        && peer_synchronization_state.claimed_max_height - own_block_tip_header.height
            > max_number_of_blocks_before_syncing as i128
}

fn stay_in_sync_mode(
    own_block_tip_header: BlockHeader,
    sync_state: &SynchronizationState,
    max_number_of_blocks_before_syncing: usize,
) -> bool {
    let max_claimed_pow = sync_state
        .peer_sync_states
        .values()
        .max_by_key(|x| x.claimed_max_pow_family);
    match max_claimed_pow {
        None => false, // we lost all connections. Can't sync.
        Some(max_claim) => {
            own_block_tip_header.proof_of_work_family < max_claim.claimed_max_pow_family
                && max_claim.claimed_max_height - own_block_tip_header.height
                    > max_number_of_blocks_before_syncing as i128
        }
    }
}

async fn handle_peer_thread_message(
    msg: PeerThreadToMain,
    mine: bool,
    main_to_miner_tx: &watch::Sender<MainToMiner>,
    state: State,
    main_to_peer_broadcast_tx: &broadcast::Sender<MainToPeerThread>,
    synchronization_state: &mut SynchronizationState,
    potential_peers: &mut PotentialPeersState,
) -> Result<()> {
    debug!("Received message sent to main thread.");
    match msg {
        PeerThreadToMain::NewBlocks(blocks) => {
            let last_block = blocks.last().unwrap().to_owned();
            {
                // Acquire locks for blockchain state in correct order to avoid deadlocks
                let mut databases: tokio::sync::MutexGuard<BlockDatabases> = state
                    .chain
                    .archival_state
                    .as_ref()
                    .unwrap()
                    .block_databases
                    .lock()
                    .await;
                let mut previous_block_header: std::sync::MutexGuard<BlockHeader> = state
                    .chain
                    .light_state
                    .latest_block_header
                    .lock()
                    .expect("Lock on block header must succeed");

                // The peer threads also check this condition, if block is more canonical than current
                // tip, but we have to check it again since the block update might have already been applied
                // through a message from another peer.
                let block_is_new = previous_block_header.proof_of_work_family
                    < last_block.header.proof_of_work_family;
                if !block_is_new {
                    return Ok(());
                }

                // Get out of sync mode if needed
                if state.net.syncing.read().unwrap().to_owned() {
                    *state.net.syncing.write().unwrap() = stay_in_sync_mode(
                        last_block.header.clone(),
                        synchronization_state,
                        state.cli.max_number_of_blocks_before_syncing,
                    );
                }

                // When receiving a block from a peer thread, we assume it is verified.
                // It is the peer thread's responsibility to verify the block.
                if mine {
                    main_to_miner_tx.send(MainToMiner::NewBlock(Box::new(last_block.clone())))?;
                }

                // Store blocks in database
                for block in blocks {
                    debug!("Storing block {:?} in database", block.hash);
                    databases
                        .block_height_to_hash
                        .put(block.header.height, block.hash);
                    databases.block_hash_to_block.put(block.hash, block);
                }

                // Update information about latest header
                *previous_block_header = last_block.header.clone();
                databases
                    .latest_block_header
                    .put((), last_block.header.clone());
            }

            // Inform all peers about new block
            main_to_peer_broadcast_tx
                .send(MainToPeerThread::Block(Box::new(last_block)))
                .expect("Peer handler broadcast was closed. This should never happen");
        }
        PeerThreadToMain::NewTransaction(_txs) => {
            error!("Unimplemented txs msg received");
        }
        PeerThreadToMain::PeerMaxBlockHeight((
            socket_addr,
            claimed_max_height,
            claimed_max_pow_family,
        )) => {
            let claimed_state =
                PeerSynchronizationState::new(claimed_max_height, claimed_max_pow_family);
            synchronization_state
                .peer_sync_states
                .insert(socket_addr, claimed_state);

            // Check if synchronization mode should be activated. Synchronization mode is entered if
            // PoW family exceeds our tip and if the height difference is beyond a threshold value.
            // TODO: If we are not checking the PoW claims of the tip this can be abused by forcing
            // the client into synchronization mode.
            let our_block_tip_header: BlockHeader =
                state.chain.light_state.get_latest_block_header();
            if enter_sync_mode(
                our_block_tip_header,
                claimed_state,
                state.cli.max_number_of_blocks_before_syncing,
            ) {
                info!(
                    "Entering synchronization mode due to peer {} indicating tip height {}; pow family: {:?}",
                    socket_addr, claimed_max_height, claimed_max_pow_family
                );
                *state.net.syncing.write().unwrap() = true;
            }
        }
        PeerThreadToMain::PeerDiscoveryAnswer((pot_peers, reported_by, distance)) => {
            let max_peers = state.cli.max_peers;
            for pot_peer in pot_peers {
                potential_peers.add(reported_by, pot_peer, max_peers as usize, distance);
            }
        }
    }

    Ok(())
}

/// Function to perform peer discovery: Finds potential peers from connected peers and attempts
/// to establish connections with one of those potential peers.
async fn peer_discovery_handler(
    state: State,
    main_to_peer_broadcast_tx: broadcast::Sender<MainToPeerThread>,
    potential_peers: &PotentialPeersState,
    peer_thread_to_main_tx: mpsc::Sender<PeerThreadToMain>,
) -> Result<()> {
    let connected_peers: Vec<PeerInfo> = match state.net.peer_map.try_lock() {
        Ok(pm) => pm.values().cloned().collect(),
        Err(_) => return Ok(()),
    };

    if connected_peers.len() > state.cli.max_peers as usize {
        // This would indicate a race-condition on the peer map field in the state which
        // we unfortunately cannot exclude. So we just disconnect from a peer that the user
        // didn't request a connection to.
        warn!(
            "Max peer parameter is exceeded. max is {} but we are connected to {}. Attempting to fix.",
            connected_peers.len(),
            state.cli.max_peers
        );
        let mut rng = thread_rng();

        // pick a peer that was not specified in the CLI arguments to disconnect from
        let peer_to_disconnect = connected_peers
            .iter()
            .filter(|peer| !state.cli.peers.contains(&peer.connected_address))
            .choose(&mut rng);
        match peer_to_disconnect {
            Some(peer) => {
                main_to_peer_broadcast_tx
                    .send(MainToPeerThread::Disconnect(peer.connected_address))?;
            }
            None => warn!("Unable to resolve max peer constraint due to manual override."),
        };

        return Ok(());
    }

    // We don't make an outgoing connection if we've reached the peer limit, *or* if we are
    // one below the peer limit as we reserve this last slot for an ingoing connection.
    if connected_peers.len() == state.cli.max_peers as usize
        || connected_peers.len() > 2 && connected_peers.len() - 1 == state.cli.max_peers as usize
    {
        return Ok(());
    }

    info!("Performing peer discovery");
    // Potential procedure for peer discovey:
    // 0) Ask all peers for their peer lists
    // 1) Get peer candidate from these responses
    // 2) Connect to one of those peers, A.
    // 3) Ask this newly connected peer, A, for its peers.
    // 4) Connect to one of those peers
    // 5) Disconnect from A. (not yet implemented)

    // 0)
    main_to_peer_broadcast_tx.send(MainToPeerThread::MakePeerDiscoveryRequest)?;

    // 1)
    let (peer_candidate, candidate_distance) = match potential_peers.get_distant_candidate(
        &connected_peers,
        state.cli.get_own_listen_address(),
        state.net.instance_id,
    ) {
        Some(candidate) => candidate,
        None => return Ok(()),
    };

    // 2)
    info!(
        "Connecting to peer {} with distance {}",
        peer_candidate, candidate_distance
    );
    let own_handshake_data = state.get_handshakedata().await;
    let main_to_peer_broadcast_rx = main_to_peer_broadcast_tx.subscribe();
    tokio::spawn(async move {
        call_peer_wrapper(
            peer_candidate,
            state.to_owned(),
            main_to_peer_broadcast_rx,
            peer_thread_to_main_tx.to_owned(),
            &own_handshake_data,
            candidate_distance,
        )
        .await;
    });

    // 3
    main_to_peer_broadcast_tx.send(MainToPeerThread::MakeSpecificPeerDiscoveryRequest(
        peer_candidate,
    ))?;

    // 4 is completed in the next call to this function provided that the in (3) connected
    // peer responded to the peer list request.

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub async fn main_loop(
    listener: TcpListener,
    state: State,
    main_to_peer_broadcast_tx: broadcast::Sender<MainToPeerThread>,
    peer_thread_to_main_tx: mpsc::Sender<PeerThreadToMain>,
    mut peer_thread_to_main_rx: mpsc::Receiver<PeerThreadToMain>,
    own_handshake_data: HandshakeData,
    mut miner_to_main_rx: mpsc::Receiver<MinerToMain>,
    main_to_miner_tx: watch::Sender<MainToMiner>,
) -> Result<()> {
    // Handle incoming connections, messages from peer threads, and messages from the mining thread
    let mut sync_state = SynchronizationState::default();
    let mut potential_peers_state = PotentialPeersState::default();
    loop {
        // This timer might have to sleep a random number of seconds for it to be guaranteed to be
        // hit without being interrupted by other processes.
        let peer_discovery_timer =
            time::sleep(Duration::from_secs(PEER_DISCOVERY_INTERVAL_IN_SECONDS));
        tokio::pin!(peer_discovery_timer);
        select! {
            // The second item contains the IP and port of the new connection.
            Ok((stream, _)) = listener.accept() => {

                // TODO: The handshake data is not handled correctly here as it should be
                // generated on each incoming transaction. Now it's just generated at startup
                // and newer updated.
                // Handle incoming connections from peer
                let state = state.clone();
                let main_to_peer_broadcast_rx_clone: broadcast::Receiver<MainToPeerThread> = main_to_peer_broadcast_tx.subscribe();
                let peer_thread_to_main_tx_clone: mpsc::Sender<PeerThreadToMain> = peer_thread_to_main_tx.clone();
                let peer_address = stream.peer_addr().unwrap();
                let own_handshake_data_clone = own_handshake_data.clone();
                let max_peers = state.cli.max_peers;
                tokio::spawn(async move {
                    match answer_peer(
                        stream,
                        state,
                        peer_address,
                        main_to_peer_broadcast_rx_clone,
                        peer_thread_to_main_tx_clone,
                        own_handshake_data_clone,
                        max_peers
                    ).await {
                        Ok(()) => (),
                        Err(err) => error!("Got error: {:?}", err),
                    }
                });

            }

            // Handle messages from peer threads
            Some(msg) = peer_thread_to_main_rx.recv() => {
                info!("Received message sent to main thread.");
                handle_peer_thread_message(
                    msg,
                    state.cli.mine,
                    &main_to_miner_tx,
                    state.clone(),
                    &main_to_peer_broadcast_tx,
                    &mut sync_state,
                    &mut potential_peers_state
                )
                .await?
            }

            // Handle messages from miner thread
            Some(main_message) = miner_to_main_rx.recv() => {
                handle_miner_thread_message(main_message, &main_to_peer_broadcast_tx, state.clone()).await?
            }

            // Start peer discovery in case we
            _ = &mut peer_discovery_timer => {
                // Check number of peers we are connected to
                peer_discovery_handler(state.clone(), main_to_peer_broadcast_tx.clone(), &potential_peers_state, peer_thread_to_main_tx.clone()).await?
            }
            // TODO: Add signal::ctrl_c/shutdown handling here
        }
    }
}
