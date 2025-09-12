/*
 * Copyright (C) 2019 Open Whisper Systems
 * Copyright (C) 2021 jessa0
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

//! Higher-level API for a Raft node.

use alloc::collections::BTreeSet;
use bytes::Bytes;
use core::fmt::Display;
use crate::core::{RaftState, ReplicationState};
use crate::message::{LogIndex, RaftMessage, SendableRaftMessage, TermId};
use crate::log::{CommittedIter, RaftLog};
use rand_core::RngCore;

use vstd::prelude::*;
use scapegoat::SgSet;
use core::default::Default;

verus! {

/// A Raft node, used for replicating a strongly-consistent distributed log of entries with arbitrary data amongst its
/// peers.
///
/// The distributed log can be used, for example, to replicate transactions in a database.
///
/// # Appending entries to the distributed log
///
/// Log entries passed to [`append`] are not guaranteed to ultimately be appended to the distributed log, and may be
/// cancelled any time [`receive`] is called before they are "committed". The provided [`RaftLog`] should provide an API
/// to find out which log entries have been cancelled. Only log entries passed to [`append`] on a particular node are
/// guaranteed to appear as cancelled in its own [`RaftLog`], but entries appended on other nodes may appear as well.
///
/// The distributed log may only be appended to by the node returned by [`leader`], but even that node is not guaranteed
/// to be able to append to the log, since it must be able to send each new entry to a majority of its peers before
/// losing leadership in order for the entry to become committed. The leader may change at any time, and therefore an
/// entry may be first returned from [`take_committed`] on a node different than that to which it was submitted.
/// However, [`take_committed`] is guaranteed to return the same entries in the same order on every node.
///
/// # Timer ticks
///
/// Timeouts in [`RaftNode`] are driven by a timer ticking at fixed interval, with the number of ticks between timeouts
/// configured by the provided [`RaftConfig`]. Any consistent time interval between ticks may be chosen, but the time
/// interval and [`RaftConfig`] must be the same on all peers in a group. Shorter timeouts will allow Raft to react
/// quicker to network disruptions, but may result in spurious leadership changes when the network latency exceeds
/// `time_interval * election_timeout_ticks`.
///
/// # Message delivery
///
/// Unicast message delivery is assumed to be non-lossy in order for replication to make progress. In other words, once
/// a non-broadcast [`SendableRaftMessage`] is returned from an API such as [`append`], [`receive`], or [`timer_tick`],
/// it must be retained and retransmitted until it is confirmed to have been processed by [`receive`] on its
/// destination. Messages may be safely delivered out-of-order or more than once, however.
///
/// To prevent unbounded queueing, the API is designed to only ever return a bounded amount of unacknowledged unicast
/// message data. This amount can be approximately controlled by [`replication_chunk_size`].
///
/// [`append`]: Self::append
/// [`leader`]: Self::leader
/// [`receive`]: Self::receive
/// [`replication_chunk_size`]: RaftConfig::replication_chunk_size
/// [`SendableRaftMessage`]: crate::message::SendableRaftMessage
/// [`take_committed`]: Self::take_committed
/// [`timer_tick`]: Self::timer_tick
#[verifier::reject_recursive_types(Log)]
#[verifier::reject_recursive_types(NodeId)]
pub struct RaftNode<Log, Random, NodeId : Ord+Default> {
    state: RaftState<Log, Random, NodeId>,
}

/// Configurable parameters of a Raft node.
#[derive(Clone, Eq, PartialEq)]
pub struct RaftConfig {
    /// The minimum number of timer ticks between leadership elections.
    pub election_timeout_ticks: u32,

    /// The number of timer ticks between sending heartbeats to peers.
    pub heartbeat_interval_ticks: u32,

    /// The maximum number of bytes to replicate to a peer at a time.
    pub replication_chunk_size: usize,
}
 }

/// An error returned while attempting to append to a Raft log.
pub enum AppendError<E> {
    /// The append to the Raft log was cancelled and should be resubmitted to the current Raft leader.
    Cancelled {
        /// Arbitrary data associated with the log entry.
        data: Bytes,
    },
    /// An error was returned by the [`RaftLog`](crate::log::RaftLog) implementation.
    RaftLogErr(E),
}

impl<Log, Random, NodeId : Ord+Default> RaftNode<Log, Random, NodeId>
where Log: RaftLog,
      Random: RngCore,
      NodeId: Ord + Clone + Display,
{
    /// Constructs a new Raft node with specified peers and configuration.
    ///
    /// The Raft node will start with an empty initial state. The `log` provided should also be in an empty initial
    /// state. Each Raft node in a group must be constructed with the same set of peers and `config`. `peers` may
    /// contain `node_id` or omit it to the same effect. `rand` must produce different values on every node in a group.
    pub fn new(
        node_id: NodeId,
        peers:   SgSet<NodeId,{usize::MAX}>,
        log:     Log,
        random:  Random,
        config:  RaftConfig,
    ) -> Self {
        Self {
            state: RaftState::new(
                node_id,
                peers,
                log,
                random,
                config,
            ),
        }
    }


    /// Request appending an entry with arbitrary `data` to the Raft log, returning messages to be sent.
    ///
    /// See ["Message delivery"] for details about delivery requirements for the returned messages.
    ///
    /// # Errors
    ///
    /// If this request would immediately be cancelled, then an error is returned.
    ///
    /// ["Message delivery"]: RaftNode#message-delivery
    #[must_use = "This function returns Raft messages to be sent."]
    pub fn append<T: Into<Bytes>>(&mut self, data: T) -> Result<impl Iterator<Item = SendableRaftMessage<NodeId>> + '_, AppendError<Log::Error>> {
        let () = self.state.client_request(data.into())?;
        Ok(self.append_entries())
    }

    /// Returns this node's configurable parameters.
    pub fn config(&self) -> &RaftConfig {
        self.state.config()
    }

    /// Returns whether this node is the leader of the latest known term.
    pub fn is_leader(&self) -> bool {
        self.state.is_leader()
    }

    /// Returns the index of the last [`LogEntry`] which has been committed and thus may be returned by
    /// [`take_committed`].
    ///
    /// [`take_committed`]: Self::take_committed
    /// [`LogEntry`]: crate::message::LogEntry
    pub fn last_committed_log_index(&self) -> LogIndex {
        *self.state.commit_idx()
    }

    /// Returns the ID of the leader, if there is one, of the latest known term, along with the term.
    pub fn leader(&self) -> (Option<&NodeId>, TermId) {
        let (leader, term) = self.state.leader();
        (leader, *term)
    }

    /// Returns a reference to the Raft log storage.
    pub fn log(&self) -> &Log {
        self.state.log()
    }

    /// Returns a mutable reference to the Raft log storage.
    pub fn log_mut(&mut self) -> &mut Log {
        self.state.log_mut()
    }

    /// Returns this node's ID.
    pub fn node_id(&self) -> &NodeId {
        self.state.node_id()
    }

    /// Returns the IDs of this node's peers.
    pub fn peers(&self) -> &SgSet<NodeId,{usize::MAX}> {
        self.state.peers()
    }

    /// Processes receipt of a `message` from a peer with ID `from`, returning messages to be sent.
    ///
    /// See ["Message delivery"] for details about delivery requirements for the returned messages.
    ///
    /// ["Message delivery"]: RaftNode#message-delivery
    #[must_use = "This function returns Raft messages to be sent."]
    pub fn receive(
        &mut self,
        message: RaftMessage,
        from:    NodeId,
    ) -> impl Iterator<Item = SendableRaftMessage<NodeId>> + '_ {
        let message = self.state.receive(message, from);
        message.into_iter().chain(self.append_entries())
    }

    /// Returns the replication state corresponding to the peer with ID `peer_node_id`.
    pub fn replication_state(&self, peer_node_id: &NodeId) -> Option<&ReplicationState> {
        self.state.replication_state(peer_node_id)
    }

    /// Returns a reference to the low-level state of the Raft node.
    pub fn state(&mut self) -> &RaftState<Log, Random, NodeId> {
        &self.state
    }

    /// Returns a mutable reference to the low-level state of the Raft node.
    pub fn state_mut(&mut self) -> &mut RaftState<Log, Random, NodeId> {
        &mut self.state
    }

    /// Returns an iterator yielding committed [log entries][`LogEntry`]. A given [`LogEntry`] will be yielded only once
    /// over the lifetime of a [`RaftNode`]. See ["Appending entries to the distributed log"] for details about log
    /// commital.
    ///
    /// ["Appending entries to the distributed log"]: RaftNode#appending-entries-to-the-distributed-log
    /// [`LogEntry`]: crate::message::LogEntry
    pub fn take_committed(&mut self) -> CommittedIter<'_, Log> {
        self.state.take_committed()
    }

    /// Ticks forward this node's internal clock by one tick, returning messages to be sent.
    ///
    /// See ["Message delivery"] for details about delivery requirements for the returned messages.
    ///
    /// ["Message delivery"]: RaftNode#message-delivery
    #[must_use = "This function returns Raft messages to be sent."]
    pub fn timer_tick(&mut self) -> impl Iterator<Item = SendableRaftMessage<NodeId>> + '_ {
        let message = self.state.timer_tick();
        message.into_iter().chain(self.append_entries())
    }

    #[must_use = "This function returns Raft messages to be sent."]
    fn append_entries(
        &mut self,
    ) -> impl Iterator<Item = SendableRaftMessage<NodeId>> + '_ {
        let peers = self.state.peers().clone().into_iter();
        peers.flat_map(move |peer| self.state.append_entries(peer))
    }
}
