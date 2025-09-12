/*
 * Copyright (C) 2019 Open Whisper Systems
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

//! Raft message types for sending between nodes.
//!
//! This module provides data types for messages to be sent between Raft nodes. The top-level message type is
//! [`RaftMessage`]. Protobuf-based serialization of all types in this module is provided through the `prost` crate if
//! the corresponding feature is enabled.

use bytes::Bytes;
use core::cmp::Ordering;
use core::fmt;
use core::ops::{Add, AddAssign, Sub};
use crate::prelude::*;

use vstd::prelude::*;


verus!{

pub assume_specification [<Bytes as Clone>::clone](b: &Bytes) -> (res: Bytes)
    ensures res == b;

#[verifier::external_type_specification]
#[verifier::external_body]
pub struct ExBytes(bytes::Bytes);


/// A [`RaftMessage`] to be sent to a destination.
pub struct SendableRaftMessage<NodeId> {
    /// The message to be sent.
    pub message: RaftMessage,

    /// The destination for the message.
    pub dest: RaftMessageDestination<NodeId>,
}


/// The destination for a [`SendableRaftMessage`].
pub enum RaftMessageDestination<NodeId> {
    /// The associated message should be sent to all known peers.
    Broadcast,
    /// The associated message should be sent to one particular peer.
    To(NodeId),
}

/// A message sent between Raft nodes.
#[derive(Clone, PartialEq)]
#[cfg_attr(feature = "prost", derive(prost::Message))]
#[cfg_attr(not(feature = "prost"), derive(Debug, Default))]
pub struct RaftMessage {
    /// The greatest Raft leadership term ID seen by the sender.
    #[cfg_attr(feature = "prost", prost(message, required, tag="2"))]
    pub term: TermId,

    /// The Remote Procedure Call contained by this message.
    ///
    /// This field is only optional in order to support protobuf serialization.
    #[cfg_attr(feature = "prost", prost(oneof="Rpc", tags="3, 4, 5, 6"))]
    pub rpc: Option<Rpc>,
}

/// A Remote Procedure Call message to a Raft node.
#[derive(Clone, PartialEq)]
#[cfg_attr(feature = "prost", derive(prost::Oneof))]
#[cfg_attr(not(feature = "prost"), derive(Debug))]
pub enum Rpc {
    /// A request to obtain leadership amongst Raft nodes.
    #[cfg_attr(feature = "prost", prost(message, tag="3"))]
    VoteRequest(VoteRequest),

    /// A response to a [`VoteRequest`] granting or denying leadership.
    #[cfg_attr(feature = "prost", prost(message, tag="4"))]
    VoteResponse(VoteResponse),

    /// A request to append entries to a Raft node's log.
    #[cfg_attr(feature = "prost", prost(message, tag="5"))]
    AppendRequest(AppendRequest),

    /// A response to an [`AppendRequest`] allowing or denying an append to the Raft node's log.
    #[cfg_attr(feature = "prost", prost(message, tag="6"))]
    AppendResponse(AppendResponse),
}

/// A request to obtain leadership amongst Raft nodes.
#[derive(Clone, PartialEq)]
#[cfg_attr(feature = "prost", derive(prost::Message))]
#[cfg_attr(not(feature = "prost"), derive(Debug, Default))]
pub struct VoteRequest {
    /// The Raft log index of the last log entry stored by this node.
    #[cfg_attr(feature = "prost", prost(message, required, tag="2"))]
    pub last_log_idx: LogIndex,

    /// The Raft leadership term of the last log entry stored by this node.
    #[cfg_attr(feature = "prost", prost(message, required, tag="3"))]
    pub last_log_term: TermId,
}

/// The response to a [`VoteRequest`] granting or denying leadership.
#[derive(Clone, PartialEq)]
#[cfg_attr(feature = "prost", derive(prost::Message))]
#[cfg_attr(not(feature = "prost"), derive(Debug, Default))]
pub struct VoteResponse {
    /// Whether the [`VoteRequest`] was granted or not.
    #[cfg_attr(feature = "prost", prost(bool, required, tag="2"))]
    pub vote_granted: bool,
}



/// A request to append entries to a Raft node's log.
#[derive(Clone, PartialEq)]
#[cfg_attr(feature = "prost", derive(prost::Message))]
#[cfg_attr(not(feature = "prost"), derive(Debug, Default))]
pub struct AppendRequest {
    /// The Raft log index immediately before the index of the first entry in [`entries`](Self::entries).
    #[cfg_attr(feature = "prost", prost(message, required, tag="1"))]
    pub prev_log_idx: LogIndex,

    /// The Raft leadership term of the log entry immediately before the first entry in [`entries`](Self::entries).
    #[cfg_attr(feature = "prost", prost(message, required, tag="2"))]
    pub prev_log_term: TermId,

    /// The Raft log index of the last log entry known by the requester to be committed.
    #[cfg_attr(feature = "prost", prost(message, required, tag="3"))]
    pub leader_commit: LogIndex,

    /// A list of consecutive Raft log entries to append.
    #[cfg_attr(feature = "prost", prost(message, repeated, tag="4"))]
    pub entries: Vec<LogEntry>,
}



/// The response to an [`AppendRequest`] allowing or denying an append to the Raft node's log.
#[derive(Clone, PartialEq)]
#[cfg_attr(feature = "prost", derive(prost::Message))]
#[cfg_attr(not(feature = "prost"), derive(Debug, Default))]
pub struct AppendResponse {
    /// Whether the [`AppendRequest`] was granted or not.
    #[cfg_attr(feature = "prost", prost(bool, required, tag="1"))]
    pub success: bool,

    /// The Raft log index of the last log entry up to which the responder's log is known to match the requester's log.
    #[cfg_attr(feature = "prost", prost(message, required, tag="2"))]
    pub match_idx: LogIndex,

    /// The Raft log index of the last log entry in the responder's log.
    #[cfg_attr(feature = "prost", prost(message, required, tag="3"))]
    pub last_log_idx: LogIndex,
}


/// An entry in a [Raft log][crate::log::RaftLog].
#[derive(Clone, PartialEq)]
#[cfg_attr(feature = "prost", derive(prost::Message))]
#[cfg_attr(not(feature = "prost"), derive(Debug, Default))]
pub struct LogEntry {
    /// The term of leadership of the node which appended this log entry.
    #[cfg_attr(feature = "prost", prost(message, required, tag="1"))]
    pub term: TermId,

    /// Arbitrary data associated with the log entry.
    #[cfg_attr(feature = "prost", prost(bytes="vec", required, tag="2"))]
    pub data: Bytes,
}


/// The unique, monotonically-increasing ID for a term of Raft group leadership.
#[derive(Clone,PartialEq,Eq,PartialOrd,Ord,Structural)]
#[cfg_attr(feature = "prost", derive(prost::Message))]
#[cfg_attr(not(feature = "prost"), derive(Debug, Default))]
pub struct TermId {
    /// The non-negative integer assigned to this term.
    #[cfg_attr(feature = "prost", prost(uint64, required, tag="1"))]
    pub id: u64,
}

/// A 1-based index into a [Raft log][crate::log::RaftLog].
#[derive(Clone, PartialEq)]
#[cfg_attr(feature = "prost", derive(prost::Message))]
#[cfg_attr(not(feature = "prost"), derive(Debug, Default))]
pub struct LogIndex {
    /// The integer representing this log index.
    #[cfg_attr(feature = "prost", prost(uint64, required, tag="1"))]
    pub id: u64,
}
}
//
// RaftMessage impls
//

impl fmt::Display for RaftMessage {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self { term, rpc } = self;
        let mut debug = fmt.debug_tuple("");
        debug.field(&format_args!("{}", term));
        if let Some(rpc) = rpc {
            debug.field(&format_args!("{}", rpc));
        } else {
            debug.field(&"None");
        }
        debug.finish()
    }
}

//
// Rpc impls
//

impl fmt::Display for Rpc {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self {
            Rpc::VoteRequest(msg)    => fmt::Display::fmt(msg, fmt),
            Rpc::VoteResponse(msg)   => fmt::Display::fmt(msg, fmt),
            Rpc::AppendRequest(msg)  => fmt::Display::fmt(msg, fmt),
            Rpc::AppendResponse(msg) => fmt::Display::fmt(msg, fmt),
        }
    }
}

//
// VoteRequest impls
//

impl fmt::Display for VoteRequest {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self { last_log_idx, last_log_term } = self;
        fmt.debug_struct("VoteRequest")
           .field("last_log_idx",  &format_args!("{}", last_log_idx))
           .field("last_log_term", &format_args!("{}", last_log_term))
           .finish()
    }
}

//
// VoteResponse impls
//

impl fmt::Display for VoteResponse {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self { vote_granted } = self;
        fmt.debug_struct("VoteResponse")
           .field("vote_granted", vote_granted)
           .finish()
    }
}

//
// AppendRequest impls
//

impl fmt::Display for AppendRequest {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self { prev_log_idx, prev_log_term, leader_commit, entries } = self;
        fmt.debug_struct("AppendRequest")
           .field("prev_log_idx",  &format_args!("{}", prev_log_idx))
           .field("prev_log_term", &format_args!("{}", prev_log_term))
           .field("leader_commit", &format_args!("{}", leader_commit))
           .field("entries",       &entries.len())
           .finish()
    }
}

//
// AppendResponse impls
//

impl fmt::Display for AppendResponse {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self { success, match_idx, last_log_idx } = self;
        fmt.debug_struct("AppendResponse")
           .field("success",      &success)
           .field("match_idx",    &format_args!("{}", match_idx))
           .field("last_log_idx", &format_args!("{}", last_log_idx))
           .finish()
    }
}


//
// TermId impls
//

impl fmt::Display for TermId {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self { id } = self;
        fmt.debug_tuple("TermId")
           .field(id)
           .finish()
    }
}


verus!{


impl Copy for TermId {}

// TODO: not working yet
impl SpecOrd<TermId> for TermId {  
    fn spec_lt(self, rhs: TermId) -> bool{
        self.id.spec_lt(rhs.id)
    }
    fn spec_le(self, rhs: TermId) -> bool{
        self.id.spec_le(rhs.id)
    }
    fn spec_gt(self, rhs: TermId) -> bool{
        self.id.spec_gt(rhs.id)
    }
    fn spec_ge(self, rhs: TermId) -> bool{
        self.id.spec_ge(rhs.id)
    }
}

// impl Eq for TermId {}
// impl PartialOrd for TermId {
//     fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
// }
// impl Ord for TermId {
//     fn cmp(&self, other: &Self) -> Ordering { self.id.cmp(&other.id) }
// }


impl AddAssign<u64> for TermId {
    fn add_assign(&mut self, rhs: u64)  {
        // TODO: decide standard
        self.id = self.id.checked_add(rhs).unwrap_or(u64::MAX);
    }
}

}

//
// LogIndex impls
//

impl LogIndex {
    /// Subtraction with a non-negative integer, checking for overflow. Returns `self - dec`, or `None` if an overflow
    /// occurred.
    pub fn checked_sub(self, dec: u64) -> Option<Self> {
        if let Some(id) = self.id.checked_sub(dec) {
            Some(Self { id })
        } else {
            None
        }
    }
}

impl fmt::Display for LogIndex {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self { id } = self;
        fmt.debug_tuple("LogIdx")
           .field(id)
           .finish()
    }
}

impl Copy for LogIndex {}
impl Eq for LogIndex {}
impl PartialOrd for LogIndex {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl Ord for LogIndex {
    fn cmp(&self, other: &Self) -> Ordering { self.id.cmp(&other.id) }
}
impl Add<u64> for LogIndex {
    type Output = Self;
    fn add(self, inc: u64) -> Self {
        Self { id: self.id.checked_add(inc).unwrap_or_else(|| panic!("overflow")) }
    }
}
impl Sub<u64> for LogIndex {
    type Output = Self;
    fn sub(self, dec: u64) -> Self {
        Self { id: self.id.saturating_sub(dec) }
    }
}
