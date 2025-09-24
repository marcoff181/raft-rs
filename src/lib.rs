/*
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

//! Raft consensus algorithm implementation.
//!
//! Raft is a consensus algorithm which replicates a strongly-consistent distributed log of entries with arbitrary data
//! amongst a group of peers. It is also fault-tolerant, allowing replication to continue while a majority of peers can
//! still communicate with each other. This crate provides an implementation of the Raft consensus algorithm with some
//! optional features not implemented, such as pre-voting, membership changes, and snapshots.
//!
//! The Raft algorithm is implemented as a state machine driven in a few ways:
//!
//! * When attempting to append a new entry to the distributed log: [`append`](node::RaftNode::append) is called.
//! * When a message is received from a peer: [`receive`](node::RaftNode::receive) is called.
//! * Every time a fixed amount of time has elapsed: [`timer_tick`](node::RaftNode::timer_tick) is called.
//!
//! Each of these functions modifies the internal state and returns [messages](message::SendableRaftMessage) to be sent
//! to peers. Once a log entry is "committed", or guaranteed to be returned at the same index on every functioning peer
//! in the group, it may be retrieved using [`take_committed`](node::RaftNode::take_committed). An append to the log may
//! be cancelled before reaching the committed state, however, which is discussed in more detail in ["Appending entries to the distributed log"].
//!
//! The backing storage for the distributed log must be provided as an implementation of the [`RaftLog`](log::RaftLog)
//! trait, with careful attention to following the trait specification. A trivial in-memory implementation is provided
//! by [`RaftLogMemory`](log::mem::RaftLogMemory).
//!
//! # Example
//!
//! ```
//! use simple_raft::log::mem::RaftLogMemory;
//! use simple_raft::node::{RaftConfig, RaftNode};
//! use simple_raft::message::{RaftMessageDestination, SendableRaftMessage};
//! use rand_chacha::ChaChaRng;
//! use rand_core::SeedableRng;
//! use std::collections::VecDeque;
//! use std::str;
//!
//! // Construct 5 Raft peers
//! type NodeId = nat;
//! let mut peers = (0..5).map(|id: NodeId| RaftNode::new(
//!     id,
//!     (0..5).collect(),
//!     RaftLogMemory::new_unbounded(),
//!     ChaChaRng::seed_from_u64(id as u64),
//!     RaftConfig {
//!         election_timeout_ticks: 10,
//!         heartbeat_interval_ticks: 1,
//!         replication_chunk_size: nat::max_value(),
//!     },
//! )).collect::<Vec<_>>();
//!
//! // Simulate reliably sending messages instantaneously between peers
//! let mut inboxes = vec![VecDeque::new(); peers.len()];
//! let send_message = |src_id: NodeId, sendable: SendableRaftMessage<NodeId>, inboxes: &mut Vec<VecDeque<_>>| {
//!     match sendable.dest {
//!         RaftMessageDestination::Broadcast => {
//!             println!("peer {} -> all: {}", src_id, &sendable.message);
//!             inboxes.iter_mut().for_each(|inbox| inbox.push_back((src_id, sendable.message.clone())))
//!         }
//!         RaftMessageDestination::To(dst_id) => {
//!             println!("peer {} -> peer {}: {}", src_id, dst_id, &sendable.message);
//!             inboxes[dst_id].push_back((src_id, sendable.message));
//!         }
//!     }
//! };
//!
//! // Loop until a log entry is committed on all peers
//! let mut appended = false;
//! let mut peers_committed = vec![false; peers.len()];
//! while !peers_committed.iter().all(|seen| *seen) {
//!     for (peer_id, peer) in peers.iter_mut().enumerate() {
//!         // Tick the timer
//!         let new_messages = peer.timer_tick();
//!         new_messages.for_each(|message| send_message(peer_id, message, &mut inboxes));
//!
//!         // Append a log entry on the leader
//!         if !appended && peer.is_leader() {
//!             if let Ok(new_messages) = peer.append("Hello world!") {
//!                 new_messages.for_each(|message| send_message(peer_id, message, &mut inboxes));
//!                 appended = true;
//!             }
//!         }
//!
//!         // Process message inbox
//!         while let Some((src_id, message)) = inboxes[peer_id].pop_front() {
//!             let new_messages = peer.receive(message, src_id);
//!             new_messages.for_each(|message| send_message(peer_id, message, &mut inboxes));
//!         }
//!
//!         // Check for committed log entries
//!         for log_entry in peer.take_committed() {
//!             if !log_entry.data.is_empty() {
//!                 println!("peer {} saw commit {}", peer_id, str::from_utf8(&log_entry.data).unwrap());
//!                 assert!(!peers_committed[peer_id]);
//!                 peers_committed[peer_id] = true;
//!             }
//!         }
//!     }
//! }
//! ```
//!
//! ["Appending entries to the distributed log"]: node::RaftNode#appending-entries-to-the-distributed-log

#![no_std]

#![allow(unused_parens)]
#![warn(missing_docs)]

extern crate alloc;

#[macro_use]
mod macros;

pub mod core;
pub mod log;
pub mod message;
pub mod node;
mod prelude;



use verus_builtin_macros::*;
use verus_state_machines_macros::tokenized_state_machine;
use vstd::prelude::*;
use vstd::prelude::nat;
use vstd::relations::total_ordering;
use vstd::multiset::Multiset;


verus! {

#[derive(Clone, PartialEq, Eq, Debug, Default)]
struct Dummy {}

// #[verifier::external_trait_specification]
// pub trait ExDefault where Self: core::marker::Sized {
//     type ExternalTraitSpecificationFor: Default;
//
//     fn default() -> Self where Self: core::marker::Sized;
// }

pub struct ElectionRecord {
    pub term : nat,
    pub leader: nat,
    pub log: Seq<LogEntry>,
    pub votes: Set<nat>,
    pub voter_log:Map<nat,Seq<LogEntry>>,
}


pub struct LogEntry {
    pub term: nat,
    // the proof is not concerned with the actual values inside the log
    pub item: (),
}

pub enum RaftMessage
{
    AppendEntriesRequest {
        src: nat,
        dest: nat,
        term: nat,
        prev_index: nat,
        prev_term: nat,
        commit_index: nat,
        entries: Seq<LogEntry>,
    },
    AppendEntriesResponse {
        src: nat,
        dest: nat,
        term: nat,
        success: bool,
        match_index: nat,
    },
    RequestVoteRequest {
        src: nat,
        dest: nat,
        term: nat,
        last_log_index: nat,
        last_log_term: nat,
    },
    RequestVoteResponse {
        src: nat,
        dest: nat,
        term: nat,
        vote_granted: bool,
    },
}

pub enum ServerState {
    Leader,
    Candidate,
    Follower,
}

//TODO: check that all message handlers check that the opposite of update_term condition is true
tokenized_state_machine!{
    RAFT{
        fields{
            #[sharding(multiset)]
            pub messages: Multiset<RaftMessage>,

            #[sharding(constant)]
            pub servers: Set<nat>,

            #[sharding(constant)]
            pub quorum: Set<Set<nat>>,

            // #[sharding(not_tokenized)]
            // term -> election_record
            #[sharding(map)]
            pub elections:Map<nat,ElectionRecord>,

            #[sharding(map)]
            pub allLogs: Map<nat,LogEntry>,

            // per-server variables
            #[sharding(map)]
            pub current_term: Map<nat,nat>,

            #[sharding(map)]
            pub state: Map<nat,ServerState>,

            #[sharding(map)]
            pub voted_for: Map<nat,Option<nat>>,

            #[sharding(map)]
            pub log: Map<nat,Seq<LogEntry>>,

            // \* The index of the latest entry in the log the state machine may apply.
            // VARIABLE commitIndex
            #[sharding(map)]
            pub commit_index: Map<nat,nat>,

            // candidates only variables
            
            //The set of servers from which the candidate has received a RequestVote response in its currentTerm.
            #[sharding(map)]
            pub votes_responded: Map<nat,Set<nat>>,

            //The set of servers from which the candidate has received a vote in its currentTerm.
            #[sharding(map)]
            pub votes_granted: Map<nat,Set<nat>>,

            // A history variable used in the proof. This would not be present in an implementation.
            // Function from each server that voted for this candidate in its currentTerm to that voter's log.
            #[sharding(map)]
            pub voter_log: Map<nat,Map<nat,Seq<LogEntry>>>,

            // the next entry to send to each follower
            #[sharding(map)]
            pub next_index: Map<nat,Map<nat,nat>>,

            // \* The latest entry that each follower has acknowledged is the same as the
            // \* leader's. This is used to calculate commitIndex on the leader.
            #[sharding(map)]
            pub match_index: Map<nat,Map<nat,nat>>,
        }

        // if two servers are leaders then their terms must be different
        // similar to the invariant defined here: https://arxiv.org/html/2403.18916v1
        #[invariant]
        pub fn election_safety(&self) -> bool { 
            forall |i: nat, j: nat| 
                    i != j &&
                    self.state.get(i) == Some(ServerState::Leader) &&
                    self.state.get(j) == Some(ServerState::Leader)
                    ==> #[trigger] self.current_term.get(i) != #[trigger] self.current_term.get(j)
        }

        // #[invariant]
        // pub fn message_correctness(&self) -> bool { 
        //     forall |m: RaftMessage|
        //         self.messages.contains(m)
        //         ==> #[trigger]self.servers.contains(m.src()) &&
        //             #[trigger]self.servers.contains(m.dest()) 
        // }

        #[invariant]
        pub fn matching_domains(&self) -> bool { 
            &&& self.servers =~= self.current_term.dom() 
            &&& self.servers =~= self.state.dom()
            &&& self.servers =~= self.voted_for.dom() 
            &&& self.servers =~= self.log.dom() 
            &&& self.servers =~= self.commit_index.dom() 
            &&& self.servers =~= self.votes_responded.dom() 
            &&& self.servers =~= self.votes_granted.dom() 
            &&& self.servers =~= self.voter_log.dom() 
            &&& self.servers =~= self.next_index.dom() 
            &&& self.servers =~= self.match_index.dom() 
        }

        #[invariant]
        pub fn leader_has_quorum(&self) -> bool { 
            forall |i:nat|
                self.state.get(i) == Some(ServerState::Leader)
                ==> #[trigger]self.quorum.contains(self.votes_granted.get(i).unwrap()) 
        }

        // need to state that: each node votes only once in its term
        //                     between the sets of nodes of the same term there can only be one
        //                     with quorum 
        
        // #[invariant]
        // pub fn votes_granted_disjoint(&self) -> bool { 
        //     forall |i:nat,j:nat|
        //         i != j &&
        //         !self.votes_granted.get(i).unwrap().is_empty() &&
        //         !self.votes_granted.get(j).unwrap().is_empty() &&
        //         self.current_term.contains_key(i) &&
        //         self.current_term.contains_key(j) &&
        //         self.current_term.get(i) == self.current_term.get(j)
        //         ==> !#[trigger](self.votes_granted.get(i).unwrap().disjoint(self.votes_granted.get(j).unwrap()))
        // }

        // // only one leader can be elected per term
        // #[invariant]
        // pub fn election_safety_v2(&self) -> bool { 
        //     forall |r1: ElectionRecord, r2:ElectionRecord| 
        //             r1.leader != r2.leader &&
        //             self.elections.contains_value(r1) &&
        //             self.elections.contains_value(r2) 
        //             ==> #[trigger] r1.term != #[trigger] r2.term
        // }

        #[invariant]
        pub fn quorum_intersects(&self) -> bool { 
            forall |i:Set<nat>,j:Set<nat>|
                i != j &&
                self.quorum.contains(i) &&
                self.quorum.contains(j) 
                ==> !#[trigger]i.disjoint(j)
        }


        #[inductive(initialize)]
        fn initialize_inductive(post: Self, servers: Set<nat>) 
        {
            // proof for `quorum_intersects` 
            assert forall |i:Set<nat>,j:Set<nat>|  
                i != j &&
                post.quorum.contains(i) &&
                post.quorum.contains(j) 
                implies !#[trigger]i.disjoint(j)
            by { 
                // verus knows that i.union(j) is a subset of servers, so we can say that it is also
                // of len <= to len of servers
                vstd::set_lib::lemma_len_subset(i.union(j),servers);
                assert(i.union(j).len() <= post.servers.len());

                // proof by contradiction
                if i.disjoint(j){
                    // ensures i.disjoint(j) ==> i.union(j).len() == i.len()+j.len();
                    vstd::set_lib::lemma_set_disjoint_lens(i,j);
                    assert(i.union(j).len() > post.servers.len());
                }
            }
        }

        init!{
            initialize(servers: Set<nat>)
            {
                require(servers.finite());
                require(servers.len() > 1);
                // require(servers.len() < 10);

                init servers = servers;
                init quorum = Set::new(|s: Set<nat>|   
                    s.subset_of(servers) && s.finite() && s.len() > servers.len() / 2  
                );
                init messages = Multiset::empty();
                init elections = Map::empty();
                init allLogs = Map::empty();
                // TODO: maybe for verification purposes building sets from empty and then adding
                // items has better guarantees
                init current_term = Map::new(|i:nat| servers has i , |i:nat| 1);
                init state = Map::new(|i:nat| servers has i, |i:nat| ServerState::Follower);
                init voted_for = Map::new(|i:nat| servers has i, |i:nat| None);
                init log = Map::new(|i:nat| servers has i, |i:nat| Seq::empty());
                init commit_index = Map::new(|i:nat| servers has i, |i:nat| 0);
                init votes_responded = Map::new(|i:nat| servers has i, |i:nat| Set::empty());
                init votes_granted = Map::new(|i:nat| servers has i, |i:nat| Set::empty());
                init voter_log = Map::new(|i:nat| servers has i, |i:nat| Map::empty());
                init next_index = Map::new(|i:nat| servers has i, |i:nat| Map::new(|j:nat| servers has j, |j:nat| 1));
                init match_index = Map::new(|i:nat| servers has i, |i:nat| Map::new(|j:nat| servers has j, |j:nat| 0));
            }
        }
        // TODO: add advanced term logic

        #[inductive(restart)]
        fn restart_inductive(pre: Self, post: Self, i: nat) { 
            // leader_has_quorum
            assert forall |j:nat|
                i != j implies pre.votes_granted.get(j)  =~= post.votes_granted.get(j)
            by{};
        }

        transition!{
            restart(i:nat){
                // /\ state'          = [state EXCEPT ![i] = Follower]
                // /\ votesResponded' = [votesResponded EXCEPT ![i] = {}]
                // /\ votesGranted'   = [votesGranted EXCEPT ![i] = {}]
                // /\ voterLog'       = [voterLog EXCEPT ![i] = [j \in {} |-> <<>>]]
                // /\ nextIndex'      = [nextIndex EXCEPT ![i] = [j \in Server |-> 1]]
                // /\ matchIndex'     = [matchIndex EXCEPT ![i] = [j \in Server |-> 0]]
                // /\ commitIndex'    = [commitIndex EXCEPT ![i] = 0]
                remove state -= [ i => let _];
                remove votes_responded -= [ i => let _];
                remove votes_granted -= [ i => let _];
                remove voter_log -= [ i => let _];
                remove next_index -= [ i => let _];
                remove match_index -= [ i => let _];
                remove commit_index -= [ i => let _];

                add state += [ i => ServerState::Follower];
                add votes_responded += [ i => Set::empty()];
                add votes_granted += [ i => Set::empty()];
                add voter_log += [ i => Map::empty()];
                add next_index += [ i => Map::new(|x:nat| pre.servers has x, |x:nat| 1)];
                add match_index += [ i => Map::new(|x:nat| pre.servers has x, |x:nat| 0)];
                add commit_index += [ i => 0];

                // /\ UNCHANGED <<messages, currentTerm, votedFor, log, elections>>
            }
        }

        #[inductive(timeout)]
        fn timeout_inductive(pre: Self, post: Self, i: nat) { 
            // election_safety
            assert(forall |j: nat| 
                pre.state.get(j) == Some(ServerState::Leader) 
                ==> post.state.get(j) == Some(ServerState::Leader) &&
                    pre.current_term.get(j) ==  post.current_term.get(j));

            // leader_has_quorum
            assert forall |j:nat|
                i != j implies pre.votes_granted.get(j)  =~= post.votes_granted.get(j)
            by{};
        }

        transition!{
            timeout(i:nat){
                // /\ state[i] \in {Follower, Candidate}
                // /\ state' = [state EXCEPT ![i] = Candidate]
                remove state -= [ i => let state];
                require(state == ServerState::Follower || state == ServerState::Candidate);
                add state += [ i => ServerState::Candidate];

                // /\ currentTerm' = [currentTerm EXCEPT ![i] = currentTerm[i] + 1]
                remove current_term -= [ i => let i_current_term];
                add current_term += [ i => (i_current_term+1)];

                // \* Most implementations would probably just set the local vote
                // \* atomically, but messaging localhost for it is weaker.
                // /\ votedFor' = [votedFor EXCEPT ![i] = Nil]
                // /\ votesResponded' = [votesResponded EXCEPT ![i] = {}]
                // /\ votesGranted'   = [votesGranted EXCEPT ![i] = {}]
                // /\ voterLog'       = [voterLog EXCEPT ![i] = [j \in {} |-> <<>>]]
                remove voted_for -= [ i => let _];
                remove votes_responded -= [ i => let _];
                remove votes_granted -= [ i => let _];
                remove voter_log -= [ i => let _];

                add voted_for += [ i => None];
                add votes_responded += [ i => Set::empty()];
                add votes_granted += [ i => Set::empty()];
                add voter_log += [ i => Map::empty()];

                // /\ UNCHANGED <<messages, leaderVars, logVars>>                
            }
        }

        #[inductive(request_vote)]
        fn request_vote_inductive(pre: Self, post: Self, i:nat,j:nat) {
        }

        transition!{
            request_vote(i:nat,j:nat){
                // /\ state[i] = Candidate
                have state >= [i  => ServerState::Candidate]; 

                // /\ j \notin votesResponded[i]
                have votes_responded >= [i  => let voted_for_me];
                require(! voted_for_me has j);

                // /\ Send([mtype         |-> RequestVoteRequest,
                //          mterm         |-> currentTerm[i],
                //          mlastLogTerm  |-> LastTerm(log[i]),
                //          mlastLogIndex |-> Len(log[i]),
                //          msource       |-> i,
                //          mdest         |-> j])
                have current_term >= [i  => let term];
                have log >= [j  =>  let current_log  ];
                let last_log_index = current_log.last();
                let last_log_term =last_log_index.term;
                let response =RaftMessage::RequestVoteRequest{
                    // TODO: confirm that last_log_index is correctly defined here
                     src:i , dest:j , term : term , last_log_index:current_log.len() , last_log_term 
                };

                add messages += {response}; 

                // /\ UNCHANGED <<serverVars, candidateVars, leaderVars, logVars>>
            }
        }

        #[inductive(become_leader)]
        fn become_leader_inductive(pre: Self, post: Self, i:nat) {
            assert forall | j: nat| 
                    i != j &&
                    pre.state.get(j) == Some(ServerState::Leader) 
                    // pre.state.get(y) == Some(ServerState::Leader)
                    implies #[trigger] pre.current_term.get(i) != #[trigger] pre.current_term.get(j) 
            by{

                // - a leader (j) still has its votes_granted, and we know its term
                // - j's votes_granted must be in the quorum for it to have become leader
                assert(pre.quorum.contains(pre.votes_granted.get(j).unwrap()));
                // - but then we also know that the votes for i are in the quorum
                assert(pre.quorum.contains(pre.votes_granted.get(i).unwrap()));
                // - for the same term we cannot have two nodes with vgranted in quorum because:
                //      - for the same term vgranted must be disjoint between each node
                //      - if vgrantedi and vgrantedj are disjoint then they cannot both be in
                //      quorum
                //  - this means that their terms are different (do this by assuming the terms are
                //  equal and arrive to a contradiction



            }
        }

        transition!{
            become_leader(i: nat){
                have log >= [ i => let i_log]; 
                have current_term >= [ i => let i_current_term];
                have voter_log >= [i => let i_voter_log];

                // /\ state[i] = Candidate
                remove state -= [i => ServerState::Candidate];

                // /\ votesGranted[i] \in Quorum
                have votes_granted >= [i => let i_votes_granted];
                require (pre.quorum.contains(i_votes_granted));

                // /\ state'      = [state EXCEPT ![i] = Leader]
                add state += [i => ServerState::Leader];

                // /\ nextIndex'  = [nextIndex EXCEPT ![i] =
                //                      [j \in Server |-> Len(log[i]) + 1]]
                remove next_index -= [i => let _];
                let i_next_index = Map::new(|j:nat| pre.servers has j ,|j:nat| i_log.len()) ;
                add next_index += [i => i_next_index];

                // /\ matchIndex' = [matchIndex EXCEPT ![i] =
                //                      [j \in Server |-> 0]]
                remove match_index -= [i => let _];
                let i_match_index = Map::new(|j:nat| pre.servers has j ,|j:nat| 0nat) ;
                add match_index += [i => i_match_index];

                // /\ elections'  = elections \cup
                //                      {[eterm     |-> currentTerm[i],
                //                        eleader   |-> i,
                //                        elog      |-> log[i],
                //                        evotes    |-> votesGranted[i],
                //                        evoterLog |-> voterLog[i]]}
                // let new_election_record = ElectionRecord{
                //     term : i_current_term,
                //     leader: i,
                //     log: i_log,
                //     votes: i_votes_granted,
                //     voter_log: i_voter_log,
                // };
                //
                // add elections += [i_current_term => new_election_record]by{
                    // assert forall |j:nat| 
                    //     j != i &&
                    //     pre.current_term.get(j) == Some(i_current_term)
                    //     implies 
                    //         #[trigger]pre.current_term.get(j) != Some(i_current_term)
                    //     by {
                    //
                    //     }
                    // assert(!pre.elections.contains_key(i_current_term));
                // }; 

                // /\ UNCHANGED <<messages, currentTerm, votedFor, candidateVars, logVars>>
            }
        }

        // -----------------------------------------------------------------------------------
        //                      Message-Based transitions
        // -----------------------------------------------------------------------------------

        // The TLA handles message receiving logic like this:
        //
        // Receive(m) ==
        //     LET i == m.mdest
        //         j == m.msource
        //     IN \* Any RPC with a newer term causes the recipient to advance
        //        \* its term first. Responses with stale terms are ignored.
        //        \/ UpdateTerm(i, j, m)
        //        \/ /\ m.mtype = RequestVoteRequest
        //           /\ HandleRequestVoteRequest(i, j, m)
        //        \/ /\ m.mtype = RequestVoteResponse
        //           /\ \/ DropStaleResponse(i, j, m)
        //              \/ HandleRequestVoteResponse(i, j, m)
        //        \/ /\ m.mtype = AppendEntriesRequest
        //           /\ HandleAppendEntriesRequest(i, j, m)
        //        \/ /\ m.mtype = AppendEntriesResponse
        //           /\ \/ DropStaleResponse(i, j, m)
        //              \/ HandleAppendEntriesResponse(i, j, m)
        //
        // Given that verus does not allow to call transitions inside transitions, these are split
        // into individual functions, with prerequisites that match those in TLA
        
        #[inductive(update_term)]
        fn update_term_inductive(pre: Self, post: Self, i: nat, m: RaftMessage) {
            // just enough to make verus understand that the rest of current_term is not affected,
            // knowing that ServerState of i becomes follower it is able to finish the proof automatically
            assert forall |x: nat| 
                x != i implies pre.current_term.get(x) == post.current_term.get(x)   
            by{};
        }

        transition!{
            // the update_term transition is triggered when *any* message arrives with a more
            // recent term. The message is not discarded so that it can be handled by the other
            // functions
            update_term(i:nat,m:RaftMessage){
                let term = match m{
                    RaftMessage::AppendEntriesRequest {
                        term,
                        ..
                    }=>term,
                    RaftMessage::AppendEntriesResponse {
                        term,
                        ..
                    }=>term,
                    RaftMessage::RequestVoteRequest {
                        term,
                        ..
                    }=>term,
                    RaftMessage::RequestVoteResponse {
                        term,
                        ..
                    }=>term,
                };
                remove current_term -= [i => let i_term  ];

                // /\ m.mterm > currentTerm[i]
                require(term > i_term);
                
                // /\ currentTerm'    = [currentTerm EXCEPT ![i] = m.mterm]
                add current_term += [i => term];

                // /\ state'          = [state       EXCEPT ![i] = Follower]
                remove state -= [ i => let _];
                add state += [ i => ServerState::Follower];

                // /\ votedFor'       = [votedFor    EXCEPT ![i] = Nil]
                remove voted_for -= [ i => let _];
                add voted_for += [ i => None];

                //    \* messages is unchanged so m can be processed further.
                // /\ UNCHANGED <<messages, candidateVars, leaderVars, logVars>>
            }
        }

        #[inductive(handle_request_vote_request)]
        fn handle_request_vote_request_inductive(pre: Self, post: Self, m:RaftMessage) { }

        transition!{
            handle_request_vote_request(m:RaftMessage){
                remove messages -= { m };
                require let  RaftMessage::RequestVoteRequest { src, dest, term, last_log_index, last_log_term } = m;

                have log >= [dest as nat =>  let current_log  ];
                let my_last_log_index = current_log.last();
                have current_term >= [dest as nat =>  let my_current_term  ];
                let my_last_log_term = my_last_log_index.term;
                remove voted_for -= [dest as nat =>  let i_voted_for  ];

                // LET logOk == \/ m.mlastLogTerm > LastTerm(log[i])
                //              \/ /\ m.mlastLogTerm = LastTerm(log[i])
                //                 /\ m.mlastLogIndex >= Len(log[i])
                let log_ok = last_log_term > my_last_log_term 
                        || (last_log_term == my_last_log_term && last_log_index >= current_log.len() );   

                //     grant == /\ m.mterm = currentTerm[i]
                //              /\ logOk
                //              /\ votedFor[i] \in {Nil, j}
                let grant = term == my_current_term
                        &&  log_ok
                        && (i_voted_for == None::<nat> || i_voted_for == Some(src as nat)); 

                // TODO: confirm this makes sense
                // IN /\ m.mterm <= currentTerm[i]
                require(term <= my_current_term);

                //    /\ \/ grant  /\ votedFor' = [votedFor EXCEPT ![i] = j]
                //       \/ ~grant /\ UNCHANGED votedFor
                let updated_voted_for = 
                match grant{
                    true => Some(src as nat),
                    false => i_voted_for
                };
                add voted_for += [dest as nat => updated_voted_for]; 

                //    /\ Reply([mtype        |-> RequestVoteResponse,
                //              mterm        |-> currentTerm[i],
                //              mvoteGranted |-> grant,
                //              \* mlog is used just for the `elections' history variable for
                //              \* the proof. It would not exist in a real implementation.
                //              mlog         |-> log[i],
                //              msource      |-> i,
                //              mdest        |-> j],
                //              m)
                let response =RaftMessage::RequestVoteResponse{
                    src : dest,
                    dest:src,
                    term: my_current_term as nat,
                    vote_granted : grant,
                };

                add messages += {response}; 
                //    /\ UNCHANGED <<state, currentTerm, candidateVars, leaderVars, logVars>>
            }
        }

       
        #[inductive(handle_request_vote_response)]
        fn handle_request_vote_response_inductive(pre: Self, post: Self, m:RaftMessage) { 
            // unpack destination 
            let msg = match m{
                RaftMessage::RequestVoteResponse{dest,src,vote_granted,..} => Some((dest,src,vote_granted)),
                _ => None
            };
            assert(msg.is_some());
            let (i,src,vote_granted) = msg.unwrap();

            // leader_has_quorum
            // assert forall |j:nat|
            //     i != j implies pre.votes_granted.get(j)  =~= post.votes_granted.get(j)
            // by{};

            // forall |i:nat|
            //     self.state.get(i) == Some(ServerState::Leader)
            //     ==> #[trigger]self.quorum.contains(self.votes_granted.get(i).unwrap()) 

            assert forall |j:nat|
                if i != j{
                    pre.votes_granted.get(j)  =~= post.votes_granted.get(j) //&&
                    // pre.state.get(j) == post.state.get(j)

                }
                else{
                    if pre.state.get(i) == Some(ServerState::Leader){
                        pre.quorum.contains(pre.votes_granted.get(i).unwrap()) &&
                        if vote_granted{
                            post.votes_granted.get(i).unwrap() == pre.votes_granted.get(i).unwrap().insert(src) &&
                                true
                            //TODO: go back here and check it works now
                            // pre.servers.contains(src)
                            // pre.quorum.contains(pre.votes_granted.get(i).unwrap().insert(src))
                        }
                        else{
                            post.votes_granted.get(i).unwrap() == pre.votes_granted.get(i).unwrap()
                        }
                        
                    }
                    else{
                        true
                    }
                }
            by{};
        }
       
        transition!{
            handle_request_vote_response(m:RaftMessage){
                // // /\ Discard(m)
                remove messages -= { m };
                require let  RaftMessage::RequestVoteResponse { src, dest, term, vote_granted } = m;

                have current_term >= [dest as nat =>  let my_current_term  ];

                // \* This tallies votes even when the current state is not Candidate, but
                // \* they won't be looked at, so it doesn't matter.

                // /\ m.mterm = currentTerm[i]
                require(term == my_current_term);

                // /\ votesResponded' = [votesResponded EXCEPT ![i] =
                //                           votesResponded[i] \cup {j}]
                remove votes_responded -= [dest as nat=> let dest_votes_responded];
                add votes_responded += [ dest as nat=> (dest_votes_responded.insert(src as nat))];

                // /\ \/ /\ m.mvoteGranted
                //       /\ votesGranted' = [votesGranted EXCEPT ![i] =
                //                               votesGranted[i] \cup {j}]
                remove votes_granted -= [dest as nat => let mut dest_votes_granted];
                let new_dest_votes_granted =  match vote_granted {
                    true => {dest_votes_granted.insert(src as nat)}
                    false => {dest_votes_granted}
                };
                add votes_granted += [dest as nat => new_dest_votes_granted ];

                //       /\ voterLog' = [voterLog EXCEPT ![i] =
                //                           voterLog[i] @@ (j :> m.mlog)]
                //    \/ /\ ~m.mvoteGranted
                //       /\ UNCHANGED <<votesGranted, voterLog>>
                remove voter_log -= [dest as nat => let mut dest_voter_log];
                let new_dest_voter_log =  match vote_granted {
                    // TODO: need to add message mlog
                    true => {dest_voter_log.insert(src as nat,Seq::empty() )}
                    false => {dest_voter_log}
                };
                add voter_log += [dest as nat => new_dest_voter_log ];

                // /\ UNCHANGED <<serverVars, votedFor, leaderVars, logVars>>
            }
        }




        // transition!{
        //     client_request(){
        //         // remove messages -= [ msg_id => let  RaftMessage::ClientRequest{dest,value}];
        //         remove messages -= set { m };
        //         require let  RaftMessage::ClientRequest{dest,value} = m;
        //
        //         have current_term >= [dest as nat => let term];
        //
        //         // /\ state[i] = Leader
        //         have state >= [dest as nat => ServerState::Leader];
        //
        //         // /\ LET entry == [term  |-> currentTerm[i],
        //         //                  value |-> v]
        //         //        newLog == Append(log[i], entry)
        //         //    IN  log' = [log EXCEPT ![i] = newLog]
        //         remove log -= [dest as nat =>  let current_log  ];
        //         add log += [dest as nat => { current_log.push(LogEntry::{term : term as nat ,item:value}) }];
        //
        //         // /\ UNCHANGED <<messages, serverVars, candidateVars,
        //         //                leaderVars, commitIndex>>
        //     }
        // }
        //

        // transition!{
        //     duplicate_message(m:RaftMessage){
        //
        //     }
        // }
        //
        // transition!{
        //     drop_message(m:RaftMessage){
        //
        //     }
        // }
    }
}

} // verus!
