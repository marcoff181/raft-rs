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

tokenized_state_machine!{
    RAFT{
        fields{
            #[sharding(constant)]
            pub num_of_server : nat,

            #[sharding(map)]
            pub messages: Map<nat,RaftMessage>,

            #[sharding(variable)]
            pub servers: Set<nat>,

            #[sharding(set)]
            pub elections:Set<ElectionRecord>,

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

        // there can never be two leaders at the same time
        // #[invariant]
        // pub fn election_safety(&self) -> bool { 
        //     forall |i: nat, j: nat| 
        //             i != j && self.state.dom().contains(i) && self.state.dom().contains(j)
        //             ==> !(#[trigger] self.state.get(i) == Some(ServerState::Leader) && #[trigger] self.state.get(j) == Some(ServerState::Leader))
        // }

        #[invariant]
        pub fn messages_is_finite(&self) -> bool { 
            self.messages.dom().finite()
        }

        #[inductive(initialize)]
        fn initialize_inductive(post: Self, servers: Set<nat>) 
        {
        }

        init!{
            initialize(servers: Set<nat>)
            {
                let num_nodes = servers.len();
                require(num_nodes > 1);

                init num_of_server = num_nodes;
                init servers = servers;
                init messages = Map::empty();
                init elections = Set::empty();
                init allLogs = Map::empty();
                init current_term = Map::new(|i:nat| servers has i , |i:nat| 1);
                init state = Map::new(|i:nat| servers has i, |i:nat| ServerState::Follower);
                init voted_for = Map::new(|i:nat| servers has i, |i:nat| None);
                init log = Map::new(|i:nat| servers has i, |i:nat| Seq::empty());
                init commit_index = Map::new(|i:nat| servers has i, |i:nat| 0);
                init votes_responded = Map::empty();
                init votes_granted = Map::new(|i:nat| servers has i, |i:nat| Set::empty());
                init voter_log = Map::new(|i:nat| servers has i, |i:nat| Map::empty());
                init next_index = Map::new(|i:nat| servers has i, |i:nat| Map::new(|j:nat| servers has j, |j:nat| 1));
                init match_index = Map::new(|i:nat| servers has i, |i:nat| Map::new(|j:nat| servers has j, |j:nat| 0));
            }
        }

        #[inductive(restart)]
        fn restart_inductive(pre: Self, post: Self, i: nat) { }

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
        fn timeout_inductive(pre: Self, post: Self, i: nat) { }

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
                let msg =RaftMessage::RequestVoteRequest{
                    // TODO: confirm that last_log_index is correctly defined here
                     src:i , dest:j , term : term , last_log_index:current_log.len() , last_log_term 
                };

                // generate fresh id to insert messages
                let r = |a:nat,b:nat| a <= b;
                birds_eye let fresh_id = { 
                    if pre.messages.dom().is_empty(){1} 
                    else{
                        pre.messages.dom().find_unique_maximal(r) + 1
                    }
                };

                add messages += [fresh_id => msg] by {  
                    assert(pre.messages.dom().finite());
                    // TODO: confirm this is needed
                    assume(!pre.messages.dom().contains(fresh_id));
                };

                // /\ UNCHANGED <<serverVars, candidateVars, leaderVars, logVars>>
            }
        }

        #[inductive(become_leader)]
        fn become_leader_inductive(pre: Self, post: Self, src:nat) {
        }

        transition!{
            become_leader(src: nat){
                have log >= [ src as nat => let src_log]; 
                have current_term >= [ src as nat => let src_current_term];
                have voter_log >= [src as nat => let src_voter_log];

                // /\ state[i] = Candidate
                remove state -= [src as nat => ServerState::Candidate];

                // /\ votesGranted[i] \in Quorum
                have votes_granted >= [src as nat => let src_votes_granted];
                let threshold = pre.servers.len() / 2;  
                let quorum = Set::new(|s: Set<nat>|   
                    s.subset_of(pre.servers) && s.finite() && s.len() > threshold  
                );  
                require (quorum has src_votes_granted);

                // /\ state'      = [state EXCEPT ![i] = Leader]
                add state += [src as nat => ServerState::Leader];

                // /\ nextIndex'  = [nextIndex EXCEPT ![i] =
                //                      [j \in Server |-> Len(log[i]) + 1]]
                remove next_index -= [src as nat => let _];
                let src_next_index = Map::new(|j:nat| pre.servers has j ,|j:nat| src_log.len()) ;
                add next_index += [src as nat => src_next_index];

                // /\ matchIndex' = [matchIndex EXCEPT ![i] =
                //                      [j \in Server |-> 0]]
                remove match_index -= [src as nat => let _];
                let src_match_index = Map::new(|j:nat| pre.servers has j ,|j:nat| 0nat) ;
                add match_index += [src as nat => src_match_index];

                // /\ elections'  = elections \cup
                //                      {[eterm     |-> currentTerm[i],
                //                        eleader   |-> i,
                //                        elog      |-> log[i],
                //                        evotes    |-> votesGranted[i],
                //                        evoterLog |-> voterLog[i]]}
                let new_election_record = ElectionRecord{
                    term : src_current_term,
                    leader: src as nat,
                    log: src_log,
                    votes: src_votes_granted,
                    voter_log:src_voter_log,
                };

                // TODO: see if this can be removed
                remove elections -= set { new_election_record};
                add elections += set { new_election_record};

                // /\ UNCHANGED <<messages, currentTerm, votedFor, candidateVars, logVars>>
            }
        }

        #[inductive(handle_request_vote_request)]
        fn handle_request_vote_request_inductive(pre: Self, post: Self, msg_id:nat) { }

        transition!{
            handle_request_vote_request(msg_id:nat){
                // remove messages -= set { m };
                // require let  RaftMessage::RequestVoteRequest { src, dest, term, last_log_index, last_log_term } = m;
                remove messages -= [ msg_id => let
                RaftMessage::RequestVoteRequest {
                    src,
                    dest,
                    term,
                    last_log_index,
                    last_log_term,
                }];

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
                
                // generate fresh id to insert messages
                let r = |a:nat,b:nat| a <= b;
                birds_eye let fresh_id = { 
                    if pre.messages.dom().is_empty(){1} 
                    else{
                        pre.messages.dom().find_unique_maximal(r) + 1
                    }
                };

                add messages += [fresh_id => response] by {  
                    assert(pre.messages.dom().finite());
                    // TODO: confirm this is needed
                    assume(!pre.messages.dom().contains(fresh_id));
                };

                //    /\ UNCHANGED <<state, currentTerm, candidateVars, leaderVars, logVars>>
            }
        }

       
        #[inductive(handle_request_vote_response)]
        fn handle_request_vote_response_inductive(pre: Self, post: Self, msg_id:nat) { }
       
        transition!{
            handle_request_vote_response(msg_id:nat){
                // /\ Discard(m)
                remove messages -= [ msg_id => let RaftMessage::RequestVoteResponse { src, dest, term, vote_granted }];

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
