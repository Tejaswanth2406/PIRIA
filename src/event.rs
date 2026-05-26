/// Event Model — §4 of PIRIA MVP Spec
///
/// PIRIA uses event sourcing as the ground truth for all state changes.
/// The event log is immutable — events are never deleted or modified.
/// Current ORB state = fold over event history.
///
/// Guarantees:
///   - Causal ordering via HLC timestamps
///   - Tamper-evidence via SHA-256 hash chain
///   - Authenticity via Ed25519 signatures
///   - Concurrency detection via vector clocks
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::{
    clock::{HlcTimestamp, VectorClock},
    error::{PiriaError, Result},
    orb::SemanticState,
};

// ── Event Types ───────────────────────────────────────────────────────────────

/// All possible event types in the PIRIA runtime.
/// Each variant carries a typed payload. §4.2
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum EventPayload {
    /// New ORB instantiated.
    OrbCreated {
        initial_trust: f32,
        permissions_bits: u64,
        probability_categories: usize,
    },

    /// Semantic state transition.
    OrbStateChanged {
        from_state: SemanticState,
        to_state: SemanticState,
    },

    /// Embedding vector refreshed.
    VectorUpdated {
        /// SHA-256 of the new vector bytes (not the vector itself — too large for log).
        new_vector_hash: [u8; 32],
        drift_distance: f32,
    },

    /// Edge added to graph.
    RelationAdded {
        target_orb_id: Uuid,
        relation_type: String,
        weight: f32,
        edge_id: Uuid,
    },

    /// Edge removed from graph.
    RelationRemoved {
        edge_id: Uuid,
        target_orb_id: Uuid,
    },

    /// Trust score adjusted.
    TrustUpdated {
        previous_score: f32,
        new_score: f32,
        evidence_type: String,
        observation_weight: f32,
        outcome: f32,
    },

    /// External input ingested.
    ObservationRecorded {
        source_id: Uuid,
        content_hash: [u8; 32],
        observation_type: String,
        observation_quality: f32,
    },

    /// Probability field updated (belief revised).
    BeliefRevised {
        /// New alpha vector snapshot.
        new_alpha: Vec<f32>,
        previous_entropy: f32,
        new_entropy: f32,
    },

    /// Invariant breach detected.
    ConstraintViolated {
        invariant_id: String,
        severity: ViolationSeverity,
        context: String,
    },

    /// Two ORBs merged into one.
    OrbMerged {
        source_orb_id: Uuid,
        merge_strategy: MergeStrategy,
    },

    /// ORB split into two.
    OrbSplit {
        child_orb_ids: [Uuid; 2],
        split_strategy: String,
    },

    /// History compressed (dormant ORB checkpoint).
    OrbCompressed {
        compression_policy: String,
        checkpoint_hash: [u8; 32],
        events_compressed: u32,
    },

    /// ORB moved to archival storage.
    OrbArchived {
        archive_key: String,
    },

    /// Network partition detected on this node.
    PartitionDetected {
        node_id: String,
        vector_clock_snapshot: VectorClock,
    },

    /// Partition healed; merge summary attached.
    PartitionHealed {
        partition_event_id: Uuid,
        merge_summary: String,
        conflicting_orbs: Vec<Uuid>,
    },

    /// Entropy spike detected in field.
    EntropySpike {
        field_entropy: f32,
        threshold: f32,
        active_orb_count: u64,
    },
}

/// Severity levels for constraint violations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ViolationSeverity {
    Warning,
    Error,
    Critical,
}

/// Merge strategies for ORB_MERGED events.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MergeStrategy {
    /// Take the ORB with higher trust as canonical.
    TrustWeighted,
    /// Take the most recently updated ORB.
    LastWriteWins,
    /// Merge probability fields; average semantic vectors.
    Blend,
}

// ── Event ─────────────────────────────────────────────────────────────────────

/// A single immutable event in the PIRIA event log. §4.1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    // ── Identity ─────────────────────────────────────────────────────
    /// Time-ordered, globally unique.
    pub event_id: Uuid,

    // ── Target ───────────────────────────────────────────────────────
    /// Target ORB — must exist at event time.
    pub orb_id: Uuid,

    /// Originating ORB or system process.
    pub actor_id: ActorId,

    // ── Causal ordering ───────────────────────────────────────────────
    /// Wall clock at originating node (Unix ms).
    pub timestamp_physical: u64,

    /// HLC timestamp for causal ordering across nodes.
    pub timestamp_logical: HlcTimestamp,

    /// Per-node sequence counters for concurrency detection.
    pub vector_clock: VectorClock,

    // ── Payload ───────────────────────────────────────────────────────
    pub payload: EventPayload,

    // ── Hash chain ────────────────────────────────────────────────────
    /// SHA-256 of the previous event in this ORB's log.
    /// [0u8; 32] for the first event.
    pub prev_event_hash: [u8; 32],

    /// SHA-256 of (event_id || payload_bytes || prev_event_hash).
    pub event_hash: [u8; 32],

    // ── Authenticity ──────────────────────────────────────────────────
    /// Ed25519 signature over event_hash.
    pub signature: Option<Vec<u8>>,
}

/// Actor identity — either a specific ORB or the system process.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ActorId {
    Orb(Uuid),
    System,
}

impl Event {
    /// Compute the canonical hash of this event:
    /// SHA-256(event_id || CBOR(payload) || prev_event_hash)
    pub fn compute_hash(
        event_id: &Uuid,
        payload: &EventPayload,
        prev_event_hash: &[u8; 32],
    ) -> Result<[u8; 32]> {
        let payload_bytes = serde_json::to_vec(payload)
            .map_err(PiriaError::Serialization)?;

        let mut hasher = Sha256::new();
        hasher.update(event_id.as_bytes());
        hasher.update(&payload_bytes);
        hasher.update(prev_event_hash);
        Ok(hasher.finalize().into())
    }

    /// Sign the event's hash with the provided signing key.
    /// Stores the 64-byte signature as `self.signature`.
    pub fn sign(&mut self, key: &SigningKey) {
        let sig: Signature = key.sign(&self.event_hash);
        self.signature = Some(sig.to_bytes().to_vec());
    }

    /// Verify the signature against a known verifying key.
    pub fn verify_signature(&self, key: &VerifyingKey) -> Result<()> {
        let sig_bytes = self
            .signature
            .as_ref()
            .ok_or_else(|| PiriaError::SignatureInvalid(self.event_id))?;

        let sig_arr: [u8; 64] = sig_bytes
            .as_slice()
            .try_into()
            .map_err(|_| PiriaError::SignatureInvalid(self.event_id))?;

        let sig = Signature::from_bytes(&sig_arr);
        key.verify(&self.event_hash, &sig)
            .map_err(|_| PiriaError::SignatureInvalid(self.event_id))
    }

    /// Verify the event hash is correct (tamper detection without a key).
    pub fn verify_hash(&self) -> Result<()> {
        let computed = Self::compute_hash(&self.event_id, &self.payload, &self.prev_event_hash)?;
        if computed != self.event_hash {
            return Err(PiriaError::HashChainBroken {
                event_id: self.event_id,
                expected: hex::encode(self.event_hash),
                actual: hex::encode(computed),
            });
        }
        Ok(())
    }
}

// ── Event Builder ─────────────────────────────────────────────────────────────

/// Builds and optionally signs events.
pub struct EventBuilder {
    signing_key: Option<SigningKey>,
    node_id: String,
}

impl EventBuilder {
    pub fn new(node_id: impl Into<String>) -> Self {
        Self { signing_key: None, node_id: node_id.into() }
    }

    pub fn with_signing_key(mut self, key: SigningKey) -> Self {
        self.signing_key = Some(key);
        self
    }

    /// Build a new event and optionally sign it.
    pub fn build(
        &self,
        orb_id: Uuid,
        actor_id: ActorId,
        payload: EventPayload,
        timestamp_logical: HlcTimestamp,
        mut vector_clock: VectorClock,
        prev_event_hash: [u8; 32],
    ) -> Result<Event> {
        vector_clock.tick(&self.node_id);
        let event_id = Uuid::now_v7();
        let event_hash = Event::compute_hash(&event_id, &payload, &prev_event_hash)?;

        let mut event = Event {
            event_id,
            orb_id,
            actor_id,
            timestamp_physical: crate::orb::now_ms(),
            timestamp_logical,
            vector_clock,
            payload,
            prev_event_hash,
            event_hash,
            signature: None,
        };

        if let Some(key) = &self.signing_key {
            event.sign(key);
        }

        Ok(event)
    }
}

// ── Event Log ────────────────────────────────────────────────────────────────

/// In-memory event log for a single ORB — forms an append-only hash chain.
///
/// Production deployment: backed by Kafka (§11.1). This implementation
/// provides the core chain logic and can be wrapped by any persistent store.
#[derive(Debug, Clone)]
pub struct OrbEventLog {
    pub orb_id: Uuid,
    events: Vec<Event>,
}

impl OrbEventLog {
    pub fn new(orb_id: Uuid) -> Self {
        Self { orb_id, events: Vec::new() }
    }

    /// The hash of the most recent event, or [0u8; 32] for an empty log.
    pub fn head_hash(&self) -> [u8; 32] {
        self.events.last().map(|e| e.event_hash).unwrap_or([0u8; 32])
    }

    /// Append an event. Validates:
    ///   1. Event targets this ORB
    ///   2. prev_event_hash matches head
    ///   3. event_hash is correct
    pub fn append(&mut self, event: Event) -> Result<()> {
        if event.orb_id != self.orb_id {
            return Err(PiriaError::Internal(format!(
                "Event targets orb {} but log is for orb {}",
                event.orb_id, self.orb_id
            )));
        }

        let expected_prev = self.head_hash();
        if event.prev_event_hash != expected_prev {
            return Err(PiriaError::HashChainBroken {
                event_id: event.event_id,
                expected: hex::encode(expected_prev),
                actual: hex::encode(event.prev_event_hash),
            });
        }

        event.verify_hash()?;
        self.events.push(event);
        Ok(())
    }

    /// Verify the entire chain from genesis. O(n).
    pub fn verify_chain(&self) -> Result<()> {
        let mut expected_prev = [0u8; 32];
        for event in &self.events {
            if event.prev_event_hash != expected_prev {
                return Err(PiriaError::HashChainBroken {
                    event_id: event.event_id,
                    expected: hex::encode(expected_prev),
                    actual: hex::encode(event.prev_event_hash),
                });
            }
            event.verify_hash()?;
            expected_prev = event.event_hash;
        }
        Ok(())
    }

    /// Verify all signatures against a known verifying key.
    pub fn verify_signatures(&self, key: &VerifyingKey) -> Result<()> {
        for event in &self.events {
            event.verify_signature(key)?;
        }
        Ok(())
    }

    /// Number of events in the log.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Iterate events in causal order (oldest first).
    pub fn iter(&self) -> impl Iterator<Item = &Event> {
        self.events.iter()
    }

    /// Events since (exclusive) a given HLC timestamp, in causal order.
    pub fn events_since(&self, since: HlcTimestamp) -> impl Iterator<Item = &Event> {
        self.events.iter().filter(move |e| e.timestamp_logical > since)
    }

    /// Reconstruct the causal chain leading to a specific event.
    /// Returns events in causal order (oldest first), up to and including target.
    pub fn causal_chain_to(&self, target_event_id: Uuid) -> Vec<&Event> {
        let mut chain = Vec::new();
        for event in &self.events {
            chain.push(event);
            if event.event_id == target_event_id {
                break;
            }
        }
        chain
    }

    /// Latest event of a specific payload type.
    pub fn latest_of_type<F>(&self, predicate: F) -> Option<&Event>
    where
        F: Fn(&EventPayload) -> bool,
    {
        self.events.iter().rev().find(|e| predicate(&e.payload))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;

    fn make_signing_key() -> SigningKey {
        SigningKey::generate(&mut OsRng)
    }

    fn make_builder(key: SigningKey) -> EventBuilder {
        EventBuilder::new("node-test").with_signing_key(key)
    }

    fn make_event(
        builder: &EventBuilder,
        orb_id: Uuid,
        prev_hash: [u8; 32],
    ) -> Event {
        builder
            .build(
                orb_id,
                ActorId::System,
                EventPayload::OrbCreated {
                    initial_trust: 0.095,
                    permissions_bits: 0x037,
                    probability_categories: 4,
                },
                HlcTimestamp::now(),
                VectorClock::new(),
                prev_hash,
            )
            .unwrap()
    }

    #[test]
    fn event_hash_correct() {
        let key = make_signing_key();
        let builder = make_builder(key);
        let orb_id = Uuid::now_v7();
        let event = make_event(&builder, orb_id, [0u8; 32]);
        event.verify_hash().unwrap();
    }

    #[test]
    fn event_signature_valid() {
        let key = make_signing_key();
        let vk = key.verifying_key();
        let builder = make_builder(key);
        let orb_id = Uuid::now_v7();
        let event = make_event(&builder, orb_id, [0u8; 32]);
        event.verify_signature(&vk).unwrap();
    }

    #[test]
    fn hash_chain_append_and_verify() {
        let key = make_signing_key();
        let builder = make_builder(key);
        let orb_id = Uuid::now_v7();
        let mut log = OrbEventLog::new(orb_id);

        let e1 = make_event(&builder, orb_id, [0u8; 32]);
        let e2 = make_event(&builder, orb_id, e1.event_hash);
        let e3 = make_event(&builder, orb_id, e2.event_hash);

        log.append(e1).unwrap();
        log.append(e2).unwrap();
        log.append(e3).unwrap();

        log.verify_chain().unwrap();
        assert_eq!(log.len(), 3);
    }

    #[test]
    fn hash_chain_rejects_broken_link() {
        let key = make_signing_key();
        let builder = make_builder(key);
        let orb_id = Uuid::now_v7();
        let mut log = OrbEventLog::new(orb_id);

        let e1 = make_event(&builder, orb_id, [0u8; 32]);
        // e2 uses wrong prev_hash (not e1.event_hash)
        let e2 = make_event(&builder, orb_id, [0xFFu8; 32]);

        log.append(e1).unwrap();
        let result = log.append(e2);
        assert!(matches!(result, Err(PiriaError::HashChainBroken { .. })));
    }

    #[test]
    fn tampered_event_detected() {
        let key = make_signing_key();
        let builder = make_builder(key);
        let orb_id = Uuid::now_v7();
        let mut event = make_event(&builder, orb_id, [0u8; 32]);

        // Tamper with payload after signing
        event.payload = EventPayload::OrbArchived { archive_key: "tampered".into() };

        assert!(event.verify_hash().is_err());
    }

    #[test]
    fn all_signatures_verified_on_chain() {
        let key = make_signing_key();
        let vk = key.verifying_key();
        let builder = make_builder(key);
        let orb_id = Uuid::now_v7();
        let mut log = OrbEventLog::new(orb_id);

        let e1 = make_event(&builder, orb_id, [0u8; 32]);
        let e2 = make_event(&builder, orb_id, e1.event_hash);
        log.append(e1).unwrap();
        log.append(e2).unwrap();

        log.verify_signatures(&vk).unwrap();
    }
}