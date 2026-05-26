use thiserror::Error;
use uuid::Uuid;

#[derive(Debug, Error)]
pub enum PiriaError {
    // ── ORB errors ─────────────────────────────────────────────────────
    #[error("ORB not found: {0}")]
    OrbNotFound(Uuid),

    #[error("ORB already exists: {0}")]
    OrbAlreadyExists(Uuid),

    #[error("Invalid ORB state transition: {from:?} → {to:?}")]
    InvalidStateTransition {
        from: crate::orb::SemanticState,
        to: crate::orb::SemanticState,
    },

    #[error("ORB degree budget exceeded: orb={orb_id}, budget={budget}")]
    DegreeBudgetExceeded { orb_id: Uuid, budget: usize },

    // ── Vector errors ──────────────────────────────────────────────────
    #[error("Semantic vector has wrong dimension: expected {expected}, got {got}")]
    VectorDimensionMismatch { expected: usize, got: usize },

    #[error("Semantic vector is not L2-normalized (norm={norm:.4})")]
    VectorNotNormalized { norm: f32 },

    // ── Event log errors ───────────────────────────────────────────────
    #[error("Event hash chain broken at event {event_id}: expected {expected}, got {actual}")]
    HashChainBroken {
        event_id: Uuid,
        expected: String,
        actual: String,
    },

    #[error("Event signature verification failed for event {0}")]
    SignatureInvalid(Uuid),

    #[error("Causal ordering violation: {cause} must precede {effect}")]
    CausalOrderingViolation { cause: Uuid, effect: Uuid },

    #[error("HLC clock skew exceeded maximum ({max_ms}ms): skew={skew_ms}ms")]
    ClockSkewExceeded { max_ms: u64, skew_ms: u64 },

    // ── Trust errors ───────────────────────────────────────────────────
    #[error("Trust ceiling violated: trust={trust:.4} > ceiling={ceiling:.4} for orb={orb_id}")]
    TrustCeilingViolated {
        orb_id: Uuid,
        trust: f32,
        ceiling: f32,
    },

    #[error("Trust concentration cap exceeded: orb={orb_id} holds {share:.2}% of total trust")]
    TrustConcentrationCapExceeded { orb_id: Uuid, share: f32 },

    // ── Invariant errors ───────────────────────────────────────────────
    #[error("Invariant registry signature invalid for key={key}")]
    InvariantSignatureInvalid { key: String },

    #[error("Invariant violation: {invariant_id} — {message}")]
    InvariantViolation {
        invariant_id: String,
        message: String,
    },

    #[error("Attempted write to read-only invariant registry")]
    InvariantRegistryReadOnly,

    // ── Entropy errors ─────────────────────────────────────────────────
    #[error("Entropy spike detected: field_entropy={value:.4} > threshold={threshold:.4}")]
    EntropySpike { value: f32, threshold: f32 },

    #[error("Probability field invalid: categories={categories}, constraint violation")]
    InvalidProbabilityField { categories: usize },

    // ── Attention errors ───────────────────────────────────────────────
    #[error("Attention budget exhausted during traversal")]
    AttentionBudgetExhausted,

    #[error("Traversal depth limit reached: max={max}")]
    TraversalDepthLimitReached { max: usize },

    // ── Integrity errors ───────────────────────────────────────────────
    #[error("Integrity hash mismatch for orb {orb_id}: stored={stored}, computed={computed}")]
    IntegrityHashMismatch {
        orb_id: Uuid,
        stored: String,
        computed: String,
    },

    // ── CRDT errors ────────────────────────────────────────────────────
    #[error("CRDT merge conflict on orb {orb_id}: {reason}")]
    CrdtMergeConflict { orb_id: Uuid, reason: String },

    #[error("Partition state incompatible for merge: {reason}")]
    PartitionMergeIncompatible { reason: String },

    // ── Serialization ──────────────────────────────────────────────────
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("CBOR serialization error: {0}")]
    CborSerialization(String),

    // ── Generic ────────────────────────────────────────────────────────
    #[error("Internal error: {0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, PiriaError>;