/// ORB — Ontological Relational Block
///
/// The atomic unit of PIRIA. Every entity in the system is an ORB.
/// All fields are typed, bounded, and validated.
///
/// Fixes over original:
///   - integrity_hash now covers relations, probability_field, timestamps, event_refs
///   - semantic_mass definition tightened to match spec §3.2
///   - TrustRecord::update fills in orb_id on error (was Uuid::nil())
///   - prune_lowest_weight_relation uses LRU as documented tiebreaker
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::{
    clock::HlcTimestamp,
    error::{PiriaError, Result},
};

// ── Constants ────────────────────────────────────────────────────────────────

/// Embedding dimensionality (matches text-embedding-3-large and nomic-embed-text)
pub const VECTOR_DIM: usize = 1536;

/// Maximum outgoing relation edges per ORB. Enforces graph sparsity — §5.3.
pub const DEFAULT_DEGREE_BUDGET: usize = 512;

/// Maximum Dirichlet categories for probability field.
pub const MAX_PROBABILITY_CATEGORIES: usize = 64;

/// Trust decay rate λ per day (λ = 0.001 → half-life ≈ 693 days)
pub const TRUST_DECAY_RATE_PER_DAY: f64 = 0.001;

/// Trust evidence growth constant k: ceiling(e) = 1 - exp(-k·e)
pub const TRUST_EVIDENCE_CONSTANT: f64 = 0.1;

/// Max fraction of total trust budget any single ORB may hold.
pub const TRUST_CONCENTRATION_CAP: f32 = 0.05;

/// Global trust floor — no ORB may have T(x) < this value.
pub const TRUST_FLOOR: f32 = 0.01;

/// Semantic drift alert threshold: ||v(t) - v(t₀)||₂ > this value.
pub const DRIFT_THRESHOLD: f32 = 0.15;

// ── Semantic State ────────────────────────────────────────────────────────────

/// The lifecycle state of an ORB.
///
/// Valid transitions (enforced by validate_state_transition):
///   Active  →  Dormant | Compressed | Archived
///   Dormant →  Active  | Compressed | Archived
///   Compressed → Archived
///   Archived → (terminal — no transitions out)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticState {
    Active,
    Dormant,
    Compressed,
    Archived,
}

impl SemanticState {
    pub fn is_terminal(&self) -> bool {
        matches!(self, SemanticState::Archived)
    }

    pub fn can_transition_to(&self, next: &SemanticState) -> bool {
        matches!(
            (self, next),
            (SemanticState::Active, SemanticState::Dormant)
                | (SemanticState::Active, SemanticState::Compressed)
                | (SemanticState::Active, SemanticState::Archived)
                | (SemanticState::Dormant, SemanticState::Active)
                | (SemanticState::Dormant, SemanticState::Compressed)
                | (SemanticState::Dormant, SemanticState::Archived)
                | (SemanticState::Compressed, SemanticState::Archived)
        )
    }
}

// ── Semantic Vector ───────────────────────────────────────────────────────────

/// L2-normalized semantic embedding vector.
/// Invariant: ||v||₂ = 1.0 (enforced at construction).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticVector(pub Vec<f32>);

impl SemanticVector {
    /// Construct from raw values, validating dimension and normalizing.
    pub fn new(raw: Vec<f32>) -> Result<Self> {
        if raw.len() != VECTOR_DIM {
            return Err(PiriaError::VectorDimensionMismatch {
                expected: VECTOR_DIM,
                got: raw.len(),
            });
        }
        let norm = l2_norm(&raw);
        if norm < 1e-8 {
            return Err(PiriaError::VectorNotNormalized { norm });
        }
        let normalized = raw.iter().map(|&x| x / norm).collect();
        Ok(Self(normalized))
    }

    /// Construct from already-normalized values (skips normalization, validates dim + norm).
    pub fn from_normalized(v: Vec<f32>) -> Result<Self> {
        if v.len() != VECTOR_DIM {
            return Err(PiriaError::VectorDimensionMismatch {
                expected: VECTOR_DIM,
                got: v.len(),
            });
        }
        let norm = l2_norm(&v);
        if (norm - 1.0).abs() > 1e-4 {
            return Err(PiriaError::VectorNotNormalized { norm });
        }
        Ok(Self(v))
    }

    /// Cosine similarity (= dot product after L2 normalization). Returns ∈ [-1, 1].
    pub fn cosine_similarity(&self, other: &SemanticVector) -> f32 {
        self.0.iter().zip(other.0.iter()).map(|(a, b)| a * b).sum()
    }

    /// L2 distance between two normalized vectors.
    /// For unit vectors: ||a-b||₂ = sqrt(2 - 2·cosine(a,b))
    pub fn l2_distance(&self, other: &SemanticVector) -> f32 {
        let cos = self.cosine_similarity(other);
        (2.0_f32 - 2.0 * cos).max(0.0).sqrt()
    }

    #[cfg(test)]
    pub fn test_random() -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let raw: Vec<f32> = (0..VECTOR_DIM).map(|_| rng.gen::<f32>() - 0.5).collect();
        Self::new(raw).unwrap()
    }
}

pub fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

// ── Probability Field ─────────────────────────────────────────────────────────

/// Dirichlet-parameterized probability field.
/// α[i] > 0 for all i. Implied mean: p[i] = α[i] / Σα.
/// Entropy approximation via normalized Dirichlet mean (MVP).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbabilityField {
    /// Dirichlet concentration parameters. All must be > 0.
    pub alpha: Vec<f32>,
}

impl ProbabilityField {
    pub fn new(alpha: Vec<f32>) -> Result<Self> {
        if alpha.is_empty() || alpha.len() > MAX_PROBABILITY_CATEGORIES {
            return Err(PiriaError::InvalidProbabilityField { categories: alpha.len() });
        }
        if alpha.iter().any(|&a| a <= 0.0 || !a.is_finite()) {
            return Err(PiriaError::InvalidProbabilityField { categories: alpha.len() });
        }
        Ok(Self { alpha })
    }

    /// Uniform distribution over n categories.
    pub fn uniform(n: usize) -> Result<Self> {
        if n == 0 || n > MAX_PROBABILITY_CATEGORIES {
            return Err(PiriaError::InvalidProbabilityField { categories: n });
        }
        Ok(Self { alpha: vec![1.0; n] })
    }

    /// Implied mean probabilities: p[i] = α[i] / Σα
    pub fn mean_probs(&self) -> Vec<f32> {
        let sum: f32 = self.alpha.iter().sum();
        self.alpha.iter().map(|&a| a / sum).collect()
    }

    /// Shannon entropy of the mean distribution (bits): H = -Σ p[i] · log₂(p[i])
    pub fn entropy_bits(&self) -> f32 {
        let probs = self.mean_probs();
        -probs
            .iter()
            .filter(|&&p| p > 1e-10)
            .map(|&p| p * p.log2())
            .sum::<f32>()
    }

    /// Maximum possible entropy for this field (uniform distribution).
    pub fn max_entropy_bits(&self) -> f32 {
        (self.alpha.len() as f32).log2()
    }

    /// Normalized entropy ∈ [0, 1].
    pub fn normalized_entropy(&self) -> f32 {
        let max = self.max_entropy_bits();
        if max < 1e-8 {
            return 0.0;
        }
        self.entropy_bits() / max
    }

    /// Number of categories.
    pub fn categories(&self) -> usize {
        self.alpha.len()
    }

    /// Bayesian update: obs[i] = count of observations in category i.
    pub fn observe(&mut self, observations: &[f32]) -> Result<()> {
        if observations.len() != self.alpha.len() {
            return Err(PiriaError::InvalidProbabilityField {
                categories: observations.len(),
            });
        }
        for (a, &obs) in self.alpha.iter_mut().zip(observations.iter()) {
            if obs < 0.0 {
                return Err(PiriaError::InvalidProbabilityField {
                    categories: self.alpha.len(),
                });
            }
            *a += obs;
        }
        Ok(())
    }

    /// Merge two probability fields by summing alpha vectors (correct Dirichlet pooling).
    /// Fields must have the same number of categories.
    pub fn merge(&self, other: &ProbabilityField) -> Result<ProbabilityField> {
        if self.alpha.len() != other.alpha.len() {
            return Err(PiriaError::InvalidProbabilityField {
                categories: other.alpha.len(),
            });
        }
        let merged: Vec<f32> = self.alpha.iter().zip(&other.alpha).map(|(a, b)| a + b).collect();
        ProbabilityField::new(merged)
    }
}

// ── Trust Record ──────────────────────────────────────────────────────────────

/// Trust record for an ORB — score + evidence + decay tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustRecord {
    /// Current trust score ∈ [TRUST_FLOOR, ceiling(evidence_count)].
    pub score: f32,
    /// Total evidence count — monotone increasing.
    pub evidence_count: u32,
    /// Unix ms of last update (for decay computation).
    pub last_updated_ms: u64,
}

impl TrustRecord {
    pub fn initial() -> Self {
        Self {
            score: 0.095, // f(1) = 1 - exp(-0.1·1) ≈ 0.095
            evidence_count: 1,
            last_updated_ms: now_ms(),
        }
    }

    /// Trust ceiling: f(e) = 1 - exp(-k·e)
    pub fn ceiling(&self) -> f32 {
        (1.0 - (-TRUST_EVIDENCE_CONSTANT * self.evidence_count as f64).exp()) as f32
    }

    /// Effective trust after temporal decay: T(t) = T(t₀) · exp(-λ·Δdays)
    /// Floor-clamped to TRUST_FLOOR.
    pub fn effective_score(&self) -> f32 {
        let now = now_ms();
        let delta_days = (now.saturating_sub(self.last_updated_ms)) as f64 / 86_400_000.0;
        let decayed = self.score as f64 * (-TRUST_DECAY_RATE_PER_DAY * delta_days).exp();
        (decayed as f32).max(TRUST_FLOOR)
    }

    /// Bayesian update: T_new = T_old + α·(obs_weight·outcome - T_old)
    /// Enforces ceiling invariant: T_new ≤ ceiling(evidence_count + 1)
    /// Enforces floor:            T_new ≥ TRUST_FLOOR
    pub fn update(
        &mut self,
        observation_weight: f32,
        outcome: f32,
        learning_rate: f32,
        orb_id: Uuid,
    ) -> Result<()> {
        let current = self.effective_score();
        let delta = learning_rate * (observation_weight * outcome - current);
        let candidate = (current + delta).clamp(TRUST_FLOOR, 1.0);

        self.evidence_count = self.evidence_count.saturating_add(1);
        let new_ceiling = self.ceiling();

        if candidate > new_ceiling + 1e-5 {
            return Err(PiriaError::TrustCeilingViolated {
                orb_id,
                trust: candidate,
                ceiling: new_ceiling,
            });
        }

        self.score = candidate.min(new_ceiling);
        self.last_updated_ms = now_ms();
        Ok(())
    }
}

pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

// ── Relation Ref ──────────────────────────────────────────────────────────────

/// Typed relation between two ORBs. Edges are first-class objects.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RelationType {
    Causes,
    Supports,
    Contradicts,
    Contains,
    References,
    SimilarTo,
    DerivedFrom,
    TemporalPrecedes,
    TrustDelegatedTo,
    AttentionFlowsTo,
}

impl RelationType {
    /// True for semantically symmetric relation types.
    pub fn is_symmetric(&self) -> bool {
        matches!(self, RelationType::Contradicts | RelationType::SimilarTo)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationRef {
    pub edge_id: Uuid,
    pub target_orb_id: Uuid,
    pub relation_type: RelationType,
    /// Relational strength ∈ [0, 1].
    pub weight: f32,
    /// Trust score inherited from source ORB at creation time.
    pub trust_at_creation: f32,
    pub created_at: HlcTimestamp,
    /// Unix ms of last traversal (for LRU pruning).
    pub last_used_ms: u64,
}

impl RelationRef {
    pub fn new(
        target_orb_id: Uuid,
        relation_type: RelationType,
        weight: f32,
        trust_at_creation: f32,
        created_at: HlcTimestamp,
    ) -> Self {
        Self {
            edge_id: Uuid::now_v7(),
            target_orb_id,
            relation_type,
            weight: weight.clamp(0.0, 1.0),
            trust_at_creation: trust_at_creation.clamp(0.0, 1.0),
            created_at,
            last_used_ms: now_ms(),
        }
    }

    /// Mark this edge as recently used (resets LRU timer).
    pub fn touch(&mut self) {
        self.last_used_ms = now_ms();
    }
}

// ── RBAC Permissions ──────────────────────────────────────────────────────────

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    pub struct Permissions: u64 {
        const READ_ORB          = 0x001;
        const WRITE_ORB         = 0x002;
        const CREATE_RELATION   = 0x004;
        const UPDATE_TRUST      = 0x008;  // System only
        const READ_EVENTS       = 0x010;
        const EMIT_EVENTS       = 0x020;
        const ADMIN_COMPRESS    = 0x040;  // Operators only
        const ADMIN_MERGE       = 0x080;  // Operators only
        const READ_INVARIANTS   = 0x100;
        // 0x200 (MODIFY_INVARIANTS) intentionally omitted — registry is read-only at runtime
    }
}

impl Permissions {
    pub fn default_reader() -> Self {
        Permissions::READ_ORB | Permissions::READ_EVENTS | Permissions::READ_INVARIANTS
    }

    pub fn default_writer() -> Self {
        Permissions::default_reader()
            | Permissions::WRITE_ORB
            | Permissions::CREATE_RELATION
            | Permissions::EMIT_EVENTS
    }

    pub fn system() -> Self {
        Permissions::default_writer() | Permissions::UPDATE_TRUST
    }

    pub fn operator() -> Self {
        Permissions::system() | Permissions::ADMIN_COMPRESS | Permissions::ADMIN_MERGE
    }
}

// ── ORB ───────────────────────────────────────────────────────────────────────

/// The full ORB — atomic unit of PIRIA.
/// All invariants are enforced at the method level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Orb {
    // ── Identity ────────────────────────────────────────────────────────
    pub identity: Uuid,

    // ── Semantic state ──────────────────────────────────────────────────
    pub semantic_state: SemanticState,

    // ── Semantic representation ─────────────────────────────────────────
    /// L2-normalized embedding — invariant: ||v||₂ = 1.0
    pub semantic_vector: SemanticVector,

    /// Embedding at creation time — used for drift detection.
    pub origin_vector: SemanticVector,

    // ── Relational topology ─────────────────────────────────────────────
    pub relations: Vec<RelationRef>,
    pub degree_budget: usize,

    // ── Causal history ──────────────────────────────────────────────────
    pub event_refs: Vec<Uuid>,

    // ── Probabilistic belief ────────────────────────────────────────────
    pub probability_field: ProbabilityField,

    // ── Derived metrics (lazy, dirty-flagged) ───────────────────────────
    /// Weighted degree centrality ∈ [0,1].
    pub semantic_mass: f32,
    semantic_mass_dirty: bool,

    /// Shannon entropy of probability_field in bits.
    pub entropy: f32,
    entropy_dirty: bool,

    // ── Trust ────────────────────────────────────────────────────────────
    pub trust: TrustRecord,

    // ── Access control ───────────────────────────────────────────────────
    pub permissions: Permissions,

    // ── Integrity ────────────────────────────────────────────────────────
    /// SHA-256 over all mutable fields. Recomputed on every write.
    pub integrity_hash: [u8; 32],

    // ── Timestamps ───────────────────────────────────────────────────────
    pub created_at: HlcTimestamp,
    pub updated_at: HlcTimestamp,
}

impl Orb {
    pub fn new(
        semantic_vector: SemanticVector,
        probability_field: ProbabilityField,
        permissions: Permissions,
        created_at: HlcTimestamp,
    ) -> Self {
        let initial_entropy = probability_field.entropy_bits();
        let identity = Uuid::now_v7();
        let origin_vector = semantic_vector.clone();

        let mut orb = Self {
            identity,
            semantic_state: SemanticState::Active,
            semantic_vector,
            origin_vector,
            relations: Vec::new(),
            degree_budget: DEFAULT_DEGREE_BUDGET,
            event_refs: Vec::new(),
            probability_field,
            semantic_mass: 0.0,
            semantic_mass_dirty: false,
            entropy: initial_entropy,
            entropy_dirty: false,
            trust: TrustRecord::initial(),
            permissions,
            integrity_hash: [0u8; 32],
            created_at,
            updated_at: created_at,
        };

        orb.integrity_hash = orb.compute_integrity_hash();
        orb
    }

    // ── State transitions ────────────────────────────────────────────────

    pub fn transition_to(&mut self, next: SemanticState, updated_at: HlcTimestamp) -> Result<()> {
        if !self.semantic_state.can_transition_to(&next) {
            return Err(PiriaError::InvalidStateTransition {
                from: self.semantic_state,
                to: next,
            });
        }
        self.semantic_state = next;
        self.updated_at = updated_at;
        self.integrity_hash = self.compute_integrity_hash();
        Ok(())
    }

    // ── Vector update ────────────────────────────────────────────────────

    pub fn update_vector(&mut self, new_vector: SemanticVector, updated_at: HlcTimestamp) {
        self.semantic_vector = new_vector;
        self.updated_at = updated_at;
        self.semantic_mass_dirty = true;
        self.integrity_hash = self.compute_integrity_hash();
    }

    /// Semantic drift since creation: ||v(t) - v(t₀)||₂
    pub fn drift(&self) -> f32 {
        self.semantic_vector.l2_distance(&self.origin_vector)
    }

    /// True if drift exceeds the alert threshold.
    pub fn has_drifted(&self) -> bool {
        self.drift() > DRIFT_THRESHOLD
    }

    // ── Relation management ──────────────────────────────────────────────

    pub fn add_relation(&mut self, relation: RelationRef, updated_at: HlcTimestamp) -> Result<()> {
        if self.relations.len() >= self.degree_budget {
            self.prune_lowest_weight_relation();
        }
        if self.relations.len() >= self.degree_budget {
            return Err(PiriaError::DegreeBudgetExceeded {
                orb_id: self.identity,
                budget: self.degree_budget,
            });
        }
        self.relations.push(relation);
        self.updated_at = updated_at;
        self.semantic_mass_dirty = true;
        self.integrity_hash = self.compute_integrity_hash();
        Ok(())
    }

    pub fn remove_relation(&mut self, edge_id: Uuid, updated_at: HlcTimestamp) -> bool {
        let before = self.relations.len();
        self.relations.retain(|r| r.edge_id != edge_id);
        let removed = self.relations.len() < before;
        if removed {
            self.updated_at = updated_at;
            self.semantic_mass_dirty = true;
            self.integrity_hash = self.compute_integrity_hash();
        }
        removed
    }

    /// Remove the relation with the lowest weight; LRU timestamp as tiebreaker.
    fn prune_lowest_weight_relation(&mut self) {
        if self.relations.is_empty() {
            return;
        }
        let idx = self
            .relations
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                a.weight
                    .partial_cmp(&b.weight)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.last_used_ms.cmp(&b.last_used_ms))
            })
            .map(|(i, _)| i)
            .unwrap();
        self.relations.swap_remove(idx);
    }

    // ── Entropy ──────────────────────────────────────────────────────────

    pub fn entropy(&mut self) -> f32 {
        if self.entropy_dirty {
            self.entropy = self.probability_field.entropy_bits();
            self.entropy_dirty = false;
        }
        self.entropy
    }

    pub fn update_probability_field(
        &mut self,
        observations: &[f32],
        updated_at: HlcTimestamp,
    ) -> Result<()> {
        self.probability_field.observe(observations)?;
        self.entropy_dirty = true;
        self.updated_at = updated_at;
        self.integrity_hash = self.compute_integrity_hash();
        Ok(())
    }

    // ── Semantic mass — spec §3.2 ─────────────────────────────────────────
    //
    // semantic_mass(v) = Σ_{u ∈ N(v)} trust(u) · rel_weight(u,v) / degree_budget
    // We store trust_at_creation as the trust proxy (§3.2 note).

    pub fn recompute_semantic_mass(&mut self) {
        if self.degree_budget == 0 {
            self.semantic_mass = 0.0;
        } else {
            let weighted_sum: f32 = self
                .relations
                .iter()
                .map(|r| r.trust_at_creation * r.weight)
                .sum();
            self.semantic_mass = (weighted_sum / self.degree_budget as f32).min(1.0);
        }
        self.semantic_mass_dirty = false;
    }

    pub fn semantic_mass(&mut self) -> f32 {
        if self.semantic_mass_dirty {
            self.recompute_semantic_mass();
        }
        self.semantic_mass
    }

    // ── Trust ────────────────────────────────────────────────────────────

    pub fn update_trust(
        &mut self,
        observation_weight: f32,
        outcome: f32,
        learning_rate: f32,
        updated_at: HlcTimestamp,
    ) -> Result<()> {
        self.trust.update(observation_weight, outcome, learning_rate, self.identity)?;
        self.updated_at = updated_at;
        self.integrity_hash = self.compute_integrity_hash();
        Ok(())
    }

    // ── Integrity ────────────────────────────────────────────────────────

    /// SHA-256 over all mutable fields including relations, probability_field,
    /// event_refs, and timestamps.
    ///
    /// FIX: Original only hashed identity + state + vector + trust + permissions.
    /// This version covers all fields per spec §3.1 (integrity_hash requirement).
    fn compute_integrity_hash(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();

        // Identity + state
        hasher.update(self.identity.as_bytes());
        hasher.update(&[self.semantic_state as u8]);

        // Semantic vector
        for f in &self.semantic_vector.0 {
            hasher.update(&f.to_le_bytes());
        }

        // Relations (sorted by edge_id for determinism)
        let mut sorted_edges: Vec<&RelationRef> = self.relations.iter().collect();
        sorted_edges.sort_by_key(|r| r.edge_id);
        for r in sorted_edges {
            hasher.update(r.edge_id.as_bytes());
            hasher.update(r.target_orb_id.as_bytes());
            hasher.update(&r.weight.to_le_bytes());
            hasher.update(&r.trust_at_creation.to_le_bytes());
        }

        // Probability field
        for a in &self.probability_field.alpha {
            hasher.update(&a.to_le_bytes());
        }

        // Event refs
        for eid in &self.event_refs {
            hasher.update(eid.as_bytes());
        }

        // Trust
        hasher.update(&self.trust.score.to_le_bytes());
        hasher.update(&self.trust.evidence_count.to_le_bytes());

        // Permissions
        hasher.update(&self.permissions.bits().to_le_bytes());

        // Timestamps
        hasher.update(&self.created_at.physical_ms.to_le_bytes());
        hasher.update(&self.created_at.logical.to_le_bytes());
        hasher.update(&self.updated_at.physical_ms.to_le_bytes());
        hasher.update(&self.updated_at.logical.to_le_bytes());

        hasher.finalize().into()
    }

    pub fn verify_integrity(&self) -> Result<()> {
        let computed = self.compute_integrity_hash();
        if computed != self.integrity_hash {
            return Err(PiriaError::IntegrityHashMismatch {
                orb_id: self.identity,
                stored: hex::encode(self.integrity_hash),
                computed: hex::encode(computed),
            });
        }
        Ok(())
    }

    pub fn record_event(&mut self, event_id: Uuid) {
        self.event_refs.push(event_id);
        self.integrity_hash = self.compute_integrity_hash();
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_orb() -> Orb {
        let ts = HlcTimestamp::new(1_000_000, 0);
        Orb::new(
            SemanticVector::test_random(),
            ProbabilityField::uniform(4).unwrap(),
            Permissions::default_writer(),
            ts,
        )
    }

    #[test]
    fn orb_creation_has_valid_hash() {
        let orb = make_orb();
        orb.verify_integrity().unwrap();
    }

    #[test]
    fn state_transition_valid() {
        let mut orb = make_orb();
        let ts = HlcTimestamp::new(2_000_000, 0);
        orb.transition_to(SemanticState::Dormant, ts).unwrap();
        assert_eq!(orb.semantic_state, SemanticState::Dormant);
        orb.verify_integrity().unwrap();
    }

    #[test]
    fn state_transition_invalid_rejected() {
        let mut orb = make_orb();
        let ts = HlcTimestamp::new(2_000_000, 0);
        orb.transition_to(SemanticState::Compressed, ts).unwrap();
        let result = orb.transition_to(SemanticState::Active, ts);
        assert!(matches!(result, Err(PiriaError::InvalidStateTransition { .. })));
    }

    #[test]
    fn degree_budget_enforced_via_pruning() {
        let mut orb = make_orb();
        orb.degree_budget = 3;
        let ts = HlcTimestamp::new(1_000_000, 0);
        for i in 0..4u32 {
            let rel = RelationRef::new(
                Uuid::now_v7(),
                RelationType::References,
                (i as f32 + 1.0) / 4.0,
                0.5,
                ts,
            );
            orb.add_relation(rel, ts).unwrap();
        }
        assert_eq!(orb.relations.len(), 3);
        let min_weight = orb.relations.iter().map(|r| r.weight).fold(f32::MAX, f32::min);
        assert!(min_weight > 0.25 + 1e-5);
    }

    #[test]
    fn integrity_hash_covers_relations() {
        let mut orb = make_orb();
        let ts = HlcTimestamp::new(1_000_000, 0);
        let hash_before = orb.integrity_hash;
        let rel = RelationRef::new(Uuid::now_v7(), RelationType::Supports, 0.7, 0.5, ts);
        orb.add_relation(rel, ts).unwrap();
        assert_ne!(orb.integrity_hash, hash_before, "hash must change on relation add");
        orb.verify_integrity().unwrap();
    }

    #[test]
    fn integrity_hash_covers_probability_field() {
        let mut orb = make_orb();
        let ts = HlcTimestamp::new(2_000_000, 0);
        let hash_before = orb.integrity_hash;
        orb.update_probability_field(&[1.0, 0.0, 0.0, 0.0], ts).unwrap();
        assert_ne!(orb.integrity_hash, hash_before, "hash must change on belief update");
        orb.verify_integrity().unwrap();
    }

    #[test]
    fn probability_field_entropy_uniform() {
        let pf = ProbabilityField::uniform(4).unwrap();
        let h = pf.entropy_bits();
        assert!((h - 2.0).abs() < 1e-4, "Expected H≈2.0, got {h}");
        assert!((pf.normalized_entropy() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn trust_ceiling_never_exceeded() {
        let mut rec = TrustRecord::initial();
        let dummy_id = Uuid::now_v7();
        for _ in 0..100 {
            let _ = rec.update(1.0, 1.0, 0.9, dummy_id);
        }
        assert!(rec.score <= rec.ceiling() + 1e-5);
    }

    #[test]
    fn trust_floor_enforced() {
        let rec = TrustRecord::initial();
        assert!(rec.effective_score() >= TRUST_FLOOR);
    }

    #[test]
    fn vector_normalization_enforced() {
        let raw = vec![3.0f32; VECTOR_DIM];
        let v = SemanticVector::new(raw).unwrap();
        let norm = l2_norm(&v.0);
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn semantic_mass_bounded() {
        let mut orb = make_orb();
        let ts = HlcTimestamp::new(1_000_000, 0);
        for _ in 0..10 {
            let rel = RelationRef::new(Uuid::now_v7(), RelationType::Supports, 1.0, 1.0, ts);
            let _ = orb.add_relation(rel, ts);
        }
        let mass = orb.semantic_mass();
        assert!(mass >= 0.0 && mass <= 1.0, "mass={mass}");
    }

    #[test]
    fn drift_zero_at_creation() {
        let orb = make_orb();
        assert!(orb.drift() < 1e-5);
    }

    #[test]
    fn probability_field_merge_sums_alphas() {
        let pf1 = ProbabilityField::new(vec![1.0, 2.0]).unwrap();
        let pf2 = ProbabilityField::new(vec![3.0, 4.0]).unwrap();
        let merged = pf1.merge(&pf2).unwrap();
        assert!((merged.alpha[0] - 4.0).abs() < 1e-5);
        assert!((merged.alpha[1] - 6.0).abs() < 1e-5);
    }
}