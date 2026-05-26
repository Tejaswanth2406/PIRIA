/// CRDT Merge Semantics — §11.3 of PIRIA MVP Spec
///
/// PIRIA uses CRDTs for the AP (eventually consistent) components of its
/// distributed architecture. This module implements merge laws for:
///
///   - Relations: OR-Set (observed-remove set)
///   - Trust:     Last-Write-Wins register, clamped to floor
///   - Probability fields: Dirichlet pooling (sum alphas)
///   - Semantic vectors: trust-weighted average (then re-normalize)
///   - Event refs: grow-only set (G-Set)
///   - ORB:        composite merge of all the above
///
/// All merge operations are:
///   - Commutative:  merge(A, B) = merge(B, A)
///   - Associative:  merge(merge(A, B), C) = merge(A, merge(B, C))
///   - Idempotent:   merge(A, A) = A
///
/// CP components (trust core, invariant registry) are NOT handled here —
/// those are reconciled via Raft on partition heal.
use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    clock::{HlcTimestamp, VectorClock},
    error::{PiriaError, Result},
    orb::{
        l2_norm, now_ms, Orb, Permissions, ProbabilityField, RelationRef, RelationType,
        SemanticState, SemanticVector, TrustRecord, TRUST_FLOOR, VECTOR_DIM,
    },
};

// ── OR-Set for Relations ───────────────────────────────────────────────────────

/// An OR-Set (Observed-Remove Set) for relation edges.
///
/// Each element has a unique tag (edge_id). Additions are recorded as
/// (edge_id, tombstone=false); removals add (edge_id, tombstone=true).
/// On merge: an edge is present iff it was added on at least one replica
/// AND not tombstoned on any replica that has seen the add.
///
/// Simplified version: Last-Write-Wins per edge_id using HLC timestamps.
/// A tombstone with a higher HLC beats an add.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RelationOrSet {
    /// Live edges: edge_id → (RelationRef, hlc of last write)
    pub live: HashMap<Uuid, (RelationRef, HlcTimestamp)>,
    /// Tombstones: edge_id → hlc of removal
    pub tombstones: HashMap<Uuid, HlcTimestamp>,
}

impl RelationOrSet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a relation edge.
    pub fn add(&mut self, relation: RelationRef, hlc: HlcTimestamp) {
        let edge_id = relation.edge_id;

        // Respect tombstone: if already removed at a later HLC, don't re-add
        if let Some(&tomb_hlc) = self.tombstones.get(&edge_id) {
            if tomb_hlc >= hlc {
                return;
            }
        }

        self.live
            .entry(edge_id)
            .and_modify(|(existing_relation, existing_hlc)| {
                if hlc > *existing_hlc {
                    *existing_relation = relation.clone();
                    *existing_hlc = hlc;
                }
            })
            .or_insert((relation, hlc));
    }

    /// Remove a relation edge by id.
    pub fn remove(&mut self, edge_id: Uuid, hlc: HlcTimestamp) {
        // Only tombstone if the add HLC is ≤ removal HLC
        let should_tombstone = self
            .live
            .get(&edge_id)
            .map(|(_, add_hlc)| *add_hlc <= hlc)
            .unwrap_or(true);

        if should_tombstone {
            self.live.remove(&edge_id);
            self.tombstones
                .entry(edge_id)
                .and_modify(|existing| {
                    if hlc > *existing {
                        *existing = hlc;
                    }
                })
                .or_insert(hlc);
        }
    }

    /// Merge two OR-Sets. Commutative, associative, idempotent.
    pub fn merge(&self, other: &RelationOrSet) -> RelationOrSet {
        let mut result = RelationOrSet::new();

        // Merge tombstones (take max HLC per edge_id)
        for (&eid, &hlc) in &self.tombstones {
            result.tombstones.insert(eid, hlc);
        }
        for (&eid, &hlc) in &other.tombstones {
            result
                .tombstones
                .entry(eid)
                .and_modify(|existing| {
                    if hlc > *existing {
                        *existing = hlc;
                    }
                })
                .or_insert(hlc);
        }

        // Merge live edges — a live edge wins unless tombstoned at >= its HLC
        for (&eid, (rel, hlc)) in &self.live {
            if result.tombstones.get(&eid).map(|&t| t >= *hlc).unwrap_or(false) {
                continue;
            }
            result.live.insert(eid, (rel.clone(), *hlc));
        }
        for (&eid, (rel, hlc)) in &other.live {
            if result.tombstones.get(&eid).map(|&t| t >= *hlc).unwrap_or(false) {
                continue;
            }
            result
                .live
                .entry(eid)
                .and_modify(|(existing_relation, existing_hlc)| {
                    if *hlc > *existing_hlc {
                        *existing_relation = rel.clone();
                        *existing_hlc = *hlc;
                    }
                })
                .or_insert((rel.clone(), *hlc));
        }

        result
    }

    /// Returns the currently live relations (no tombstones).
    pub fn relations(&self) -> Vec<&RelationRef> {
        self.live.values().map(|(r, _)| r).collect()
    }

    /// Number of live edges.
    pub fn len(&self) -> usize {
        self.live.len()
    }

    pub fn is_empty(&self) -> bool {
        self.live.is_empty()
    }
}

// ── LWW Register for Trust ────────────────────────────────────────────────────

/// Last-Write-Wins register for trust scores.
///
/// On merge: the replica with the higher HLC wins.
/// Post-merge: floors are applied (trust ≥ TRUST_FLOOR).
///
/// Note: the CP trust core uses Raft for strong consistency. This register
/// is used only for the eventual-consistency replica / cache layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustLwwRegister {
    pub record: TrustRecord,
    /// HLC of the last write to this register.
    pub last_write_hlc: HlcTimestamp,
}

impl TrustLwwRegister {
    pub fn new(record: TrustRecord, hlc: HlcTimestamp) -> Self {
        Self { record, last_write_hlc: hlc }
    }

    /// Merge: take the record with the higher HLC. Floor-clamp post-merge.
    pub fn merge(&self, other: &TrustLwwRegister) -> TrustLwwRegister {
        let winner = if self.last_write_hlc >= other.last_write_hlc {
            self.clone()
        } else {
            other.clone()
        };

        // Apply trust floor invariant
        let mut result = winner;
        result.record.score = result.record.score.max(TRUST_FLOOR);
        result
    }
}

// ── G-Set for Event Refs ──────────────────────────────────────────────────────

/// Grow-only set (G-Set) for event references.
/// Merging is set union. Elements are never removed.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EventRefGSet {
    pub refs: HashSet<Uuid>,
}

impl EventRefGSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, event_id: Uuid) {
        self.refs.insert(event_id);
    }

    pub fn merge(&self, other: &EventRefGSet) -> EventRefGSet {
        EventRefGSet {
            refs: self.refs.union(&other.refs).copied().collect(),
        }
    }
}

// ── Semantic Vector Merge ─────────────────────────────────────────────────────

/// Merge two semantic vectors using trust-weighted averaging.
///
/// v_merged = (trust_a · v_a + trust_b · v_b) / ||trust_a · v_a + trust_b · v_b||₂
///
/// This preserves the unit-sphere invariant and respects epistemic authority.
pub fn merge_semantic_vectors(
    va: &[f32],
    trust_a: f32,
    vb: &[f32],
    trust_b: f32,
) -> Result<SemanticVector> {
    if va.len() != VECTOR_DIM || vb.len() != VECTOR_DIM {
        return Err(PiriaError::VectorDimensionMismatch {
            expected: VECTOR_DIM,
            got: if va.len() != VECTOR_DIM { va.len() } else { vb.len() },
        });
    }

    let trust_a = sanitize_merge_trust(trust_a);
    let trust_b = sanitize_merge_trust(trust_b);
    let raw: Vec<f32> = va
        .iter()
        .zip(vb.iter())
        .map(|(&a, &b)| trust_a * a + trust_b * b)
        .collect();

    if !raw.iter().all(|v| v.is_finite()) {
        return Err(PiriaError::VectorNotNormalized { norm: f32::NAN });
    }

    if l2_norm(&raw) < 1e-8 {
        return if trust_b > trust_a
            || (trust_b == trust_a && vector_tie_breaks_after(vb, va))
        {
            SemanticVector::new(vb.to_vec())
        } else {
            SemanticVector::new(va.to_vec())
        };
    }

    SemanticVector::new(raw)
}

fn sanitize_merge_trust(trust: f32) -> f32 {
    if trust.is_finite() {
        trust.max(0.0)
    } else {
        0.0
    }
}

fn vector_tie_breaks_after(candidate: &[f32], current: &[f32]) -> bool {
    for (&candidate_value, &current_value) in candidate.iter().zip(current.iter()) {
        match candidate_value.total_cmp(&current_value) {
            std::cmp::Ordering::Greater => return true,
            std::cmp::Ordering::Less => return false,
            std::cmp::Ordering::Equal => {}
        }
    }
    false
}

// ── SemanticState Merge ───────────────────────────────────────────────────────

/// Merge two semantic states by taking the "more advanced" one.
///
/// The state lattice is:
///   Active < Dormant < Compressed < Archived
///
/// Rationale: archival is irreversible; compressed > dormant > active.
/// A partition that archived an ORB wins over one that kept it active.
pub fn merge_semantic_states(a: SemanticState, b: SemanticState) -> SemanticState {
    // Assign a total order matching the lifecycle lattice
    let rank = |s: SemanticState| match s {
        SemanticState::Active => 0u8,
        SemanticState::Dormant => 1,
        SemanticState::Compressed => 2,
        SemanticState::Archived => 3,
    };
    if rank(a) >= rank(b) { a } else { b }
}

// ── ORB CRDT State ────────────────────────────────────────────────────────────

/// The complete CRDT state for a single ORB.
///
/// This is the "CRDT envelope" around an ORB — it holds all the per-field
/// CRDT structures needed to merge two replicas of the same ORB.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrbCrdtState {
    pub orb_id: Uuid,
    pub semantic_state: SemanticState,
    pub semantic_vector: Vec<f32>,
    pub relations: RelationOrSet,
    pub probability_field: ProbabilityField,
    pub trust: TrustLwwRegister,
    pub event_refs: EventRefGSet,
    pub permissions: Permissions,
    pub created_at: HlcTimestamp,
    pub updated_at: HlcTimestamp,
    /// Vector clock snapshot at last local update.
    pub vector_clock: VectorClock,
}

impl OrbCrdtState {
    /// Snapshot the current ORB state into CRDT form.
    pub fn from_orb(orb: &Orb, hlc: HlcTimestamp, vector_clock: VectorClock) -> Self {
        let mut relations = RelationOrSet::new();
        for rel in &orb.relations {
            relations.add(rel.clone(), orb.updated_at);
        }

        let mut event_refs = EventRefGSet::new();
        for &eid in &orb.event_refs {
            event_refs.add(eid);
        }

        Self {
            orb_id: orb.identity,
            semantic_state: orb.semantic_state,
            semantic_vector: orb.semantic_vector.0.clone(),
            relations,
            probability_field: orb.probability_field.clone(),
            trust: TrustLwwRegister::new(orb.trust.clone(), hlc),
            event_refs,
            permissions: orb.permissions,
            created_at: orb.created_at,
            updated_at: orb.updated_at,
            vector_clock,
        }
    }

    /// Merge two CRDT states for the same ORB.
    ///
    /// Merge laws per field:
    ///   semantic_state:    lattice max (archived > compressed > dormant > active)
    ///   semantic_vector:   trust-weighted average → re-normalize
    ///   relations:         OR-Set merge
    ///   probability_field: Dirichlet pooling (sum alphas)
    ///   trust:             LWW (higher HLC wins), floor-clamped
    ///   event_refs:        G-Set union
    ///   permissions:       bitwise AND (conservative — lose no permissions on merge)
    ///   updated_at:        max HLC
    pub fn merge(&self, other: &OrbCrdtState) -> Result<OrbCrdtState> {
        if self.orb_id != other.orb_id {
            return Err(PiriaError::CrdtMergeConflict {
                orb_id: self.orb_id,
                reason: format!(
                    "Cannot merge CRDT states for different ORBs: {} vs {}",
                    self.orb_id, other.orb_id
                ),
            });
        }

        let merged_state = merge_semantic_states(self.semantic_state, other.semantic_state);

        let trust_a = self.trust.record.effective_score();
        let trust_b = other.trust.record.effective_score();
        let merged_vector = merge_semantic_vectors(
            &self.semantic_vector,
            trust_a,
            &other.semantic_vector,
            trust_b,
        )?;

        let merged_relations = self.relations.merge(&other.relations);

        let merged_prob = self.probability_field.merge(&other.probability_field)?;

        let merged_trust = self.trust.merge(&other.trust);

        let merged_event_refs = self.event_refs.merge(&other.event_refs);

        // Conservative permission merge: AND (no one loses permissions during partition)
        let merged_permissions = self.permissions & other.permissions;

        let merged_vector_clock = self.vector_clock.merged_with(&other.vector_clock);

        let merged_updated_at = self.updated_at.max(other.updated_at);

        Ok(OrbCrdtState {
            orb_id: self.orb_id,
            semantic_state: merged_state,
            semantic_vector: merged_vector.0,
            relations: merged_relations,
            probability_field: merged_prob,
            trust: merged_trust,
            event_refs: merged_event_refs,
            permissions: merged_permissions,
            created_at: self.created_at.min(other.created_at),
            updated_at: merged_updated_at,
            vector_clock: merged_vector_clock,
        })
    }
}

// ── Partition Merge Coordinator ───────────────────────────────────────────────

/// Coordinates partition heal — reconciles divergent CRDT states.
///
/// On heal, each side provides its OrbCrdtState snapshots.
/// The coordinator merges them and reports conflicts.
pub struct PartitionMergeCoordinator;

impl PartitionMergeCoordinator {
    /// Merge partition A's ORB states with partition B's.
    ///
    /// Returns:
    ///   - merged: successfully merged CRDT states
    ///   - conflicts: ORB ids that could not be auto-merged (require operator resolution)
    pub fn reconcile(
        partition_a: &[OrbCrdtState],
        partition_b: &[OrbCrdtState],
    ) -> PartitionHealResult {
        let mut a_map: HashMap<Uuid, &OrbCrdtState> = HashMap::new();
        for state in partition_a {
            a_map.insert(state.orb_id, state);
        }

        let mut b_map: HashMap<Uuid, &OrbCrdtState> = HashMap::new();
        for state in partition_b {
            b_map.insert(state.orb_id, state);
        }

        let mut merged: Vec<OrbCrdtState> = Vec::new();
        let mut conflicts: Vec<(Uuid, String)> = Vec::new();

        // All ORBs on partition A
        for (&orb_id, &a_state) in &a_map {
            match b_map.get(&orb_id) {
                Some(&b_state) => {
                    // Both partitions have this ORB — merge
                    match a_state.merge(b_state) {
                        Ok(m) => merged.push(m),
                        Err(e) => conflicts.push((orb_id, e.to_string())),
                    }
                }
                None => {
                    // Only on A — take as-is
                    merged.push(a_state.clone());
                }
            }
        }

        // ORBs only on partition B
        for (&orb_id, &b_state) in &b_map {
            if !a_map.contains_key(&orb_id) {
                merged.push(b_state.clone());
            }
        }

        PartitionHealResult { merged, conflicts }
    }
}

/// Result of a partition heal operation.
pub struct PartitionHealResult {
    /// Successfully merged CRDT states (ready to write back).
    pub merged: Vec<OrbCrdtState>,
    /// ORBs that could not be auto-merged.
    pub conflicts: Vec<(Uuid, String)>,
}

impl PartitionHealResult {
    pub fn is_clean(&self) -> bool {
        self.conflicts.is_empty()
    }

    pub fn conflict_count(&self) -> usize {
        self.conflicts.len()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::HlcTimestamp;
    use crate::orb::{Orb, Permissions, ProbabilityField, SemanticVector};

    fn make_test_orb() -> Orb {
        let ts = HlcTimestamp::new(1_000_000, 0);
        Orb::new(
            SemanticVector::test_random(),
            ProbabilityField::uniform(4).unwrap(),
            Permissions::default_writer(),
            ts,
        )
    }

    fn make_crdt(orb: &Orb, hlc: HlcTimestamp) -> OrbCrdtState {
        OrbCrdtState::from_orb(orb, hlc, VectorClock::new())
    }

    // ── OR-Set ────────────────────────────────────────────────────────────────

    #[test]
    fn or_set_add_and_remove() {
        let mut set = RelationOrSet::new();
        let ts1 = HlcTimestamp::new(1_000, 0);
        let ts2 = HlcTimestamp::new(2_000, 0);

        let rel = crate::orb::RelationRef::new(
            Uuid::now_v7(),
            RelationType::Supports,
            0.8,
            0.5,
            ts1,
        );
        let eid = rel.edge_id;

        set.add(rel, ts1);
        assert_eq!(set.len(), 1);

        set.remove(eid, ts2);
        assert_eq!(set.len(), 0);
    }

    #[test]
    fn or_set_merge_commutativity() {
        let ts = HlcTimestamp::new(1_000, 0);
        let mut a = RelationOrSet::new();
        let mut b = RelationOrSet::new();

        let rel_a = crate::orb::RelationRef::new(
            Uuid::now_v7(), RelationType::Causes, 0.6, 0.5, ts,
        );
        let rel_b = crate::orb::RelationRef::new(
            Uuid::now_v7(), RelationType::Supports, 0.7, 0.5, ts,
        );

        a.add(rel_a, ts);
        b.add(rel_b, ts);

        let ab = a.merge(&b);
        let ba = b.merge(&a);

        assert_eq!(ab.len(), ba.len());
        assert_eq!(ab.len(), 2);
    }

    #[test]
    fn or_set_merge_idempotent() {
        let ts = HlcTimestamp::new(1_000, 0);
        let mut a = RelationOrSet::new();
        let rel = crate::orb::RelationRef::new(
            Uuid::now_v7(), RelationType::References, 0.5, 0.5, ts,
        );
        a.add(rel, ts);

        let aa = a.merge(&a);
        assert_eq!(aa.len(), 1);
    }

    #[test]
    fn or_set_newer_add_replaces_payload() {
        let ts1 = HlcTimestamp::new(1_000, 0);
        let ts2 = HlcTimestamp::new(2_000, 0);
        let mut set = RelationOrSet::new();

        let mut old_rel = crate::orb::RelationRef::new(
            Uuid::now_v7(),
            RelationType::References,
            0.2,
            0.5,
            ts1,
        );
        let mut new_rel = old_rel.clone();
        new_rel.weight = 0.9;
        new_rel.relation_type = RelationType::Supports;
        old_rel.weight = 0.2;

        set.add(old_rel, ts1);
        set.add(new_rel, ts2);

        let relation = set.relations()[0];
        assert_eq!(&relation.relation_type, &RelationType::Supports);
        assert!((relation.weight - 0.9).abs() < 1e-5);
    }

    #[test]
    fn or_set_merge_newer_add_replaces_payload() {
        let ts1 = HlcTimestamp::new(1_000, 0);
        let ts2 = HlcTimestamp::new(2_000, 0);
        let mut a = RelationOrSet::new();
        let mut b = RelationOrSet::new();

        let old_rel = crate::orb::RelationRef::new(
            Uuid::now_v7(),
            RelationType::References,
            0.2,
            0.5,
            ts1,
        );
        let mut new_rel = old_rel.clone();
        new_rel.weight = 0.9;
        new_rel.relation_type = RelationType::Supports;

        a.add(old_rel, ts1);
        b.add(new_rel, ts2);

        let merged = a.merge(&b);
        let relation = merged.relations()[0];
        assert_eq!(&relation.relation_type, &RelationType::Supports);
        assert!((relation.weight - 0.9).abs() < 1e-5);
    }

    #[test]
    fn or_set_concurrent_add_remove_remove_wins_if_later() {
        let ts_add = HlcTimestamp::new(1_000, 0);
        let ts_remove = HlcTimestamp::new(2_000, 0);

        let mut a = RelationOrSet::new();
        let mut b = RelationOrSet::new();

        let rel = crate::orb::RelationRef::new(
            Uuid::now_v7(), RelationType::References, 0.5, 0.5, ts_add,
        );
        let eid = rel.edge_id;

        a.add(rel.clone(), ts_add);
        b.add(rel, ts_add);
        b.remove(eid, ts_remove);

        let merged = a.merge(&b);
        // Remove at ts_remove beats add at ts_add → edge should be gone
        assert_eq!(merged.len(), 0);
    }

    // ── Trust LWW ─────────────────────────────────────────────────────────────

    #[test]
    fn trust_lww_higher_hlc_wins() {
        let ts1 = HlcTimestamp::new(1_000, 0);
        let ts2 = HlcTimestamp::new(2_000, 0);

        let mut r1 = TrustRecord::initial();
        r1.score = 0.3;
        let mut r2 = TrustRecord::initial();
        r2.score = 0.7;

        let reg1 = TrustLwwRegister::new(r1, ts1);
        let reg2 = TrustLwwRegister::new(r2, ts2);

        let merged = reg1.merge(&reg2);
        assert!((merged.record.score - 0.7).abs() < 1e-5);
    }

    #[test]
    fn trust_lww_floor_applied() {
        let ts = HlcTimestamp::new(1_000, 0);
        let mut r = TrustRecord::initial();
        r.score = 0.0; // below floor
        let reg = TrustLwwRegister::new(r.clone(), ts);
        let merged = reg.merge(&TrustLwwRegister::new(r, ts));
        assert!(merged.record.score >= TRUST_FLOOR);
    }

    // ── G-Set ─────────────────────────────────────────────────────────────────

    #[test]
    fn gset_merge_is_union() {
        let mut a = EventRefGSet::new();
        let mut b = EventRefGSet::new();
        let id1 = Uuid::now_v7();
        let id2 = Uuid::now_v7();
        let id3 = Uuid::now_v7();

        a.add(id1);
        a.add(id2);
        b.add(id2);
        b.add(id3);

        let merged = a.merge(&b);
        assert_eq!(merged.refs.len(), 3);
        assert!(merged.refs.contains(&id1));
        assert!(merged.refs.contains(&id2));
        assert!(merged.refs.contains(&id3));
    }

    #[test]
    fn semantic_vector_merge_survives_opposing_vectors() {
        let mut va = vec![0.0; VECTOR_DIM];
        va[0] = 1.0;
        let mut vb = vec![0.0; VECTOR_DIM];
        vb[0] = -1.0;

        let merged = merge_semantic_vectors(&va, 1.0, &vb, 1.0).unwrap();
        let reverse = merge_semantic_vectors(&vb, 1.0, &va, 1.0).unwrap();
        let norm = l2_norm(&merged.0);

        assert!(norm.is_finite());
        assert!((norm - 1.0).abs() < 1e-4);
        assert_eq!(merged.0, reverse.0);
    }

    #[test]
    fn semantic_vector_merge_sanitizes_invalid_trust() {
        let mut va = vec![0.0; VECTOR_DIM];
        va[0] = 1.0;
        let mut vb = vec![0.0; VECTOR_DIM];
        vb[1] = 1.0;

        let merged = merge_semantic_vectors(&va, f32::NAN, &vb, 1.0).unwrap();

        assert!(merged.0[1] > 0.99);
    }

    // ── Semantic State Lattice ────────────────────────────────────────────────

    #[test]
    fn state_merge_archived_always_wins() {
        assert_eq!(
            merge_semantic_states(SemanticState::Active, SemanticState::Archived),
            SemanticState::Archived
        );
        assert_eq!(
            merge_semantic_states(SemanticState::Archived, SemanticState::Dormant),
            SemanticState::Archived
        );
    }

    #[test]
    fn state_merge_commutativity() {
        let states = [
            SemanticState::Active,
            SemanticState::Dormant,
            SemanticState::Compressed,
            SemanticState::Archived,
        ];
        for &a in &states {
            for &b in &states {
                assert_eq!(merge_semantic_states(a, b), merge_semantic_states(b, a));
            }
        }
    }

    // ── ORB CRDT Merge ────────────────────────────────────────────────────────

    #[test]
    fn orb_crdt_merge_commutativity() {
        let orb = make_test_orb();
        let ts = HlcTimestamp::new(1_000_000, 0);
        let a = make_crdt(&orb, ts);
        let b = make_crdt(&orb, HlcTimestamp::new(1_001_000, 0));

        let ab = a.merge(&b).unwrap();
        let ba = b.merge(&a).unwrap();

        // Same ORB id, same state after merge
        assert_eq!(ab.orb_id, ba.orb_id);
        assert_eq!(ab.semantic_state, ba.semantic_state);
    }

    #[test]
    fn orb_crdt_merge_different_ids_rejected() {
        let orb_a = make_test_orb();
        let orb_b = make_test_orb();
        let ts = HlcTimestamp::new(1_000_000, 0);
        let a = make_crdt(&orb_a, ts);
        let b = make_crdt(&orb_b, ts);

        assert!(matches!(a.merge(&b), Err(PiriaError::CrdtMergeConflict { .. })));
    }

    // ── Partition Heal ────────────────────────────────────────────────────────

    #[test]
    fn partition_heal_merges_all_orbs() {
        let orb1 = make_test_orb();
        let orb2 = make_test_orb();
        let ts = HlcTimestamp::new(1_000_000, 0);

        let partition_a = vec![make_crdt(&orb1, ts)];
        let partition_b = vec![make_crdt(&orb2, ts)];

        let result = PartitionMergeCoordinator::reconcile(&partition_a, &partition_b);
        assert!(result.is_clean());
        assert_eq!(result.merged.len(), 2);
    }

    #[test]
    fn partition_heal_same_orb_both_sides() {
        let orb = make_test_orb();
        let ts_a = HlcTimestamp::new(1_000_000, 0);
        let ts_b = HlcTimestamp::new(1_500_000, 0);

        let partition_a = vec![make_crdt(&orb, ts_a)];
        let partition_b = vec![make_crdt(&orb, ts_b)];

        let result = PartitionMergeCoordinator::reconcile(&partition_a, &partition_b);
        assert!(result.is_clean());
        assert_eq!(result.merged.len(), 1); // same ORB merged into one
    }
}
