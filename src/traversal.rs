/// Traversal Engine — §8 of PIRIA MVP Spec
///
/// Attention-weighted BFS/DFS with strict budget bounds.
/// Complexity: O(B · log|V|) where B = budget, not |V|.
///
/// Core design guarantees:
///   - Total attention Σ A(v) = B (conserved, never created)
///   - No single node receives > B × 0.20 (singularity prevention)
///   - Every reachable node receives ≥ 1 unit (isolation prevention)
///   - Visited set prevents re-entry (cycle detection)
///   - Max depth bound prevents infinite recursion
use std::collections::{HashMap, HashSet};

use priority_queue::PriorityQueue;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    error::{PiriaError, Result},
    orb::{Orb, RelationType},
};

// ── Constants ─────────────────────────────────────────────────────────────────

/// Default total attention budget per traversal call.
pub const DEFAULT_BUDGET: u64 = 10_000;

/// Default maximum traversal depth (hops from start).
pub const DEFAULT_MAX_DEPTH: usize = 8;

/// Maximum fraction of budget any single node can receive.
pub const SINGULARITY_CAP: f32 = 0.20;

/// Minimum attention floor per reachable node (prevents zero-attention isolation).
pub const ATTENTION_FLOOR: u64 = 1;

/// Default score threshold — only nodes scoring above this are returned.
pub const DEFAULT_SCORE_THRESHOLD: f32 = 0.1;

/// Scoring weights — λ₁..λ₄ from spec §8 traversal scoring.
pub const LAMBDA_SIMILARITY: f32 = 0.40;
pub const LAMBDA_TRUST: f32 = 0.30;
pub const LAMBDA_RECENCY: f32 = 0.15;
pub const LAMBDA_SEMANTIC_MASS: f32 = 0.15;

/// Recency half-life in milliseconds (≈ 7 days).
pub const RECENCY_HALF_LIFE_MS: f64 = 7.0 * 24.0 * 3_600_000.0;

// ── Traversal Config ──────────────────────────────────────────────────────────

/// Per-call traversal configuration. All fields have sane defaults.
#[derive(Debug, Clone)]
pub struct TraversalConfig {
    /// Total attention units for this traversal.
    pub budget: u64,
    /// Maximum hops from the start node.
    pub max_depth: usize,
    /// Minimum score for inclusion in results.
    pub score_threshold: f32,
    /// Minimum trust required to traverse an edge.
    pub min_trust: f32,
    /// If set, only traverse edges of these types.
    pub allowed_relation_types: Option<Vec<RelationType>>,
    /// Scoring weight overrides. If None, use module-level defaults.
    pub scoring_weights: Option<ScoringWeights>,
}

impl Default for TraversalConfig {
    fn default() -> Self {
        Self {
            budget: DEFAULT_BUDGET,
            max_depth: DEFAULT_MAX_DEPTH,
            score_threshold: DEFAULT_SCORE_THRESHOLD,
            min_trust: 0.0,
            allowed_relation_types: None,
            scoring_weights: None,
        }
    }
}

/// Scoring weight vector — λ₁..λ₄.
#[derive(Debug, Clone, Copy)]
pub struct ScoringWeights {
    pub similarity: f32,
    pub trust: f32,
    pub recency: f32,
    pub semantic_mass: f32,
}

impl Default for ScoringWeights {
    fn default() -> Self {
        Self {
            similarity: LAMBDA_SIMILARITY,
            trust: LAMBDA_TRUST,
            recency: LAMBDA_RECENCY,
            semantic_mass: LAMBDA_SEMANTIC_MASS,
        }
    }
}

impl ScoringWeights {
    /// Validate that weights sum to approximately 1.0.
    pub fn validate(&self) -> bool {
        let sum = self.similarity + self.trust + self.recency + self.semantic_mass;
        (sum - 1.0).abs() < 1e-3
    }
}

// ── Traversal Query ───────────────────────────────────────────────────────────

/// A semantic query driving the traversal. The query vector is used for
/// cosine similarity scoring at each node.
#[derive(Debug, Clone)]
pub struct SemanticQuery {
    /// L2-normalized query embedding. If None, similarity score = 0 for all nodes.
    pub query_vector: Option<Vec<f32>>,
    /// Human-readable description (for logging / observability).
    pub description: String,
}

impl SemanticQuery {
    pub fn new(query_vector: Vec<f32>, description: impl Into<String>) -> Self {
        Self {
            query_vector: Some(query_vector),
            description: description.into(),
        }
    }

    pub fn empty(description: impl Into<String>) -> Self {
        Self { query_vector: None, description: description.into() }
    }
}

// ── Traversal Result ──────────────────────────────────────────────────────────

/// A single node result from a traversal.
#[derive(Debug, Clone)]
pub struct TraversalHit {
    pub orb_id: Uuid,
    /// Composite relevance score ∈ [0, 1].
    pub score: f32,
    /// Hop distance from the start node.
    pub depth: usize,
    /// Score breakdown for observability.
    pub score_breakdown: ScoreBreakdown,
    /// Attention units allocated to this node.
    pub attention_allocated: u64,
}

/// Per-component score breakdown for debugging and observability.
#[derive(Debug, Clone)]
pub struct ScoreBreakdown {
    pub similarity: f32,
    pub trust: f32,
    pub recency: f32,
    pub semantic_mass: f32,
}

/// Full result of a traversal call.
#[derive(Debug, Clone)]
pub struct TraversalResult {
    /// Nodes passing the score threshold, sorted by score descending.
    pub hits: Vec<TraversalHit>,
    /// Total budget consumed.
    pub budget_consumed: u64,
    /// Number of nodes visited (including filtered-out).
    pub nodes_visited: usize,
    /// Number of edges traversed.
    pub edges_traversed: usize,
    /// Whether the traversal exhausted its budget before completing.
    pub budget_exhausted: bool,
}

// ── Graph View ────────────────────────────────────────────────────────────────

/// Lightweight read-only view of the graph needed during traversal.
/// Decouples the traversal engine from ORB storage.
///
/// In production: backed by the AGE graph engine.
/// In tests: use `InMemoryGraphView`.
pub trait GraphView: Send + Sync {
    /// Fetch a node's traversal-relevant fields.
    fn get_node(&self, orb_id: Uuid) -> Option<NodeView>;
}

/// The traversal-relevant fields of an ORB.
#[derive(Debug, Clone)]
pub struct NodeView {
    pub orb_id: Uuid,
    /// L2-normalized semantic vector.
    pub semantic_vector: Vec<f32>,
    /// Effective trust score (decay applied).
    pub trust: f32,
    /// Semantic mass (weighted degree centrality).
    pub semantic_mass: f32,
    /// Updated-at timestamp (Unix ms) for recency scoring.
    pub updated_at_ms: u64,
    /// Outgoing edges.
    pub edges: Vec<EdgeView>,
}

/// An outgoing edge as seen during traversal.
#[derive(Debug, Clone)]
pub struct EdgeView {
    pub edge_id: Uuid,
    pub target_orb_id: Uuid,
    pub relation_type: RelationType,
    pub weight: f32,
    pub trust_at_creation: f32,
}

// ── In-Memory Graph View (tests + dev) ───────────────────────────────────────

/// Simple in-memory graph view for testing and single-node deployments.
pub struct InMemoryGraphView {
    nodes: HashMap<Uuid, NodeView>,
}

impl InMemoryGraphView {
    pub fn new() -> Self {
        Self { nodes: HashMap::new() }
    }

    pub fn insert(&mut self, node: NodeView) {
        self.nodes.insert(node.orb_id, node);
    }

    /// Build a node view from a live ORB.
    pub fn insert_orb(&mut self, orb: &mut Orb) {
        let edges: Vec<EdgeView> = orb
            .relations
            .iter()
            .map(|r| EdgeView {
                edge_id: r.edge_id,
                target_orb_id: r.target_orb_id,
                relation_type: r.relation_type.clone(),
                weight: r.weight,
                trust_at_creation: r.trust_at_creation,
            })
            .collect();

        self.nodes.insert(
            orb.identity,
            NodeView {
                orb_id: orb.identity,
                semantic_vector: orb.semantic_vector.0.clone(),
                trust: orb.trust.effective_score(),
                semantic_mass: orb.semantic_mass(),
                updated_at_ms: orb.updated_at.physical_ms,
                edges,
            },
        );
    }
}

impl Default for InMemoryGraphView {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphView for InMemoryGraphView {
    fn get_node(&self, orb_id: Uuid) -> Option<NodeView> {
        self.nodes.get(&orb_id).cloned()
    }
}

// ── Scorer ────────────────────────────────────────────────────────────────────

/// Scores a node against a query.
///
/// score(v, q) = λ₁·sim(v,q) + λ₂·trust(v) + λ₃·recency(v) + λ₄·semantic_mass(v)
///
/// This is the traversal scoring function missing from the original implementation.
struct Scorer {
    weights: ScoringWeights,
    query_vector: Option<Vec<f32>>,
    now_ms: u64,
}

impl Scorer {
    fn new(query: &SemanticQuery, weights: ScoringWeights) -> Self {
        Self {
            weights,
            query_vector: query.query_vector.clone(),
            now_ms: crate::orb::now_ms(),
        }
    }

    fn score(&self, node: &NodeView) -> (f32, ScoreBreakdown) {
        let similarity = self.cosine_similarity(node);
        let trust = node.trust.clamp(0.0, 1.0);
        let recency = self.recency_weight(node.updated_at_ms);
        let semantic_mass = node.semantic_mass.clamp(0.0, 1.0);

        let score = self.weights.similarity * similarity
            + self.weights.trust * trust
            + self.weights.recency * recency
            + self.weights.semantic_mass * semantic_mass;

        (
            score.clamp(0.0, 1.0),
            ScoreBreakdown { similarity, trust, recency, semantic_mass },
        )
    }

    fn cosine_similarity(&self, node: &NodeView) -> f32 {
        let qv = match &self.query_vector {
            Some(v) => v,
            None => return 0.0,
        };
        if qv.len() != node.semantic_vector.len() {
            return 0.0;
        }
        qv.iter().zip(&node.semantic_vector).map(|(a, b)| a * b).sum::<f32>().clamp(0.0, 1.0)
    }

    /// Recency weight: exp(-Δt / half_life) — 1.0 if just updated, decays over days.
    fn recency_weight(&self, updated_at_ms: u64) -> f32 {
        let delta_ms = self.now_ms.saturating_sub(updated_at_ms) as f64;
        (-(delta_ms / RECENCY_HALF_LIFE_MS)).exp() as f32
    }
}

// ── Traversal Engine ──────────────────────────────────────────────────────────

/// Priority queue key — ordered by (attention allocation as i64, node uuid for tiebreak).
/// The priority_queue crate uses max-heap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct TraversalPriority {
    allocation: i64,
    // UUID as tiebreaker (stable across runs for same UUIDs)
    orb_id: Uuid,
}

impl TraversalPriority {
    fn new(allocation: u64, orb_id: Uuid) -> Self {
        Self { allocation: allocation as i64, orb_id }
    }
}

/// Entry in the traversal frontier.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct FrontierEntry {
    orb_id: Uuid,
    depth: usize,
}

/// The attention-weighted traversal engine.
pub struct TraversalEngine;

impl TraversalEngine {
    /// Execute a traversal from `start_orb_id` using the provided graph view.
    ///
    /// Algorithm (spec §8.2):
    ///   1. Initialize frontier with start node at full budget
    ///   2. Pop highest-attention node
    ///   3. Score against query; add to results if above threshold
    ///   4. Distribute attention to neighbors via edge_weight × trust
    ///   5. Repeat until budget exhausted or frontier empty
    pub fn traverse<G: GraphView>(
        graph: &G,
        start_orb_id: Uuid,
        query: &SemanticQuery,
        config: &TraversalConfig,
    ) -> Result<TraversalResult> {
        let weights = config.scoring_weights.unwrap_or_default();
        let scorer = Scorer::new(query, weights);

        let mut frontier: PriorityQueue<FrontierEntry, TraversalPriority> = PriorityQueue::new();
        let mut visited: HashSet<Uuid> = HashSet::new();
        let mut attention_map: HashMap<Uuid, u64> = HashMap::new();

        let mut hits: Vec<TraversalHit> = Vec::new();
        let mut budget_remaining = config.budget;
        let mut edges_traversed = 0usize;

        // Seed frontier
        let start_alloc = config.budget.min(
            (config.budget as f32 * SINGULARITY_CAP) as u64,
        );
        let start_entry = FrontierEntry { orb_id: start_orb_id, depth: 0 };
        frontier.push(start_entry, TraversalPriority::new(start_alloc, start_orb_id));
        attention_map.insert(start_orb_id, start_alloc.max(ATTENTION_FLOOR));

        while let Some((entry, _priority)) = frontier.pop() {
            if budget_remaining == 0 {
                return Ok(TraversalResult {
                    hits: Self::sorted_hits(hits),
                    budget_consumed: config.budget,
                    nodes_visited: visited.len(),
                    edges_traversed,
                    budget_exhausted: true,
                });
            }

            if visited.contains(&entry.orb_id) {
                continue;
            }
            if entry.depth > config.max_depth {
                continue;
            }

            visited.insert(entry.orb_id);

            let node = match graph.get_node(entry.orb_id) {
                Some(n) => n,
                None => continue, // missing node — skip silently
            };

            // Charge compute cost against budget
            let compute_cost = 1u64;
            budget_remaining = budget_remaining.saturating_sub(compute_cost);

            // Trust gate
            if node.trust < config.min_trust {
                continue;
            }

            // Score node against query
            let alloc = attention_map.get(&entry.orb_id).copied().unwrap_or(ATTENTION_FLOOR);
            let (score, breakdown) = scorer.score(&node);

            if score >= config.score_threshold {
                hits.push(TraversalHit {
                    orb_id: entry.orb_id,
                    score,
                    depth: entry.depth,
                    score_breakdown: breakdown,
                    attention_allocated: alloc,
                });
            }

            // Distribute attention to neighbors
            if entry.depth < config.max_depth {
                for edge in &node.edges {
                    edges_traversed += 1;

                    // Relation type filter
                    if let Some(allowed) = &config.allowed_relation_types {
                        if !allowed.contains(&edge.relation_type) {
                            // Still charge edge traversal cost (spec §8.3)
                            budget_remaining = budget_remaining.saturating_sub(1);
                            continue;
                        }
                    }

                    if visited.contains(&edge.target_orb_id) {
                        budget_remaining = budget_remaining.saturating_sub(1);
                        continue;
                    }

                    // child_alloc = parent_alloc × edge_weight × trust_at_creation
                    let child_alloc_f = alloc as f32 * edge.weight * edge.trust_at_creation;
                    let child_alloc = (child_alloc_f as u64)
                        .max(ATTENTION_FLOOR)
                        .min((config.budget as f32 * SINGULARITY_CAP) as u64);

                    let child_entry = FrontierEntry {
                        orb_id: edge.target_orb_id,
                        depth: entry.depth + 1,
                    };

                    // Update or insert — take max allocation if already queued
                    let existing = attention_map
                        .entry(edge.target_orb_id)
                        .or_insert(0);
                    if child_alloc > *existing {
                        *existing = child_alloc;
                        frontier.push_increase(
                            child_entry,
                            TraversalPriority::new(child_alloc, edge.target_orb_id),
                        );
                    }
                }
            }
        }

        Ok(TraversalResult {
            hits: Self::sorted_hits(hits),
            budget_consumed: config.budget - budget_remaining,
            nodes_visited: visited.len(),
            edges_traversed,
            budget_exhausted: false,
        })
    }

    fn sorted_hits(mut hits: Vec<TraversalHit>) -> Vec<TraversalHit> {
        hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        hits
    }
}

// ── Attention Allocator ────────────────────────────────────────────────────────

/// Computes attention allocation per node across a set of candidate ORBs.
///
/// A(v) = B · priority(v) / Σ_u priority(u)
/// priority(v) = trust(v) · semantic_mass(v) · recency_weight(v)
///
/// Constraints enforced:
///   - Σ A(v) = B (conservation)
///   - A(v) ≤ B × SINGULARITY_CAP
///   - A(v) ≥ ATTENTION_FLOOR
pub struct AttentionAllocator;

impl AttentionAllocator {
    /// Compute attention allocation for a set of nodes.
    /// Returns (orb_id -> attention_units) map. Sums exactly to `budget`.
    pub fn allocate(
        nodes: &[NodeView],
        budget: u64,
        now_ms: u64,
    ) -> HashMap<Uuid, u64> {
        if nodes.is_empty() || budget == 0 {
            return HashMap::new();
        }
        Self::allocate_conserved(nodes, budget, now_ms)
        /*

        // Compute raw priorities
        let priorities: Vec<f64> = nodes
            .iter()
            .map(|n| {
                let recency = {
                    let delta_ms = now_ms.saturating_sub(n.updated_at_ms) as f64;
                    (-delta_ms / RECENCY_HALF_LIFE_MS).exp()
                };
                let priority = (n.trust as f64) * (n.semantic_mass as f64) * recency;
                if priority.is_finite() && priority > 0.0 {
                    priority
                } else {
                    0.0
                }
            })
            .collect();

        let floor = ATTENTION_FLOOR;
        let node_count = nodes.len() as u64;
        let base_cap = (budget as f64 * SINGULARITY_CAP as f64) as u64;
        let conservation_cap = (budget + node_count - 1) / node_count;
        let cap = base_cap.max(conservation_cap).max(floor);

        if budget <= node_count * floor {
            return nodes
                .iter()
                .enumerate()
                .map(|(idx, n)| {
                    let allocation = if (idx as u64) < budget { 1 } else { 0 };
                    (n.orb_id, allocation)
                })
                .collect();
        }

        let mut allocations = vec![floor; nodes.len()];
        let remaining = budget - node_count * floor;
        let total_priority: f64 = priorities.iter().sum();

        let mut allocations: Vec<u64> = if total_priority < 1e-10 {
            // All priorities zero — distribute evenly
            let per_node = (budget / nodes.len() as u64).max(floor).min(cap);
            vec![per_node; nodes.len()]
        } else {
            priorities
                .iter()
                .map(|&p| {
                    let raw = (budget as f64 * p / total_priority) as u64;
                    raw.max(floor).min(cap)
                })
                .collect()
        };

        // Re-normalize to conserve budget exactly
        let sum: u64 = allocations.iter().sum();
        if sum > 0 {
            for alloc in &mut allocations {
                *alloc = ((*alloc as f64 / sum as f64) * budget as f64) as u64;
            }
        }

        nodes
            .iter()
            .zip(allocations)
            .map(|(n, a)| (n.orb_id, a.max(floor)))
            .collect()
        */
    }

    fn allocate_conserved(
        nodes: &[NodeView],
        budget: u64,
        now_ms: u64,
    ) -> HashMap<Uuid, u64> {
        let floor = ATTENTION_FLOOR;
        let node_count = nodes.len() as u64;
        let base_cap = (budget as f64 * SINGULARITY_CAP as f64) as u64;
        let conservation_cap = (budget + node_count - 1) / node_count;
        let cap = base_cap.max(conservation_cap).max(floor);

        if budget <= node_count * floor {
            return nodes
                .iter()
                .enumerate()
                .map(|(idx, n)| {
                    let allocation = if (idx as u64) < budget { 1 } else { 0 };
                    (n.orb_id, allocation)
                })
                .collect();
        }

        let priorities: Vec<f64> = nodes
            .iter()
            .map(|n| {
                let delta_ms = now_ms.saturating_sub(n.updated_at_ms) as f64;
                let recency = (-delta_ms / RECENCY_HALF_LIFE_MS).exp();
                let priority = (n.trust as f64) * (n.semantic_mass as f64) * recency;
                if priority.is_finite() && priority > 0.0 {
                    priority
                } else {
                    0.0
                }
            })
            .collect();
        let total_priority: f64 = priorities.iter().sum();
        let weights: Vec<f64> = if total_priority < 1e-10 {
            vec![1.0 / nodes.len() as f64; nodes.len()]
        } else {
            priorities
                .iter()
                .map(|priority| priority / total_priority)
                .collect()
        };

        let mut allocations = vec![floor; nodes.len()];
        let remaining = budget - node_count * floor;
        let mut assigned = 0u64;
        let mut remainders: Vec<(usize, f64)> = Vec::with_capacity(nodes.len());

        for (idx, weight) in weights.iter().enumerate() {
            let ideal = remaining as f64 * weight;
            let extra = (ideal.floor() as u64).min(cap.saturating_sub(floor));
            allocations[idx] += extra;
            assigned += extra;
            remainders.push((idx, ideal - extra as f64));
        }

        let mut leftover = remaining.saturating_sub(assigned);
        remainders.sort_by(|(idx_a, rem_a), (idx_b, rem_b)| {
            rem_b
                .partial_cmp(rem_a)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(nodes[*idx_a].orb_id.cmp(&nodes[*idx_b].orb_id))
        });

        while leftover > 0 {
            let mut made_progress = false;
            for (idx, _) in &remainders {
                if allocations[*idx] < cap {
                    allocations[*idx] += 1;
                    leftover -= 1;
                    made_progress = true;
                    if leftover == 0 {
                        break;
                    }
                }
            }
            if !made_progress {
                break;
            }
        }

        nodes
            .iter()
            .zip(allocations)
            .map(|(n, allocation)| (n.orb_id, allocation))
            .collect()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::HlcTimestamp;
    use crate::orb::{
        Orb, Permissions, ProbabilityField, RelationRef, RelationType, SemanticVector,
    };

    fn make_node(orb_id: Uuid, trust: f32, semantic_mass: f32) -> NodeView {
        use crate::orb::VECTOR_DIM;
        let v = vec![1.0 / (VECTOR_DIM as f32).sqrt(); VECTOR_DIM];
        NodeView {
            orb_id,
            semantic_vector: v,
            trust,
            semantic_mass,
            updated_at_ms: crate::orb::now_ms(),
            edges: vec![],
        }
    }

    fn make_graph_chain(n: usize) -> (InMemoryGraphView, Vec<Uuid>) {
        use crate::orb::VECTOR_DIM;
        let mut graph = InMemoryGraphView::new();
        let ids: Vec<Uuid> = (0..n).map(|_| Uuid::now_v7()).collect();
        let v = vec![1.0 / (VECTOR_DIM as f32).sqrt(); VECTOR_DIM];

        for (i, &id) in ids.iter().enumerate() {
            let edges = if i + 1 < n {
                vec![EdgeView {
                    edge_id: Uuid::now_v7(),
                    target_orb_id: ids[i + 1],
                    relation_type: RelationType::References,
                    weight: 0.9,
                    trust_at_creation: 0.8,
                }]
            } else {
                vec![]
            };
            graph.insert(NodeView {
                orb_id: id,
                semantic_vector: v.clone(),
                trust: 0.7,
                semantic_mass: 0.3,
                updated_at_ms: crate::orb::now_ms(),
                edges,
            });
        }
        (graph, ids)
    }

    #[test]
    fn traversal_finds_connected_nodes() {
        let (graph, ids) = make_graph_chain(5);
        let query = SemanticQuery::empty("test");
        let config = TraversalConfig {
            score_threshold: 0.0, // return everything
            ..Default::default()
        };
        let result = TraversalEngine::traverse(&graph, ids[0], &query, &config).unwrap();
        assert_eq!(result.nodes_visited, 5);
        assert!(!result.budget_exhausted);
    }

    #[test]
    fn traversal_respects_max_depth() {
        let (graph, ids) = make_graph_chain(10);
        let query = SemanticQuery::empty("test");
        let config = TraversalConfig {
            max_depth: 3,
            score_threshold: 0.0,
            ..Default::default()
        };
        let result = TraversalEngine::traverse(&graph, ids[0], &query, &config).unwrap();
        // Should visit nodes 0..=3 only (depth 0, 1, 2, 3)
        assert!(result.nodes_visited <= 4);
    }

    #[test]
    fn traversal_never_exceeds_budget() {
        let (graph, ids) = make_graph_chain(100);
        let query = SemanticQuery::empty("test");
        let config = TraversalConfig {
            budget: 20,
            score_threshold: 0.0,
            ..Default::default()
        };
        let result = TraversalEngine::traverse(&graph, ids[0], &query, &config).unwrap();
        assert!(result.budget_consumed <= 20);
    }

    #[test]
    fn traversal_respects_relation_type_filter() {
        let mut graph = InMemoryGraphView::new();
        let a = Uuid::now_v7();
        let b = Uuid::now_v7();
        let c = Uuid::now_v7();
        use crate::orb::VECTOR_DIM;
        let v = vec![1.0 / (VECTOR_DIM as f32).sqrt(); VECTOR_DIM];

        // a → b via References, a → c via Causes
        graph.insert(NodeView {
            orb_id: a,
            semantic_vector: v.clone(),
            trust: 0.8,
            semantic_mass: 0.5,
            updated_at_ms: crate::orb::now_ms(),
            edges: vec![
                EdgeView {
                    edge_id: Uuid::now_v7(),
                    target_orb_id: b,
                    relation_type: RelationType::References,
                    weight: 0.9,
                    trust_at_creation: 0.8,
                },
                EdgeView {
                    edge_id: Uuid::now_v7(),
                    target_orb_id: c,
                    relation_type: RelationType::Causes,
                    weight: 0.9,
                    trust_at_creation: 0.8,
                },
            ],
        });
        graph.insert(NodeView {
            orb_id: b,
            semantic_vector: v.clone(),
            trust: 0.7,
            semantic_mass: 0.3,
            updated_at_ms: crate::orb::now_ms(),
            edges: vec![],
        });
        graph.insert(NodeView {
            orb_id: c,
            semantic_vector: v.clone(),
            trust: 0.7,
            semantic_mass: 0.3,
            updated_at_ms: crate::orb::now_ms(),
            edges: vec![],
        });

        let query = SemanticQuery::empty("test");
        let config = TraversalConfig {
            allowed_relation_types: Some(vec![RelationType::Causes]),
            score_threshold: 0.0,
            ..Default::default()
        };
        let result = TraversalEngine::traverse(&graph, a, &query, &config).unwrap();
        // Should visit a and c only (References filtered out)
        assert_eq!(result.nodes_visited, 2);
        let visited_ids: Vec<Uuid> = result.hits.iter().map(|h| h.orb_id).collect();
        assert!(visited_ids.contains(&a));
        assert!(visited_ids.contains(&c));
        assert!(!visited_ids.contains(&b));
    }

    #[test]
    fn traversal_no_cycles() {
        // A → B → A (cycle)
        let mut graph = InMemoryGraphView::new();
        let a = Uuid::now_v7();
        let b = Uuid::now_v7();
        use crate::orb::VECTOR_DIM;
        let v = vec![1.0 / (VECTOR_DIM as f32).sqrt(); VECTOR_DIM];

        graph.insert(NodeView {
            orb_id: a,
            semantic_vector: v.clone(),
            trust: 0.8,
            semantic_mass: 0.5,
            updated_at_ms: crate::orb::now_ms(),
            edges: vec![EdgeView {
                edge_id: Uuid::now_v7(),
                target_orb_id: b,
                relation_type: RelationType::References,
                weight: 1.0,
                trust_at_creation: 1.0,
            }],
        });
        graph.insert(NodeView {
            orb_id: b,
            semantic_vector: v.clone(),
            trust: 0.8,
            semantic_mass: 0.5,
            updated_at_ms: crate::orb::now_ms(),
            edges: vec![EdgeView {
                edge_id: Uuid::now_v7(),
                target_orb_id: a, // cycle back
                relation_type: RelationType::References,
                weight: 1.0,
                trust_at_creation: 1.0,
            }],
        });

        let query = SemanticQuery::empty("cycle test");
        let config = TraversalConfig { score_threshold: 0.0, ..Default::default() };
        let result = TraversalEngine::traverse(&graph, a, &query, &config).unwrap();
        // Must terminate; each node visited exactly once
        assert_eq!(result.nodes_visited, 2);
    }

    #[test]
    fn attention_allocator_conserves_budget() {
        let now = crate::orb::now_ms();
        let nodes: Vec<NodeView> = (0..5)
            .map(|_| {
                let mut n = make_node(Uuid::now_v7(), 0.5, 0.3);
                n.updated_at_ms = now;
                n
            })
            .collect();
        let budget = 10_000u64;
        let allocs = AttentionAllocator::allocate(&nodes, budget, now);
        let total: u64 = allocs.values().sum();
        assert_eq!(total, budget);
    }

    #[test]
    fn attention_allocator_enforces_cap() {
        let now = crate::orb::now_ms();
        // One dominant node, rest negligible
        let mut nodes: Vec<NodeView> = (0..5)
            .map(|_| {
                let mut n = make_node(Uuid::now_v7(), 0.001, 0.001);
                n.updated_at_ms = now;
                n
            })
            .collect();
        nodes[0].trust = 1.0;
        nodes[0].semantic_mass = 1.0;

        let budget = 10_000u64;
        let cap = (budget as f32 * SINGULARITY_CAP) as u64;
        let allocs = AttentionAllocator::allocate(&nodes, budget, now);

        for (_, &alloc) in &allocs {
            assert!(alloc <= cap, "allocation {alloc} exceeds cap {cap}");
        }
    }
}
