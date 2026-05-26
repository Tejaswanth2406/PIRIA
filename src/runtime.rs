/// Single-node PIRIA runtime.
///
/// This layer wires ORBs, event logs, entropy monitoring, traversal, and local
/// persistence into one usable in-process API. It is intentionally not a
/// distributed or LLM-integrated runtime.
use std::{collections::HashMap, path::Path};

use uuid::Uuid;

use crate::{
    clock::{HlcClock, HlcTimestamp, VectorClock},
    entropy::{
        EntropyAlert, EntropyMonitor, FieldEntropySnapshot, OrbEntropySample,
    },
    error::{PiriaError, Result},
    event::{ActorId, Event, EventBuilder, EventPayload, OrbEventLog},
    orb::{
        Orb, Permissions, ProbabilityField, RelationRef, RelationType, SemanticState,
        SemanticVector,
    },
    storage::JsonGraphStore,
    traversal::{
        InMemoryGraphView, SemanticQuery, TraversalConfig, TraversalEngine,
        TraversalResult,
    },
};

pub struct PiriaRuntime {
    node_id: String,
    clock: HlcClock,
    vector_clock: VectorClock,
    event_builder: EventBuilder,
    orbs: HashMap<Uuid, Orb>,
    event_logs: HashMap<Uuid, OrbEventLog>,
    entropy_monitor: EntropyMonitor,
    store: Option<JsonGraphStore>,
}

impl PiriaRuntime {
    pub fn new(node_id: impl Into<String>) -> Self {
        let node_id = node_id.into();
        Self {
            event_builder: EventBuilder::new(node_id.clone()),
            node_id,
            clock: HlcClock::new(),
            vector_clock: VectorClock::new(),
            orbs: HashMap::new(),
            event_logs: HashMap::new(),
            entropy_monitor: EntropyMonitor::new(),
            store: None,
        }
    }

    pub fn open_persistent(
        node_id: impl Into<String>,
        root: impl AsRef<Path>,
    ) -> Result<Self> {
        let mut runtime = Self::new(node_id);
        let store = JsonGraphStore::open(root)?;

        if let Some(snapshot) = store.load_snapshot()? {
            for orb in snapshot.orbs {
                orb.verify_integrity()?;
                runtime.orbs.insert(orb.identity, orb);
            }
        }

        runtime.event_logs = store.load_event_logs()?;
        runtime.store = Some(store);
        Ok(runtime)
    }

    pub fn node_id(&self) -> &str {
        &self.node_id
    }

    pub fn orb_count(&self) -> usize {
        self.orbs.len()
    }

    pub fn event_count(&self) -> usize {
        self.event_logs.values().map(OrbEventLog::len).sum()
    }

    pub fn get_orb(&self, orb_id: Uuid) -> Option<&Orb> {
        self.orbs.get(&orb_id)
    }

    pub fn get_orb_mut(&mut self, orb_id: Uuid) -> Option<&mut Orb> {
        self.orbs.get_mut(&orb_id)
    }

    pub fn event_log(&self, orb_id: Uuid) -> Option<&OrbEventLog> {
        self.event_logs.get(&orb_id)
    }

    pub fn create_orb(
        &mut self,
        semantic_vector: SemanticVector,
        probability_field: ProbabilityField,
        permissions: Permissions,
    ) -> Result<Uuid> {
        let timestamp = self.clock.now()?;
        let mut orb = Orb::new(
            semantic_vector,
            probability_field,
            permissions,
            timestamp,
        );
        let orb_id = orb.identity;
        let event = self.build_event(
            orb_id,
            ActorId::System,
            EventPayload::OrbCreated {
                initial_trust: orb.trust.score,
                permissions_bits: orb.permissions.bits(),
                probability_categories: orb.probability_field.categories(),
            },
            timestamp,
        )?;

        self.append_event(event.clone())?;
        orb.record_event(event.event_id);
        self.orbs.insert(orb_id, orb);
        self.persist_snapshot()?;
        Ok(orb_id)
    }

    pub fn add_relation(
        &mut self,
        source_orb_id: Uuid,
        target_orb_id: Uuid,
        relation_type: RelationType,
        weight: f32,
    ) -> Result<Uuid> {
        let timestamp = self.clock.now()?;
        let trust_at_creation = self
            .orbs
            .get(&source_orb_id)
            .ok_or(PiriaError::OrbNotFound(source_orb_id))?
            .trust
            .effective_score();
        if !self.orbs.contains_key(&target_orb_id) {
            return Err(PiriaError::OrbNotFound(target_orb_id));
        }

        let relation = RelationRef::new(
            target_orb_id,
            relation_type.clone(),
            weight,
            trust_at_creation,
            timestamp,
        );
        let edge_id = relation.edge_id;
        self.orbs
            .get_mut(&source_orb_id)
            .ok_or(PiriaError::OrbNotFound(source_orb_id))?
            .add_relation(relation, timestamp)?;

        let event = self.build_event(
            source_orb_id,
            ActorId::System,
            EventPayload::RelationAdded {
                target_orb_id,
                relation_type: format!("{relation_type:?}"),
                weight: weight.clamp(0.0, 1.0),
                edge_id,
            },
            timestamp,
        )?;
        self.append_event_and_record(source_orb_id, event)?;
        self.persist_snapshot()?;
        Ok(edge_id)
    }

    pub fn update_probability_field(
        &mut self,
        orb_id: Uuid,
        observations: &[f32],
    ) -> Result<Vec<EntropyAlert>> {
        let timestamp = self.clock.now()?;
        let previous_entropy = self
            .orbs
            .get_mut(&orb_id)
            .ok_or(PiriaError::OrbNotFound(orb_id))?
            .entropy();

        let orb = self
            .orbs
            .get_mut(&orb_id)
            .ok_or(PiriaError::OrbNotFound(orb_id))?;
        orb.update_probability_field(observations, timestamp)?;
        let new_entropy = orb.entropy();
        let new_alpha = orb.probability_field.alpha.clone();

        let event = self.build_event(
            orb_id,
            ActorId::System,
            EventPayload::BeliefRevised {
                new_alpha,
                previous_entropy,
                new_entropy,
            },
            timestamp,
        )?;
        self.append_event_and_record(orb_id, event)?;
        let alerts = self.observe_entropy();
        self.persist_snapshot()?;
        Ok(alerts)
    }

    pub fn transition_orb(
        &mut self,
        orb_id: Uuid,
        next: SemanticState,
    ) -> Result<()> {
        let timestamp = self.clock.now()?;
        let from_state = self
            .orbs
            .get(&orb_id)
            .ok_or(PiriaError::OrbNotFound(orb_id))?
            .semantic_state;
        self.orbs
            .get_mut(&orb_id)
            .ok_or(PiriaError::OrbNotFound(orb_id))?
            .transition_to(next, timestamp)?;

        let event = self.build_event(
            orb_id,
            ActorId::System,
            EventPayload::OrbStateChanged {
                from_state,
                to_state: next,
            },
            timestamp,
        )?;
        self.append_event_and_record(orb_id, event)?;
        self.persist_snapshot()
    }

    pub fn traverse(
        &self,
        start_orb_id: Uuid,
        query: &SemanticQuery,
        config: &TraversalConfig,
    ) -> Result<TraversalResult> {
        let graph = self.graph_view();
        TraversalEngine::traverse(&graph, start_orb_id, query, config)
    }

    pub fn observe_entropy(&mut self) -> Vec<EntropyAlert> {
        let snapshot = self.entropy_snapshot();
        self.entropy_monitor.ingest_snapshot(snapshot)
    }

    pub fn entropy_snapshot(&mut self) -> FieldEntropySnapshot {
        let mut sum = 0.0;
        let mut count = 0u64;

        for orb in self.orbs.values_mut() {
            if orb.semantic_state != SemanticState::Active {
                continue;
            }
            let entropy = orb.entropy();
            let max_entropy = orb.probability_field.max_entropy_bits();
            sum += OrbEntropySample {
                orb_id: orb.identity,
                entropy_bits: entropy,
                max_entropy_bits: max_entropy,
                sampled_at_ms: crate::orb::now_ms(),
            }
            .normalized();
            count += 1;
        }

        let mean = if count == 0 { 0.0 } else { sum / count as f32 };
        FieldEntropySnapshot::new(mean, count, crate::orb::now_ms())
    }

    pub fn graph_view(&self) -> InMemoryGraphView {
        let mut graph = InMemoryGraphView::new();
        for orb in self.orbs.values().cloned() {
            let mut orb = orb;
            graph.insert_orb(&mut orb);
        }
        graph
    }

    pub fn persist_snapshot(&self) -> Result<()> {
        if let Some(store) = &self.store {
            store.save_snapshot(self.orbs.values())?;
        }
        Ok(())
    }

    fn build_event(
        &mut self,
        orb_id: Uuid,
        actor_id: ActorId,
        payload: EventPayload,
        timestamp: HlcTimestamp,
    ) -> Result<Event> {
        let prev_hash = self
            .event_logs
            .get(&orb_id)
            .map(OrbEventLog::head_hash)
            .unwrap_or([0u8; 32]);
        self.event_builder.build(
            orb_id,
            actor_id,
            payload,
            timestamp,
            self.vector_clock.clone(),
            prev_hash,
        )
    }

    fn append_event_and_record(&mut self, orb_id: Uuid, event: Event) -> Result<()> {
        let event_id = event.event_id;
        self.append_event(event)?;
        self.orbs
            .get_mut(&orb_id)
            .ok_or(PiriaError::OrbNotFound(orb_id))?
            .record_event(event_id);
        Ok(())
    }

    fn append_event(&mut self, event: Event) -> Result<()> {
        event.verify_hash()?;
        let orb_id = event.orb_id;
        self.event_logs
            .entry(orb_id)
            .or_insert_with(|| OrbEventLog::new(orb_id))
            .append(event.clone())?;
        if let Some(store) = &self.store {
            store.append_event(&event)?;
        }
        self.vector_clock = event.vector_clock.clone();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orb::VECTOR_DIM;

    fn vector() -> SemanticVector {
        SemanticVector::new(vec![1.0; VECTOR_DIM]).unwrap()
    }

    #[test]
    fn runtime_creates_orb_and_event() {
        let mut runtime = PiriaRuntime::new("node-test");
        let orb_id = runtime
            .create_orb(
                vector(),
                ProbabilityField::uniform(4).unwrap(),
                Permissions::default_writer(),
            )
            .unwrap();

        assert!(runtime.get_orb(orb_id).is_some());
        assert_eq!(runtime.event_count(), 1);
    }

    #[test]
    fn runtime_traverses_relation() {
        let mut runtime = PiriaRuntime::new("node-test");
        let a = runtime
            .create_orb(
                vector(),
                ProbabilityField::uniform(4).unwrap(),
                Permissions::default_writer(),
            )
            .unwrap();
        let b = runtime
            .create_orb(
                vector(),
                ProbabilityField::uniform(4).unwrap(),
                Permissions::default_writer(),
            )
            .unwrap();
        runtime
            .add_relation(a, b, RelationType::References, 0.9)
            .unwrap();

        let result = runtime
            .traverse(
                a,
                &SemanticQuery::empty("runtime traversal"),
                &TraversalConfig {
                    score_threshold: 0.0,
                    ..Default::default()
                },
            )
            .unwrap();

        assert_eq!(result.nodes_visited, 2);
    }
}
