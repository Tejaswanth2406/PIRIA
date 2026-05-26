# PIRIA

PIRIA is the MVP runtime core for the Physics-Inspired Relational Intelligence Architecture.

## Project Structure

- `Cargo.toml` - Rust package manifest
- `src/main.rs` - runtime entrypoint
- `src/lib.rs` - library crate exposing core modules
- `src/orb.rs` - ORB model and integrity semantics
- `src/clock.rs` - HLC and vector clock utilities
- `src/event.rs` - event model and ORB event log
- `src/storage.rs` - JSON snapshot + JSONL event WAL for single-node persistence
- `src/runtime.rs` - single-node runtime wiring ORBs, events, traversal, entropy, and storage
- `src/traversal.rs` - attention-bounded graph traversal engine
- `src/crdt.rs` - CRDT merge state skeleton
- `src/entropy.rs` - entropy monitoring skeleton
- `src/error.rs` - shared error and result types

## Current Milestone

PIRIA now targets a single-node persistent cognitive graph:

- ORB graph state is snapshotted to `.piria/graph_snapshot.json`
- immutable events are appended to `.piria/events.jsonl`
- WAL events are hash-verified and reconstructed into per-ORB event logs on boot
- the runtime can create ORBs, add relations, update belief fields, traverse the graph, and observe entropy

LLM integration is intentionally out of scope for this milestone.

## Build

```bash
cargo build
```

## Run

```bash
cargo run
```

## Test

```bash
cargo test
```
