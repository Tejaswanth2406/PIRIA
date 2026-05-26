/// Single-node durable storage.
///
/// This is deliberately simple: ORB state is snapshotted as JSON, while events
/// are appended to a JSONL write-ahead log. It gives the runtime crash recovery
/// and auditability without introducing Postgres/AGE yet.
use std::{
    collections::HashMap,
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    error::{PiriaError, Result},
    event::{Event, OrbEventLog},
    orb::{now_ms, Orb},
};

const SNAPSHOT_FILE: &str = "graph_snapshot.json";
const WAL_FILE: &str = "events.jsonl";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSnapshot {
    pub saved_at_ms: u64,
    pub orbs: Vec<Orb>,
}

#[derive(Debug, Clone)]
pub struct JsonGraphStore {
    root: PathBuf,
    snapshot_path: PathBuf,
    wal_path: PathBuf,
}

impl JsonGraphStore {
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root).map_err(io_error)?;
        let snapshot_path = root.join(SNAPSHOT_FILE);
        let wal_path = root.join(WAL_FILE);

        if !wal_path.exists() {
            File::create(&wal_path).map_err(io_error)?;
        }

        Ok(Self {
            root,
            snapshot_path,
            wal_path,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn snapshot_path(&self) -> &Path {
        &self.snapshot_path
    }

    pub fn wal_path(&self) -> &Path {
        &self.wal_path
    }

    pub fn save_snapshot<'a>(&self, orbs: impl IntoIterator<Item = &'a Orb>) -> Result<()> {
        let snapshot = GraphSnapshot {
            saved_at_ms: now_ms(),
            orbs: orbs.into_iter().cloned().collect(),
        };
        let bytes = serde_json::to_vec_pretty(&snapshot)?;
        let tmp_path = self.snapshot_path.with_extension("json.tmp");
        fs::write(&tmp_path, bytes).map_err(io_error)?;
        fs::rename(tmp_path, &self.snapshot_path).map_err(io_error)?;
        Ok(())
    }

    pub fn load_snapshot(&self) -> Result<Option<GraphSnapshot>> {
        if !self.snapshot_path.exists() {
            return Ok(None);
        }
        let bytes = fs::read(&self.snapshot_path).map_err(io_error)?;
        let snapshot = serde_json::from_slice(&bytes)?;
        Ok(Some(snapshot))
    }

    pub fn append_event(&self, event: &Event) -> Result<()> {
        event.verify_hash()?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.wal_path)
            .map_err(io_error)?;
        serde_json::to_writer(&mut file, event)?;
        file.write_all(b"\n").map_err(io_error)?;
        file.sync_data().map_err(io_error)?;
        Ok(())
    }

    pub fn read_events(&self) -> Result<Vec<Event>> {
        let file = File::open(&self.wal_path).map_err(io_error)?;
        let reader = BufReader::new(file);
        let mut events = Vec::new();

        for (idx, line) in reader.lines().enumerate() {
            let line = line.map_err(io_error)?;
            if line.trim().is_empty() {
                continue;
            }
            let event: Event = serde_json::from_str(&line).map_err(|err| {
                PiriaError::Internal(format!("WAL decode error at line {}: {err}", idx + 1))
            })?;
            event.verify_hash()?;
            events.push(event);
        }

        Ok(events)
    }

    pub fn load_event_logs(&self) -> Result<HashMap<Uuid, OrbEventLog>> {
        let mut logs: HashMap<Uuid, OrbEventLog> = HashMap::new();
        for event in self.read_events()? {
            let orb_id = event.orb_id;
            logs.entry(orb_id)
                .or_insert_with(|| OrbEventLog::new(orb_id))
                .append(event)?;
        }

        for log in logs.values() {
            log.verify_chain()?;
        }

        Ok(logs)
    }
}

fn io_error(error: std::io::Error) -> PiriaError {
    PiriaError::Internal(error.to_string())
}
