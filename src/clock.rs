use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::{PiriaError, Result};

pub const MAX_CLOCK_SKEW_MS: u64 = 500;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct HlcTimestamp {
    pub physical_ms: u64,
    pub logical: u32,
}

impl HlcTimestamp {
    pub fn new(physical_ms: u64, logical: u32) -> Self {
        Self { physical_ms, logical }
    }

    pub fn zero() -> Self {
        Self { physical_ms: 0, logical: 0 }
    }

    pub fn now() -> Self {
        Self { physical_ms: wall_clock_ms(), logical: 0 }
    }
}

pub struct HlcClock {
    state: Mutex<HlcTimestamp>,
}

impl HlcClock {
    pub fn new() -> Self {
        Self { state: Mutex::new(HlcTimestamp::zero()) }
    }

    pub fn now(&self) -> Result<HlcTimestamp> {
        let wall = wall_clock_ms();
        let mut state = self.state.lock().unwrap();
        let next = if wall > state.physical_ms {
            HlcTimestamp::new(wall, 0)
        } else {
            HlcTimestamp::new(state.physical_ms, state.logical.saturating_add(1))
        };
        *state = next;
        Ok(next)
    }

    pub fn receive(&self, remote: &HlcTimestamp) -> Result<HlcTimestamp> {
        let wall = wall_clock_ms();
        if remote.physical_ms > wall + MAX_CLOCK_SKEW_MS {
            return Err(PiriaError::Internal("Clock skew exceeded".into()));
        }

        let mut state = self.state.lock().unwrap();
        let local_phys = state.physical_ms.max(wall);
        let next = if local_phys > remote.physical_ms {
            HlcTimestamp::new(local_phys, state.logical.saturating_add(1))
        } else if remote.physical_ms > local_phys {
            HlcTimestamp::new(remote.physical_ms, remote.logical.saturating_add(1))
        } else {
            HlcTimestamp::new(local_phys, state.logical.max(remote.logical).saturating_add(1))
        };
        *state = next;
        Ok(next)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct VectorClock(pub HashMap<String, u64>);

impl VectorClock {
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    pub fn tick(&mut self, node_id: &str) {
        let entry = self.0.entry(node_id.to_owned()).or_insert(0);
        *entry += 1;
    }

    pub fn merged_with(&self, other: &VectorClock) -> VectorClock {
        let mut merged = self.0.clone();
        for (node_id, counter) in &other.0 {
            let entry = merged.entry(node_id.clone()).or_insert(0);
            *entry = (*entry).max(*counter);
        }
        VectorClock(merged)
    }
}

fn wall_clock_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
