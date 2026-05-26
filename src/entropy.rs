/// Entropy Monitor - PIRIA MVP Spec section 9
///
/// Monitors Shannon entropy per ORB and mean normalized ORB entropy across the
/// active field. This is intentionally not a full graph/semantic field entropy:
/// it measures average local uncertainty, while topology-aware entropy belongs
/// in a later structural monitor.
///
/// The entropy monitor never globally minimizes entropy. It keeps uncertainty
/// bounded within acceptable ranges. High entropy proposes compression;
/// unbounded growth or collapse triggers governance alerts.
use std::collections::{HashSet, VecDeque};

// -- Thresholds ---------------------------------------------------------------

/// Per-ORB entropy warning threshold: normalized H > 0.8.
pub const ORB_ENTROPY_WARNING_FRACTION: f32 = 0.80;

/// Mean normalized ORB entropy warning threshold.
pub const MEAN_ORB_ENTROPY_WARNING: f32 = 0.60;

/// Mean normalized ORB entropy critical threshold.
pub const MEAN_ORB_ENTROPY_CRITICAL: f32 = 0.85;

/// Backward-compatible alias for older callers.
pub const FIELD_ENTROPY_WARNING: f32 = MEAN_ORB_ENTROPY_WARNING;

/// Backward-compatible alias for older callers.
pub const FIELD_ENTROPY_CRITICAL: f32 = MEAN_ORB_ENTROPY_CRITICAL;

/// Duration (ms) mean ORB entropy must exceed CRITICAL threshold to trigger
/// emergency compression.
pub const FIELD_ENTROPY_CRITICAL_SUSTAIN_MS: u64 = 10 * 60 * 1_000;

/// Samples farther apart than this cannot prove continuous critical entropy.
pub const FIELD_ENTROPY_MAX_SUSTAIN_SAMPLE_GAP_MS: u64 = 2 * 60 * 1_000;

/// Ignore derivative checks for intervals shorter than this to avoid tiny-dt
/// explosions from bursty sampling.
pub const MIN_ENTROPY_DERIVATIVE_INTERVAL_MS: u64 = 100;

/// Default max absolute rate of change in normalized entropy units per second.
pub const DEFAULT_ENTROPY_SPIKE_RATE: f32 = 0.05;

/// EMA weight for the newest mean entropy observation.
pub const DEFAULT_ENTROPY_EMA_ALPHA: f32 = 0.35;

/// Normal ORB creation throughput.
pub const CREATION_THROTTLE_NORMAL: f32 = 1.0;

/// Moderate backpressure while the mean ORB entropy is elevated.
pub const CREATION_THROTTLE_WARNING: f32 = 0.7;

/// Severe backpressure for rapid entropy motion.
pub const CREATION_THROTTLE_SEVERE: f32 = 0.3;

/// Full stop reserved for sustained critical entropy.
pub const CREATION_THROTTLE_PAUSED: f32 = 0.0;

// -- Alert Severity -----------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertSeverity {
    Warning,
    Critical,
}

// -- Entropy Alert ------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct EntropyAlert {
    pub severity: AlertSeverity,
    pub kind: EntropyAlertKind,
    pub timestamp_ms: u64,
}

#[derive(Debug, Clone)]
pub enum EntropyAlertKind {
    /// A specific ORB's entropy exceeded the per-ORB warning threshold.
    OrbEntropyHigh {
        orb_id: uuid::Uuid,
        normalized_entropy: f32,
        threshold: f32,
    },
    /// Mean normalized ORB entropy exceeded threshold.
    MeanOrbEntropyHigh {
        mean_normalized_entropy: f32,
        threshold: f32,
    },
    /// Legacy alert shape retained for downstream callers that still construct
    /// field-level alerts outside this monitor.
    FieldEntropyHigh {
        field_entropy: f32,
        threshold: f32,
    },
    /// Positive entropy velocity exceeded threshold.
    EntropySpike {
        rate: f32,
        threshold: f32,
    },
    /// Negative entropy velocity exceeded threshold.
    EntropyCollapse {
        rate: f32,
        threshold: f32,
    },
    /// Mean ORB entropy exceeded critical level and has been sustained.
    SustainedCriticalEntropy {
        mean_normalized_entropy: f32,
        sustained_ms: u64,
    },
}

// -- ORB Entropy Sample -------------------------------------------------------

/// A single entropy observation for one ORB.
#[derive(Debug, Clone)]
pub struct OrbEntropySample {
    pub orb_id: uuid::Uuid,
    /// Shannon entropy in bits for this ORB's probability_field.
    pub entropy_bits: f32,
    /// Maximum possible entropy for this ORB's field (log2(categories)).
    pub max_entropy_bits: f32,
    /// Unix ms of this sample.
    pub sampled_at_ms: u64,
}

impl OrbEntropySample {
    /// Normalized entropy in [0, 1]. Non-finite malformed inputs are treated as
    /// zero uncertainty so they do not poison moving averages.
    pub fn normalized(&self) -> f32 {
        if !self.entropy_bits.is_finite()
            || !self.max_entropy_bits.is_finite()
            || self.max_entropy_bits < 1e-8
        {
            return 0.0;
        }
        (self.entropy_bits / self.max_entropy_bits).clamp(0.0, 1.0)
    }

    /// True if this ORB should be flagged for belief revision.
    pub fn is_high(&self) -> bool {
        self.normalized() > ORB_ENTROPY_WARNING_FRACTION
    }
}

// -- Field-Level Snapshot -----------------------------------------------------

/// Mean normalized ORB entropy across active ORBs at a point in time.
///
/// This is not graph/semantic field entropy. It intentionally represents:
/// mean(normalized ORB Shannon entropy).
#[derive(Debug, Clone)]
pub struct FieldEntropySnapshot {
    pub mean_normalized_entropy: f32,
    pub active_orb_count: u64,
    pub sampled_at_ms: u64,
}

impl FieldEntropySnapshot {
    pub fn new(mean_normalized_entropy: f32, active_orb_count: u64, sampled_at_ms: u64) -> Self {
        Self {
            mean_normalized_entropy,
            active_orb_count,
            sampled_at_ms,
        }
    }

    fn sanitized(mut self) -> Self {
        self.mean_normalized_entropy = sanitize_normalized(self.mean_normalized_entropy);
        self
    }
}

// -- Entropy Monitor ----------------------------------------------------------

/// Stateful entropy monitor. Maintains a rolling window of mean ORB entropy
/// snapshots for rate-of-change and sustained-threshold detection.
pub struct EntropyMonitor {
    /// Rolling history of field snapshots (bounded window).
    history: VecDeque<FieldEntropySnapshot>,
    max_history: usize,
    /// When mean ORB entropy first exceeded CRITICAL threshold.
    critical_since_ms: Option<u64>,
    /// Max absolute dH/dt before spike/collapse alert fires.
    spike_rate_threshold: f32,
    min_derivative_interval_ms: u64,
    max_sustain_sample_gap_ms: u64,
    ema_alpha: f32,
    smoothed_mean_normalized_entropy: Option<f32>,
    /// ORB creation throughput factor in [0, 1].
    pub creation_throttle_factor: f32,
    /// Backward-compatible hard pause flag. New callers should prefer
    /// creation_throttle_factor.
    pub orb_creation_paused: bool,
}

impl EntropyMonitor {
    pub fn new() -> Self {
        Self {
            history: VecDeque::new(),
            max_history: 128,
            critical_since_ms: None,
            spike_rate_threshold: DEFAULT_ENTROPY_SPIKE_RATE,
            min_derivative_interval_ms: MIN_ENTROPY_DERIVATIVE_INTERVAL_MS,
            max_sustain_sample_gap_ms: FIELD_ENTROPY_MAX_SUSTAIN_SAMPLE_GAP_MS,
            ema_alpha: DEFAULT_ENTROPY_EMA_ALPHA,
            smoothed_mean_normalized_entropy: None,
            creation_throttle_factor: CREATION_THROTTLE_NORMAL,
            orb_creation_paused: false,
        }
    }

    pub fn with_spike_threshold(mut self, threshold: f32) -> Self {
        self.spike_rate_threshold = sanitize_non_negative(threshold, DEFAULT_ENTROPY_SPIKE_RATE);
        self
    }

    pub fn with_ema_alpha(mut self, alpha: f32) -> Self {
        self.ema_alpha = sanitize_normalized(alpha);
        self
    }

    pub fn with_min_derivative_interval_ms(mut self, interval_ms: u64) -> Self {
        self.min_derivative_interval_ms = interval_ms.max(1);
        self
    }

    pub fn with_max_sustain_sample_gap_ms(mut self, gap_ms: u64) -> Self {
        self.max_sustain_sample_gap_ms = gap_ms.max(1);
        self
    }

    /// Ingest a new mean-ORB-entropy snapshot. Returns any triggered alerts.
    pub fn ingest_snapshot(&mut self, snapshot: FieldEntropySnapshot) -> Vec<EntropyAlert> {
        let snapshot = snapshot.sanitized();
        let mut alerts = Vec::new();
        let now = snapshot.sampled_at_ms;
        let h = snapshot.mean_normalized_entropy;
        let previous = self.history.back();
        let mut throttle = if h > MEAN_ORB_ENTROPY_WARNING {
            CREATION_THROTTLE_WARNING
        } else {
            CREATION_THROTTLE_NORMAL
        };

        if h > MEAN_ORB_ENTROPY_WARNING {
            alerts.push(EntropyAlert {
                severity: AlertSeverity::Warning,
                kind: EntropyAlertKind::MeanOrbEntropyHigh {
                    mean_normalized_entropy: h,
                    threshold: MEAN_ORB_ENTROPY_WARNING,
                },
                timestamp_ms: now,
            });
        }

        let previous_smoothed = self.smoothed_mean_normalized_entropy;
        let smoothed_h = previous_smoothed
            .map(|prev| self.ema_alpha * h + (1.0 - self.ema_alpha) * prev)
            .unwrap_or(h);

        if let (Some(prev_snapshot), Some(prev_smoothed_h)) = (previous, previous_smoothed) {
            let dt_ms = now.saturating_sub(prev_snapshot.sampled_at_ms);
            if dt_ms >= self.min_derivative_interval_ms {
                let dh_dt = (smoothed_h - prev_smoothed_h) / dt_ms as f32 * 1000.0;
                if dh_dt > self.spike_rate_threshold {
                    throttle = throttle.min(CREATION_THROTTLE_SEVERE);
                    alerts.push(EntropyAlert {
                        severity: AlertSeverity::Critical,
                        kind: EntropyAlertKind::EntropySpike {
                            rate: dh_dt,
                            threshold: self.spike_rate_threshold,
                        },
                        timestamp_ms: now,
                    });
                } else if dh_dt < -self.spike_rate_threshold {
                    throttle = throttle.min(CREATION_THROTTLE_SEVERE);
                    alerts.push(EntropyAlert {
                        severity: AlertSeverity::Critical,
                        kind: EntropyAlertKind::EntropyCollapse {
                            rate: dh_dt,
                            threshold: self.spike_rate_threshold,
                        },
                        timestamp_ms: now,
                    });
                }
            }
        }

        let sample_gap_too_large = previous
            .map(|prev| now.saturating_sub(prev.sampled_at_ms) > self.max_sustain_sample_gap_ms)
            .unwrap_or(false);

        if h > MEAN_ORB_ENTROPY_CRITICAL {
            if sample_gap_too_large {
                self.critical_since_ms = Some(now);
            }
            let critical_since = *self.critical_since_ms.get_or_insert(now);
            let sustained_ms = now.saturating_sub(critical_since);
            if sustained_ms >= FIELD_ENTROPY_CRITICAL_SUSTAIN_MS {
                throttle = CREATION_THROTTLE_PAUSED;
                alerts.push(EntropyAlert {
                    severity: AlertSeverity::Critical,
                    kind: EntropyAlertKind::SustainedCriticalEntropy {
                        mean_normalized_entropy: h,
                        sustained_ms,
                    },
                    timestamp_ms: now,
                });
            }
        } else {
            self.critical_since_ms = None;
        }

        self.smoothed_mean_normalized_entropy = Some(smoothed_h);
        self.set_creation_throttle(throttle);
        self.push_history(snapshot);

        alerts
    }

    /// Evaluate per-ORB samples. Returns alerts for high-entropy ORBs.
    pub fn evaluate_orbs(&self, samples: &[OrbEntropySample]) -> Vec<EntropyAlert> {
        samples
            .iter()
            .filter(|s| s.is_high())
            .map(|s| EntropyAlert {
                severity: AlertSeverity::Warning,
                kind: EntropyAlertKind::OrbEntropyHigh {
                    orb_id: s.orb_id,
                    normalized_entropy: s.normalized(),
                    threshold: ORB_ENTROPY_WARNING_FRACTION,
                },
                timestamp_ms: s.sampled_at_ms,
            })
            .collect()
    }

    /// Current mean normalized ORB entropy from the most recent snapshot.
    pub fn current_mean_orb_entropy(&self) -> Option<f32> {
        self.history.back().map(|s| s.mean_normalized_entropy)
    }

    /// Backward-compatible name for current_mean_orb_entropy.
    pub fn current_field_entropy(&self) -> Option<f32> {
        self.current_mean_orb_entropy()
    }

    /// Returns the simple moving average over the requested window.
    pub fn smoothed_entropy(&self, window: usize) -> Option<f32> {
        if self.history.is_empty() || window == 0 {
            return None;
        }

        let mut sum = 0.0;
        let mut count = 0usize;
        for snapshot in self.history.iter().rev().take(window) {
            sum += snapshot.mean_normalized_entropy;
            count += 1;
        }

        (count > 0).then_some(sum / count as f32)
    }

    /// Identify ORB clusters that are candidates for consolidation.
    ///
    /// This performs true single-linkage over high-entropy ORBs by returning
    /// connected components in the similarity graph. The merge pipeline should
    /// still enforce semantic coherence before executing destructive merges.
    pub fn consolidation_candidates(
        &self,
        high_entropy_orbs: &[OrbEntropySample],
        similarity_threshold: f32,
        similarity_fn: impl Fn(uuid::Uuid, uuid::Uuid) -> f32,
    ) -> Vec<Vec<uuid::Uuid>> {
        let candidates: Vec<uuid::Uuid> = high_entropy_orbs
            .iter()
            .filter(|s| s.is_high())
            .map(|s| s.orb_id)
            .collect();

        self.connected_similarity_components(&candidates, similarity_threshold, &similarity_fn)
    }

    /// Single-linkage candidates filtered by cluster coherence.
    ///
    /// Use this when chain collapse is a concern. A connected component is kept
    /// only when its average pairwise similarity and weakest pairwise link both
    /// satisfy the provided bounds.
    pub fn consolidation_candidates_with_coherence(
        &self,
        high_entropy_orbs: &[OrbEntropySample],
        edge_similarity_threshold: f32,
        min_average_similarity: f32,
        min_pairwise_similarity: f32,
        similarity_fn: impl Fn(uuid::Uuid, uuid::Uuid) -> f32,
    ) -> Vec<Vec<uuid::Uuid>> {
        let candidates: Vec<uuid::Uuid> = high_entropy_orbs
            .iter()
            .filter(|s| s.is_high())
            .map(|s| s.orb_id)
            .collect();

        self.connected_similarity_components(&candidates, edge_similarity_threshold, &similarity_fn)
            .into_iter()
            .filter(|cluster| {
                cluster_coherence(
                    cluster,
                    sanitize_normalized(min_average_similarity),
                    sanitize_normalized(min_pairwise_similarity),
                    &similarity_fn,
                )
            })
            .collect()
    }

    fn connected_similarity_components(
        &self,
        candidates: &[uuid::Uuid],
        similarity_threshold: f32,
        similarity_fn: &impl Fn(uuid::Uuid, uuid::Uuid) -> f32,
    ) -> Vec<Vec<uuid::Uuid>> {
        let mut clusters = Vec::new();
        let mut visited: HashSet<uuid::Uuid> = HashSet::new();

        for &start in candidates {
            if !visited.insert(start) {
                continue;
            }

            let mut cluster = Vec::new();
            let mut frontier = vec![start];
            while let Some(id) = frontier.pop() {
                cluster.push(id);
                for &other in candidates {
                    if visited.contains(&other) || id == other {
                        continue;
                    }
                    if similarity_fn(id, other) > similarity_threshold {
                        visited.insert(other);
                        frontier.push(other);
                    }
                }
            }

            if cluster.len() >= 2 {
                clusters.push(cluster);
            }
        }

        clusters
    }

    fn set_creation_throttle(&mut self, factor: f32) {
        self.creation_throttle_factor = sanitize_normalized(factor);
        self.orb_creation_paused = self.creation_throttle_factor <= CREATION_THROTTLE_PAUSED;
    }

    fn push_history(&mut self, snapshot: FieldEntropySnapshot) {
        self.history.push_back(snapshot);
        if self.history.len() > self.max_history {
            self.history.pop_front();
        }
    }
}

impl Default for EntropyMonitor {
    fn default() -> Self {
        Self::new()
    }
}

fn sanitize_normalized(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn sanitize_non_negative(value: f32, fallback: f32) -> f32 {
    if value.is_finite() && value >= 0.0 {
        value
    } else {
        fallback
    }
}

fn cluster_coherence(
    cluster: &[uuid::Uuid],
    min_average_similarity: f32,
    min_pairwise_similarity: f32,
    similarity_fn: &impl Fn(uuid::Uuid, uuid::Uuid) -> f32,
) -> bool {
    if cluster.len() < 2 {
        return false;
    }

    let mut sum = 0.0;
    let mut count = 0usize;
    let mut weakest = f32::INFINITY;

    for i in 0..cluster.len() {
        for j in (i + 1)..cluster.len() {
            let similarity = sanitize_normalized(similarity_fn(cluster[i], cluster[j]));
            sum += similarity;
            count += 1;
            weakest = weakest.min(similarity);
        }
    }

    count > 0
        && sum / count as f32 >= min_average_similarity
        && weakest >= min_pairwise_similarity
}

// -- Tests --------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn snapshot(h: f32, ts: u64, n: u64) -> FieldEntropySnapshot {
        FieldEntropySnapshot::new(h, n, ts)
    }

    fn orb_sample(entropy: f32, max_entropy: f32) -> OrbEntropySample {
        OrbEntropySample {
            orb_id: Uuid::now_v7(),
            entropy_bits: entropy,
            max_entropy_bits: max_entropy,
            sampled_at_ms: 0,
        }
    }

    #[test]
    fn no_alert_below_thresholds() {
        let mut monitor = EntropyMonitor::new();
        let alerts = monitor.ingest_snapshot(snapshot(0.3, 1_000, 100));
        assert!(alerts.is_empty());
        assert_eq!(monitor.creation_throttle_factor, CREATION_THROTTLE_NORMAL);
    }

    #[test]
    fn warning_fires_above_mean_orb_threshold() {
        let mut monitor = EntropyMonitor::new();
        let alerts =
            monitor.ingest_snapshot(snapshot(MEAN_ORB_ENTROPY_WARNING + 0.01, 1_000, 100));
        assert!(alerts.iter().any(|a| matches!(
            a.kind,
            EntropyAlertKind::MeanOrbEntropyHigh { .. }
        )));
        assert_eq!(monitor.creation_throttle_factor, CREATION_THROTTLE_WARNING);
    }

    #[test]
    fn spike_alert_fires_on_rapid_increase() {
        let mut monitor = EntropyMonitor::new().with_spike_threshold(0.001);
        monitor.ingest_snapshot(snapshot(0.1, 1_000, 100));

        let alerts = monitor.ingest_snapshot(snapshot(0.5, 2_000, 100));
        assert!(alerts
            .iter()
            .any(|a| matches!(a.kind, EntropyAlertKind::EntropySpike { .. })));
        assert_eq!(monitor.creation_throttle_factor, CREATION_THROTTLE_SEVERE);
        assert!(!monitor.orb_creation_paused);
    }

    #[test]
    fn collapse_alert_fires_on_rapid_decrease() {
        let mut monitor = EntropyMonitor::new().with_spike_threshold(0.001);
        monitor.ingest_snapshot(snapshot(0.9, 1_000, 100));

        let alerts = monitor.ingest_snapshot(snapshot(0.1, 2_000, 100));
        assert!(alerts
            .iter()
            .any(|a| matches!(a.kind, EntropyAlertKind::EntropyCollapse { .. })));
        assert_eq!(monitor.creation_throttle_factor, CREATION_THROTTLE_SEVERE);
    }

    #[test]
    fn derivative_ignores_tiny_sampling_interval() {
        let mut monitor = EntropyMonitor::new().with_spike_threshold(0.001);
        monitor.ingest_snapshot(snapshot(0.1, 1_000, 100));

        let alerts = monitor.ingest_snapshot(snapshot(0.5, 1_001, 100));
        assert!(!alerts
            .iter()
            .any(|a| matches!(a.kind, EntropyAlertKind::EntropySpike { .. })));
    }

    #[test]
    fn creation_throttle_recovers_when_entropy_drops() {
        let mut monitor = EntropyMonitor::new().with_spike_threshold(0.001);
        monitor.ingest_snapshot(snapshot(0.1, 1_000, 100));
        monitor.ingest_snapshot(snapshot(0.9, 2_000, 100));
        assert_eq!(monitor.creation_throttle_factor, CREATION_THROTTLE_SEVERE);

        monitor.ingest_snapshot(snapshot(0.2, 100_000, 100));
        assert_eq!(monitor.creation_throttle_factor, CREATION_THROTTLE_NORMAL);
    }

    #[test]
    fn sustained_critical_entropy_pauses_creation() {
        let mut monitor = EntropyMonitor::new();
        let start = 1_000;
        monitor.ingest_snapshot(snapshot(0.9, start, 100));
        let mut alerts = Vec::new();

        for offset in (FIELD_ENTROPY_MAX_SUSTAIN_SAMPLE_GAP_MS
            ..=FIELD_ENTROPY_CRITICAL_SUSTAIN_MS)
            .step_by(FIELD_ENTROPY_MAX_SUSTAIN_SAMPLE_GAP_MS as usize)
        {
            alerts = monitor.ingest_snapshot(snapshot(0.9, start + offset, 100));
        }

        assert!(alerts.iter().any(|a| matches!(
            a.kind,
            EntropyAlertKind::SustainedCriticalEntropy { .. }
        )));
        assert_eq!(monitor.creation_throttle_factor, CREATION_THROTTLE_PAUSED);
        assert!(monitor.orb_creation_paused);
    }

    #[test]
    fn sustained_critical_requires_non_sparse_sampling() {
        let mut monitor =
            EntropyMonitor::new().with_max_sustain_sample_gap_ms(FIELD_ENTROPY_CRITICAL_SUSTAIN_MS);
        monitor.ingest_snapshot(snapshot(0.9, 1_000, 100));

        let alerts = monitor.ingest_snapshot(snapshot(
            0.9,
            1_000 + FIELD_ENTROPY_CRITICAL_SUSTAIN_MS + 1,
            100,
        ));

        assert!(!alerts.iter().any(|a| matches!(
            a.kind,
            EntropyAlertKind::SustainedCriticalEntropy { .. }
        )));
    }

    #[test]
    fn malformed_entropy_is_sanitized() {
        let mut monitor = EntropyMonitor::new();
        monitor.ingest_snapshot(snapshot(f32::NAN, 1_000, 100));
        assert_eq!(monitor.current_mean_orb_entropy(), Some(0.0));

        monitor.ingest_snapshot(snapshot(2.0, 2_000, 100));
        assert_eq!(monitor.current_mean_orb_entropy(), Some(1.0));
    }

    #[test]
    fn orb_entropy_alert_for_high_entropy_orb() {
        let monitor = EntropyMonitor::new();
        let samples = vec![orb_sample(1.0, 1.0)];
        let alerts = monitor.evaluate_orbs(&samples);
        assert_eq!(alerts.len(), 1);
    }

    #[test]
    fn orb_below_threshold_no_alert() {
        let monitor = EntropyMonitor::new();
        let samples = vec![orb_sample(0.5, 2.0)];
        let alerts = monitor.evaluate_orbs(&samples);
        assert!(alerts.is_empty());
    }

    #[test]
    fn consolidation_candidates_groups_similar_orbs() {
        let monitor = EntropyMonitor::new();
        let id1 = Uuid::now_v7();
        let id2 = Uuid::now_v7();
        let id3 = Uuid::now_v7();

        let samples = vec![
            OrbEntropySample {
                orb_id: id1,
                entropy_bits: 1.9,
                max_entropy_bits: 2.0,
                sampled_at_ms: 0,
            },
            OrbEntropySample {
                orb_id: id2,
                entropy_bits: 1.9,
                max_entropy_bits: 2.0,
                sampled_at_ms: 0,
            },
            OrbEntropySample {
                orb_id: id3,
                entropy_bits: 1.9,
                max_entropy_bits: 2.0,
                sampled_at_ms: 0,
            },
        ];

        let clusters = monitor.consolidation_candidates(&samples, 0.9, |a, b| {
            if (a == id1 && b == id2) || (a == id2 && b == id1) {
                0.95
            } else {
                0.1
            }
        });

        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].len(), 2);
    }

    #[test]
    fn consolidation_candidates_are_transitive_single_linkage() {
        let monitor = EntropyMonitor::new();
        let id1 = Uuid::now_v7();
        let id2 = Uuid::now_v7();
        let id3 = Uuid::now_v7();

        let samples = vec![
            OrbEntropySample {
                orb_id: id1,
                entropy_bits: 1.9,
                max_entropy_bits: 2.0,
                sampled_at_ms: 0,
            },
            OrbEntropySample {
                orb_id: id2,
                entropy_bits: 1.9,
                max_entropy_bits: 2.0,
                sampled_at_ms: 0,
            },
            OrbEntropySample {
                orb_id: id3,
                entropy_bits: 1.9,
                max_entropy_bits: 2.0,
                sampled_at_ms: 0,
            },
        ];

        let clusters = monitor.consolidation_candidates(&samples, 0.9, |a, b| {
            if (a == id1 && b == id2)
                || (a == id2 && b == id1)
                || (a == id2 && b == id3)
                || (a == id3 && b == id2)
            {
                0.95
            } else {
                0.1
            }
        });

        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].len(), 3);
    }

    #[test]
    fn coherence_filter_rejects_weak_chain_clusters() {
        let monitor = EntropyMonitor::new();
        let id1 = Uuid::now_v7();
        let id2 = Uuid::now_v7();
        let id3 = Uuid::now_v7();

        let samples = vec![
            OrbEntropySample {
                orb_id: id1,
                entropy_bits: 1.9,
                max_entropy_bits: 2.0,
                sampled_at_ms: 0,
            },
            OrbEntropySample {
                orb_id: id2,
                entropy_bits: 1.9,
                max_entropy_bits: 2.0,
                sampled_at_ms: 0,
            },
            OrbEntropySample {
                orb_id: id3,
                entropy_bits: 1.9,
                max_entropy_bits: 2.0,
                sampled_at_ms: 0,
            },
        ];

        let similarity = |a, b| {
            if (a == id1 && b == id2)
                || (a == id2 && b == id1)
                || (a == id2 && b == id3)
                || (a == id3 && b == id2)
            {
                0.95
            } else {
                0.1
            }
        };

        let clusters =
            monitor.consolidation_candidates_with_coherence(&samples, 0.9, 0.8, 0.8, similarity);

        assert!(clusters.is_empty());
    }
}
