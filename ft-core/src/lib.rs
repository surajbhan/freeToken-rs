//! ft-core: Rust port of FreeToken's MoE offload-engine core.
//!
//! Ported from FlashML-org/FreeToken (Apache-2.0), `python/freetoken/moe/`:
//! - `SlotCache`: global LRU cache mapping (layer, expert) -> GPU slot
//!   (reference implementation of flashlib's slot_cache semantics).
//! - `qstar`: the bandwidth-adaptive hybrid split — fetch a pcie_bw/cpu_bw
//!   fraction of each step's misses over PCIe, CPU computes the rest, so both
//!   finish together (paper's q* policy; engine.py:644).
//! - `plan_runs`: coalesce missing experts into contiguous host-bank runs so
//!   each decode step issues one batched copy of few large entries
//!   (offload_cache.py copy_missing / fused copy plan).

pub mod q4_0;

use std::collections::HashMap;

/// Global key: experts are cached across layers in one pool, matching
/// FreeToken's "global LRU expert caching".
pub type ExpertKey = (u32, u32); // (layer, expert)

/// Result of a per-step cache lookup for one layer's routed experts.
#[derive(Debug, PartialEq, Eq)]
pub struct Lookup {
    /// (expert, slot) pairs already resident on GPU.
    pub hits: Vec<(u32, u32)>,
    /// (expert, slot) pairs that must be filled (slot already assigned,
    /// LRU victim evicted).
    pub misses: Vec<(u32, u32)>,
}

/// LRU slot cache: fixed number of GPU expert slots shared by all layers.
pub struct SlotCache {
    capacity: u32,
    map: HashMap<ExpertKey, u32>,
    /// slot -> key currently occupying it (None = free)
    slots: Vec<Option<ExpertKey>>,
    /// monotone clock for LRU age; slot -> last touch tick
    last_used: Vec<u64>,
    tick: u64,
    pub hits_total: u64,
    pub misses_total: u64,
}

impl SlotCache {
    pub fn new(capacity: u32) -> Self {
        Self {
            capacity,
            map: HashMap::new(),
            slots: vec![None; capacity as usize],
            last_used: vec![0; capacity as usize],
            tick: 0,
            hits_total: 0,
            misses_total: 0,
        }
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Look up one layer's routed experts, assigning slots to misses by
    /// evicting the least-recently-used residents. Experts of the current
    /// step are touched before any eviction, so a step's own hits can never
    /// be evicted by its own misses.
    pub fn lookup(&mut self, layer: u32, experts: &[u32]) -> Lookup {
        self.tick += 1;
        let mut hits = Vec::new();
        let mut missing = Vec::new();
        for &e in experts {
            match self.map.get(&(layer, e)) {
                Some(&slot) => {
                    self.last_used[slot as usize] = self.tick;
                    hits.push((e, slot));
                }
                None => missing.push(e),
            }
        }
        let mut misses = Vec::new();
        for e in missing {
            let slot = self.take_lru_slot();
            if let Some(old) = self.slots[slot as usize].take() {
                self.map.remove(&old);
            }
            self.slots[slot as usize] = Some((layer, e));
            self.last_used[slot as usize] = self.tick;
            self.map.insert((layer, e), slot);
            misses.push((e, slot));
        }
        self.hits_total += hits.len() as u64;
        self.misses_total += misses.len() as u64;
        Lookup { hits, misses }
    }

    fn take_lru_slot(&mut self) -> u32 {
        // Free slot first, else the oldest touch.
        let mut best = 0usize;
        let mut best_age = u64::MAX;
        for (i, occ) in self.slots.iter().enumerate() {
            if occ.is_none() {
                return i as u32;
            }
            if self.last_used[i] < best_age {
                best_age = self.last_used[i];
                best = i;
            }
        }
        best as u32
    }

    /// Un-admit an entry whose slot was assigned by `lookup` but whose
    /// weights were never actually copied in (e.g. a miss served by the CPU
    /// path). Leaving it mapped would make the next route a false hit that
    /// reads whatever the slot last held.
    pub fn forget(&mut self, layer: u32, expert: u32) {
        if let Some(slot) = self.map.remove(&(layer, expert)) {
            self.slots[slot as usize] = None;
            self.last_used[slot as usize] = 0;
        }
    }

    pub fn hit_rate(&self) -> f64 {
        let total = self.hits_total + self.misses_total;
        if total == 0 {
            0.0
        } else {
            self.hits_total as f64 / total as f64
        }
    }
}

/// q* policy: fraction of each step's misses to fetch over PCIe so the PCIe
/// fetch and the CPU GEMV over the remainder take equal time.
/// fetched : cpu = pcie_bw : (cpu_bw - pcie_bw)  =>  fraction = pcie/cpu.
pub fn qstar_fraction(pcie_bw_gbs: f64, cpu_bw_gbs: f64) -> f64 {
    if cpu_bw_gbs <= 0.0 || pcie_bw_gbs <= 0.0 {
        return 1.0; // no CPU path measured: pure offload
    }
    (pcie_bw_gbs / cpu_bw_gbs).clamp(0.0, 1.0)
}

/// Backend recommendation, ported from benchbw.py::recommend — hybrid pays off
/// once the CPU can chew through experts materially faster than PCIe can ship
/// them.
pub fn recommend(cpu_bw_gbs: f64, pcie_bw_gbs: f64) -> &'static str {
    const THRESHOLD: f64 = 2.0;
    if cpu_bw_gbs > THRESHOLD * pcie_bw_gbs {
        "hybrid"
    } else {
        "offload"
    }
}

/// Split one step's miss list per q*: `(fetch, cpu)` expert index lists.
/// Rounds the fetch count to the integer that best balances the overlap.
pub fn split_misses(misses: &[u32], fraction: f64) -> (Vec<u32>, Vec<u32>) {
    let n = misses.len() as f64;
    let k = (n * fraction).round() as usize;
    let k = k.min(misses.len());
    (misses[..k].to_vec(), misses[k..].to_vec())
}

/// A contiguous run of experts in a host bank: copy `count` rows starting at
/// expert `start` into GPU slots `slots` (one batched entry per run).
#[derive(Debug, PartialEq, Eq)]
pub struct CopyRun {
    pub start_expert: u32,
    pub count: u32,
    pub slots: Vec<u32>,
}

/// Coalesce (expert, slot) misses of one layer into runs contiguous in the
/// host bank (experts are laid out consecutively per layer), mirroring the
/// coalesced-run planning in offload_cache.py. Fewer, larger H2D entries keep
/// cudaMemcpyBatchAsync at full PCIe rate.
pub fn plan_runs(mut misses: Vec<(u32, u32)>) -> Vec<CopyRun> {
    misses.sort_by_key(|&(e, _)| e);
    let mut runs: Vec<CopyRun> = Vec::new();
    for (e, s) in misses {
        match runs.last_mut() {
            Some(r) if r.start_expert + r.count == e => {
                r.count += 1;
                r.slots.push(s);
            }
            _ => runs.push(CopyRun {
                start_expert: e,
                count: 1,
                slots: vec![s],
            }),
        }
    }
    runs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cold_lookup_is_all_misses_then_all_hits() {
        let mut c = SlotCache::new(8);
        let l = c.lookup(0, &[3, 1, 4]);
        assert_eq!(l.hits.len(), 0);
        assert_eq!(l.misses.len(), 3);
        let l = c.lookup(0, &[3, 1, 4]);
        assert_eq!(l.hits.len(), 3);
        assert_eq!(l.misses.len(), 0);
    }

    #[test]
    fn lru_evicts_oldest_across_layers() {
        let mut c = SlotCache::new(2);
        c.lookup(0, &[10]); // slot A
        c.lookup(1, &[20]); // slot B
        c.lookup(0, &[10]); // touch A -> B is now LRU
        let l = c.lookup(2, &[30]); // must evict (1,20)
        assert_eq!(l.misses.len(), 1);
        let l = c.lookup(0, &[10]);
        assert_eq!(l.hits.len(), 1, "recently-touched expert survived");
        let l = c.lookup(1, &[20]);
        assert_eq!(l.misses.len(), 1, "LRU expert was evicted");
    }

    #[test]
    fn step_hits_never_evicted_by_same_step_misses() {
        let mut c = SlotCache::new(2);
        c.lookup(0, &[1, 2]);
        // 1 hits, 3 misses; the miss must evict 2 (untouched), never 1.
        let l = c.lookup(0, &[1, 3]);
        assert_eq!(l.hits, vec![(1, 0)]);
        assert_eq!(l.misses.len(), 1);
        assert_eq!(c.lookup(0, &[1]).hits.len(), 1);
    }

    #[test]
    fn qstar_matches_paper_ratio() {
        assert!((qstar_fraction(12.0, 48.0) - 0.25).abs() < 1e-9);
        assert_eq!(qstar_fraction(12.0, 0.0), 1.0);
        assert_eq!(qstar_fraction(50.0, 25.0), 1.0); // PCIe faster than CPU: fetch all
    }

    #[test]
    fn recommend_matches_python() {
        assert_eq!(recommend(48.0, 12.0), "hybrid");
        assert_eq!(recommend(20.0, 12.0), "offload");
    }

    #[test]
    fn split_rounds_to_balance() {
        let m: Vec<u32> = (0..8).collect();
        let (fetch, cpu) = split_misses(&m, 0.25);
        assert_eq!(fetch.len(), 2);
        assert_eq!(cpu.len(), 6);
    }

    #[test]
    fn runs_coalesce_contiguous_experts() {
        let runs = plan_runs(vec![(5, 0), (3, 1), (4, 2), (9, 3)]);
        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0].start_expert, 3);
        assert_eq!(runs[0].count, 3);
        assert_eq!(runs[0].slots, vec![1, 2, 0]);
        assert_eq!(runs[1].start_expert, 9);
    }
}
