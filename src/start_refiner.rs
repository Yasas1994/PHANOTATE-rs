//! Learned start-site scoring model.

use crate::orf::Orf;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const NUM_START_FEATURES: usize = 11;

/// Fixed feature vector for a candidate start site.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct StartSiteFeatures(pub [f64; NUM_START_FEATURES]);

impl StartSiteFeatures {
    /// Build features from an ORF. This must stay in sync with the Python
    /// training script that produces the JSON model consumed by `StartModel`.
    pub fn new(orf: &Orf) -> Self {
        let mut f = [0.0; NUM_START_FEATURES];
        let codon = orf.start_codon();
        if codon.eq_ignore_ascii_case(b"atg") {
            f[0] = 1.0;
        } else if codon.eq_ignore_ascii_case(b"gtg") {
            f[1] = 1.0;
        } else if codon.eq_ignore_ascii_case(b"ttg") {
            f[2] = 1.0;
        } else {
            f[3] = 1.0;
        }
        f[4] = (orf.rbs_score as f64 / crate::rbs_scanner::NUM_RBS_BINS as f64).clamp(0.0, 1.0);
        f[5] = orf.non_sd_rbs_score.clamp(0.0, 10.0);
        f[6] = (orf.seq.len() as f64).max(1.0).ln();
        f[7] = orf.coding_potential.clamp(0.001, 1000.0);
        match orf.frame.abs() {
            1 => f[8] = 1.0,
            2 => f[9] = 1.0,
            3 => f[10] = 1.0,
            _ => {}
        }
        Self(f)
    }
}

/// Linear start-site model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartModel {
    pub coeffs: [f64; NUM_START_FEATURES],
    pub mean: [f64; NUM_START_FEATURES],
    pub std: [f64; NUM_START_FEATURES],
}

impl StartModel {
    /// Load a model from a JSON file produced by `train_start_model.py`.
    pub fn from_json(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read start model: {:?}", path))?;
        let model: Self = serde_json::from_str(&text)
            .with_context(|| format!("Failed to parse start model JSON: {:?}", path))?;
        Ok(model)
    }

    /// Score a candidate start. Returns a multiplier in [0.5, 2.0].
    pub fn score(&self, features: &StartSiteFeatures) -> f64 {
        let mut z = 0.0;
        for i in 0..NUM_START_FEATURES {
            let denom = if self.std[i] == 0.0 { 1.0 } else { self.std[i] };
            let v = (features.0[i] - self.mean[i]) / denom;
            z += v * self.coeffs[i];
        }
        let p = 1.0 / (1.0 + (-z).exp());
        (p * 1.5 + 0.5).clamp(0.5, 2.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_vector_length_matches_constant() {
        let f = StartSiteFeatures::new(&Orf {
            start: 1,
            stop: 100,
            frame: 1,
            seq: vec![b'a'; 100],
            rbs_score: 5,
            rbs_motif: None,
            pstop: 0.01,
            sd_rbs_score: 1.5,
            non_sd_rbs_score: 1.2,
            hold: 10.0,
            coding_potential: 0.1,
            start_score: 1.0,
            weight: 1.0,
        });
        assert_eq!(f.0.len(), NUM_START_FEATURES);
    }

    #[test]
    fn model_score_is_within_bounds() {
        let model = StartModel {
            coeffs: [0.1; NUM_START_FEATURES],
            mean: [0.0; NUM_START_FEATURES],
            std: [1.0; NUM_START_FEATURES],
        };
        let f = StartSiteFeatures([0.0; NUM_START_FEATURES]);
        let s = model.score(&f);
        assert!(s >= 0.5);
        assert!(s <= 2.0);
    }
}
