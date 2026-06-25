//! Learned per-ORF scoring model used by `--model`.
//!
//! The model is a JSON logistic-regression classifier that consumes the same
//! `OrfFeatures` vector used by `--export-features`. It predicts the
//! probability that an ORF is a real gene, and the probability is converted to
//! a negative log-odds edge weight for the shortest-path search.

use crate::ml_features::OrfFeatures;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const NUM_ORF_SCORE_FEATURES: usize = crate::ml_features::NUM_FEATURES;

/// Logistic-regression model for ORF scoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrfScoreModel {
    pub version: u32,
    pub num_features: usize,
    pub coeffs: [f64; NUM_ORF_SCORE_FEATURES],
    pub mean: [f64; NUM_ORF_SCORE_FEATURES],
    pub std: [f64; NUM_ORF_SCORE_FEATURES],
}

impl OrfScoreModel {
    /// Load a model from a JSON file produced by `scripts/train_orf_score_model.py`.
    pub fn from_json(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read ORF score model: {:?}", path))?;
        let model: Self = serde_json::from_str(&text)
            .with_context(|| format!("Failed to parse ORF score model JSON: {:?}", path))?;
        anyhow::ensure!(
            model.num_features == NUM_ORF_SCORE_FEATURES,
            "Model expected {} features but code expects {}",
            model.num_features,
            NUM_ORF_SCORE_FEATURES
        );
        Ok(model)
    }

    /// Predict the probability that the ORF is a real gene and convert it to a
    /// graph edge weight. The weight is the negative log-odds, clamped to
    /// [-10, 10] to keep the shortest-path search numerically stable.
    pub fn score(&self, features: &OrfFeatures) -> f64 {
        let mut z = 0.0;
        for i in 0..NUM_ORF_SCORE_FEATURES {
            let denom = if self.std[i] == 0.0 { 1.0 } else { self.std[i] };
            let v = (features.0[i] as f64 - self.mean[i]) / denom;
            z += v * self.coeffs[i];
        }
        let p = 1.0 / (1.0 + (-z).exp());
        let logit = (p / (1.0 - p)).ln();
        (-logit).clamp(-10.0, 10.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml_features::NUM_FEATURES;

    fn zero_model() -> OrfScoreModel {
        OrfScoreModel {
            version: 1,
            num_features: NUM_FEATURES,
            coeffs: [0.0; NUM_FEATURES],
            mean: [0.0; NUM_FEATURES],
            std: [1.0; NUM_FEATURES],
        }
    }

    #[test]
    fn zero_model_returns_neutral_weight() {
        let model = zero_model();
        let features = OrfFeatures([0.0f32; NUM_FEATURES]);
        let w = model.score(&features);
        assert!(w.abs() < 1e-6, "zero-coeff model should give weight ~0");
    }

    #[test]
    fn positive_coeff_increases_weight() {
        let mut model = zero_model();
        model.coeffs[0] = 2.0;
        let features = OrfFeatures([1.0f32; NUM_FEATURES]);
        let w = model.score(&features);
        assert!(w < 0.0, "higher probability should give negative weight");
    }

    #[test]
    fn weight_is_clamped() {
        let mut model = zero_model();
        model.coeffs[0] = 100.0;
        let features = OrfFeatures([10.0f32; NUM_FEATURES]);
        let w = model.score(&features);
        assert!(w >= -10.0 && w <= 10.0);
    }
}
