//! ONNX-based ORF scorer used by `--model`.
//!
//! Loads an ONNX model that consumes the same [`OrfFeatures`] vector used by
//! `--export-features` and returns a negative log-odds graph edge weight.

use crate::ml_features::{OrfFeatures, NUM_FEATURES};
use anyhow::{Context, Result};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::fmt;
use std::path::Path;
use std::sync::Mutex;

/// Thread-safe wrapper around an ONNX inference session.
pub struct OnnxScorer {
    session: Mutex<Session>,
}

impl fmt::Debug for OnnxScorer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OnnxScorer").finish_non_exhaustive()
    }
}

impl OnnxScorer {
    /// Load an ONNX model from disk.
    pub fn from_file(path: &Path) -> Result<Self> {
        let session = Session::builder()
            .with_context(|| "Failed to create ONNX session builder")?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .commit_from_file(path)
            .with_context(|| format!("Failed to load ONNX model from {:?}", path))?;
        Ok(Self {
            session: Mutex::new(session),
        })
    }

    /// Run inference and return the positive-class probability.
    pub fn probability(&self, features: &OrfFeatures) -> f64 {
        let input_values: Vec<f32> = features.0.to_vec();
        let tensor = match Tensor::from_array(([1usize, NUM_FEATURES], input_values)) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Failed to create ONNX input tensor: {e}");
                return 0.5;
            }
        };

        let mut session = self
            .session
            .lock()
            .expect("ONNX session mutex was poisoned");

        let outputs = match session.run(ort::inputs![tensor]) {
            Ok(out) => out,
            Err(e) => {
                eprintln!("ONNX inference failed: {e}");
                return 0.5;
            }
        };

        let mut p: Option<f64> = None;
        for (_, output) in outputs.iter() {
            if let Ok((shape, data)) = output.try_extract_tensor::<f32>() {
                p = Some(if shape.len() == 2 && shape[1] == 2 {
                    // Two-class probability vector: take the positive class.
                    data[1] as f64
                } else {
                    // Single-value output (probability or logit).
                    data.first().copied().unwrap_or(0.5f32) as f64
                });
                break;
            }
        }

        match p {
            Some(v) => v.clamp(1e-6, 1.0 - 1e-6),
            None => {
                eprintln!("Failed to extract ONNX output tensor: no f32 output found");
                0.5
            }
        }
    }

    /// Convenience helper: compute the positive-class probability for an ORF.
    pub fn probability_for_orf(&self, orf: &crate::orf::Orf) -> f64 {
        let features = orf.extract_features();
        self.probability(&features)
    }

    /// Run inference and convert the model output to a graph edge weight.
    ///
    /// The ONNX model is expected to accept a `[1, NUM_FEATURES]` `f32` tensor
    /// (a single ORF feature vector) and emit either a single `f32` probability
    /// or a `[1, 2]` probability vector. The positive-class probability is
    /// converted to a negative log-odds weight and multiplied by `scale`.
    ///
    /// `threshold` is the probability cutoff that separates "reward" (negative
    /// weight) from "penalty" (positive weight). An ORF whose predicted
    /// probability equals `threshold` gets weight 0. The default 0.5 corresponds
    /// to the natural logit decision boundary.
    pub fn score(&self, features: &OrfFeatures, scale: f64, threshold: f64) -> f64 {
        let p = self.probability(features);
        let logit = (p / (1.0 - p)).ln();
        let t = threshold.clamp(1e-6, 1.0 - 1e-6);
        let offset = (t / (1.0 - t)).ln();
        -scale * (logit - offset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_onnx_file_returns_error() {
        let result = OnnxScorer::from_file(Path::new("/nonexistent/model.onnx"));
        assert!(
            result.is_err(),
            "loading a non-existent ONNX file should fail"
        );
    }

    #[test]
    fn feature_array_has_expected_shape() {
        let features = OrfFeatures([0.0f32; NUM_FEATURES]);
        let arr = ndarray::Array2::from_shape_vec(
            (1, NUM_FEATURES),
            features.0.iter().copied().collect(),
        );
        assert!(arr.is_ok());
        assert_eq!(arr.unwrap().shape(), &[1, NUM_FEATURES]);
    }
}
