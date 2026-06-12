//! ONNX-based ML scorer for hybrid ORF weight adjustment.
//!
//! Loads an ONNX regression model via ONNX Runtime (ort) and applies a
//! multiplicative adjustment to the heuristic ORF score. The adjustment is
//! clamped to prevent extreme values from destabilising the shortest-path
//! search.
//!
//! This implementation uses the `ort` crate (ONNX Runtime) which supports
//! all ONNX operators including TreeEnsembleRegressor from XGBoost and
//! scikit-learn exports.

use crate::ml_features::OrfFeatures;
use anyhow::{Context, Result};
use ndarray::Array2;
use ort::session::Session;
use ort::value::Tensor;
use std::sync::Mutex;

/// Minimum multiplicative adjustment factor.
const MIN_ADJUSTMENT: f64 = 0.5;
/// Maximum multiplicative adjustment factor.
const MAX_ADJUSTMENT: f64 = 2.0;

/// Wrapper around an ONNX Runtime session for ORF score adjustment.
///
/// The session is wrapped in a Mutex because `Session::run()` requires `&mut self`,
/// but we need to call inference from multiple threads (via rayon parallel genome processing).
/// The underlying ONNX Runtime C session is thread-safe, so the Mutex is just an API adapter.
pub struct MlScorer {
    session: Mutex<Session>,
}

impl MlScorer {
    /// Load an ONNX model from disk.
    ///
    /// The model must accept a single input tensor of shape `[N, 13]`
    /// (batch size × features) and produce a single output tensor of
    /// shape `[N, 1]` containing log-scale adjustment values.
    pub fn from_file(path: &str) -> Result<Self> {
        let mut builder = Session::builder()
            .context("Failed to create ONNX session builder")?;
        let session = builder
            .commit_from_file(path)
            .with_context(|| format!("Failed to load ONNX model from {}", path))?;
        Ok(MlScorer {
            session: Mutex::new(session),
        })
    }

    /// Predict a single adjustment factor for one ORF.
    ///
    /// Returns a multiplicative adjustment in the range [0.5, 2.0].
    /// The heuristic weight should be multiplied by this value to obtain
    /// the final ML-adjusted weight.
    pub fn predict_adjustment(&self, features: &OrfFeatures) -> f64 {
        match self.predict_adjustment_inner(features) {
            Ok(adj) => adj,
            Err(e) => {
                eprintln!("Warning: ML inference failed ({}), using neutral adjustment 1.0", e);
                1.0
            }
        }
    }

    fn predict_adjustment_inner(&self, features: &OrfFeatures) -> Result<f64> {
        // Build a [1, NUM_FEATURES] ndarray
        let input_data: Vec<f32> = features.as_slice().to_vec();
        let input_array = Array2::from_shape_vec((1, crate::ml_features::NUM_FEATURES), input_data)
            .context("Failed to create input tensor")?;

        // Create Tensor value from ndarray
        let input_tensor = Tensor::from_array(input_array)
            .context("Failed to convert ndarray to ONNX tensor")?;

        // Run inference (lock the session for mutable access)
        let mut session = self.session.lock().unwrap();
        let outputs = session
            .run(ort::inputs![input_tensor])
            .context("ONNX model inference failed")?;

        // Extract the single output value
        // try_extract_tensor returns (&Shape, &[f32]) - we want the data slice
        let (_shape, output_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .context("Failed to extract output tensor")?;

        let output_val = output_data
            .iter()
            .next()
            .copied()
            .unwrap_or(0.0f32) as f64;

        // Clamp to safe range and convert from log-space
        let clamped = clamp_adjustment(output_val);
        let adjustment = clamped.exp();

        Ok(adjustment)
    }

    /// Predict adjustments for a batch of ORFs.
    ///
    /// More efficient than calling `predict_adjustment` in a loop because
    /// it amortises the tensor setup overhead.
    pub fn predict_adjustments_batch(&self, features: &[OrfFeatures]) -> Vec<f64> {
        if features.is_empty() {
            return Vec::new();
        }

        match self.predict_batch_inner(features) {
            Ok(adjs) => adjs,
            Err(e) => {
                eprintln!(
                    "Warning: batch ML inference failed ({}), using neutral adjustments",
                    e
                );
                vec![1.0; features.len()]
            }
        }
    }

    fn predict_batch_inner(&self, features: &[OrfFeatures]) -> Result<Vec<f64>> {
        let n = features.len();
        let num_features = crate::ml_features::NUM_FEATURES;

        // Flatten all features into one Vec
        let flat: Vec<f32> = features
            .iter()
            .flat_map(|f| f.as_slice().iter().copied())
            .collect();

        let input_array = Array2::from_shape_vec((n, num_features), flat)
            .context("Failed to create batch input tensor")?;

        let input_tensor = Tensor::from_array(input_array)
            .context("Failed to convert batch ndarray to ONNX tensor")?;

        // Run inference (lock the session for mutable access)
        let mut session = self.session.lock().unwrap();
        let outputs = session
            .run(ort::inputs![input_tensor])
            .context("ONNX batch inference failed")?;

        let (_shape, output_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .context("Failed to extract batch output tensor")?;

        let mut adjustments = Vec::with_capacity(n);
        for i in 0..n {
            let val = output_data
                .iter()
                .nth(i)
                .copied()
                .unwrap_or(0.0f32) as f64;
            let clamped = clamp_adjustment(val);
            adjustments.push(clamped.exp());
        }

        Ok(adjustments)
    }
}

/// Clamp a log-scale adjustment value to the safe range.
fn clamp_adjustment(log_val: f64) -> f64 {
    let ln_min = MIN_ADJUSTMENT.ln();
    let ln_max = MAX_ADJUSTMENT.ln();
    log_val.clamp(ln_min, ln_max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adjustment_bounds() {
        // Verify that extreme log values are properly clamped
        assert!((MIN_ADJUSTMENT - 0.5).abs() < 0.001);
        assert!((MAX_ADJUSTMENT - 2.0).abs() < 0.001);

        let ln_min = MIN_ADJUSTMENT.ln();
        let ln_max = MAX_ADJUSTMENT.ln();
        assert!(ln_min < 0.0);
        assert!(ln_max > 0.0);

        // Test clamping
        assert_eq!(clamp_adjustment(-10.0), ln_min);
        assert_eq!(clamp_adjustment(10.0), ln_max);
        assert_eq!(clamp_adjustment(0.0), 0.0);
    }

    #[test]
    fn test_clamped_exp() {
        // exp(clamp(log(0.5))) == 0.5
        assert!((clamp_adjustment(MIN_ADJUSTMENT.ln()).exp() - 0.5).abs() < 0.001);
        // exp(clamp(log(2.0))) == 2.0
        assert!((clamp_adjustment(MAX_ADJUSTMENT.ln()).exp() - 2.0).abs() < 0.001);
        // exp(clamp(0)) == 1.0 (neutral)
        assert!((clamp_adjustment(0.0).exp() - 1.0).abs() < 0.001);
    }

    // Note: Testing with a real ONNX model requires a model file.
    // Integration tests can use a dummy ONNX model created via the Python script.
}
