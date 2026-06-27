//! Per-genome automatic threshold calibration for learned ORF scoring models.
//!
//! When `--auto-threshold` is enabled, PHANOTATE-rs can replace the fixed CLI
//! `--model-threshold` with a value computed from the distribution of predicted
//! ORF probabilities for the current genome.
//!
//! Two calibration strategies are supported:
//!
//! * `Length` — choose a threshold that selects roughly a target number of genes
//!   per kilobase of sequence. The target density can be tuned via the
//!   `PHANOTATE_MODEL_GENES_PER_KB` environment variable.
//! * `Percentile` — choose the probability at a given percentile of the ORF
//!   probability distribution. The percentile can be tuned via the
//!   `PHANOTATE_MODEL_THRESHOLD_PERCENTILE` environment variable.
//!
//! In both cases the computed threshold is clamped to be no lower than the
//! user-supplied CLI threshold, so the calibration can only make the caller
//! more selective, never more permissive.

use crate::orf::Orf;

/// Default expected gene density for `AutoThresholdMode::Length`.
///
/// One gene per kilobase is a conservative baseline for compact phage genomes.
pub const DEFAULT_GENES_PER_KB: f64 = 1.0;

/// Default percentile for `AutoThresholdMode::Percentile`.
///
/// The 80th percentile keeps the upper tail of high-confidence ORFs.
pub const DEFAULT_THRESHOLD_PERCENTILE: f64 = 80.0;

/// Environment variable that overrides [`DEFAULT_GENES_PER_KB`].
const ENV_GENES_PER_KB: &str = "PHANOTATE_MODEL_GENES_PER_KB";

/// Environment variable that overrides [`DEFAULT_THRESHOLD_PERCENTILE`].
const ENV_THRESHOLD_PERCENTILE: &str = "PHANOTATE_MODEL_THRESHOLD_PERCENTILE";

/// Strategy used to calibrate the model decision threshold per genome.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum AutoThresholdMode {
    /// Use the fixed CLI `--model-threshold` value.
    #[default]
    None,
    /// Select a threshold that yields roughly `PHANOTATE_MODEL_GENES_PER_KB`
    /// genes per kilobase.
    Length,
    /// Select the probability at percentile `PHANOTATE_MODEL_THRESHOLD_PERCENTILE`
    /// of the ORF probability distribution.
    Percentile,
}

/// Trait abstracting over ORF probability scorers.
///
/// This allows unit tests to inject deterministic scorers without loading an
/// ONNX model.
pub trait OrfScorer {
    /// Return the positive-class probability for `orf`.
    fn probability_for_orf(&self, orf: &Orf) -> f64;
}

impl OrfScorer for crate::onnx_scorer::OnnxScorer {
    fn probability_for_orf(&self, orf: &Orf) -> f64 {
        self.probability_for_orf(orf)
    }
}

/// Parameters that tune the automatic threshold calculation.
///
/// Values are normally read from environment variables via
/// [`read_threshold_params`], but they can be constructed directly in tests to
/// avoid mutating global process state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThresholdParams {
    /// Target number of genes per kilobase for length-based calibration.
    pub genes_per_kb: f64,
    /// Percentile of the ORF probability distribution for percentile-based
    /// calibration.
    pub percentile: f64,
}

impl Default for ThresholdParams {
    fn default() -> Self {
        Self {
            genes_per_kb: DEFAULT_GENES_PER_KB,
            percentile: DEFAULT_THRESHOLD_PERCENTILE,
        }
    }
}

/// Read threshold calibration parameters from environment variables.
///
/// Unset, malformed, non-finite, or out-of-range values fall back to the
/// documented defaults.
pub fn read_threshold_params() -> ThresholdParams {
    ThresholdParams {
        genes_per_kb: parse_positive_env_var(ENV_GENES_PER_KB, DEFAULT_GENES_PER_KB),
        percentile: parse_percentile_env_var(
            ENV_THRESHOLD_PERCENTILE,
            DEFAULT_THRESHOLD_PERCENTILE,
        ),
    }
}

/// Parse a finite, strictly positive `f64` from an environment variable.
///
/// Returns `default` if the variable is unset, cannot be parsed, is non-finite,
/// or is not strictly positive.
fn parse_positive_env_var(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(default)
}

/// Parse a finite percentile value in `[0, 100]` from an environment variable.
///
/// Returns `default` if the variable is unset, cannot be parsed, is non-finite,
/// or lies outside the valid percentile range.
fn parse_percentile_env_var(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|v| v.is_finite() && (0.0..=100.0).contains(v))
        .unwrap_or(default)
}

/// Compute the effective model decision threshold for a genome.
///
/// This convenience wrapper reads [`ThresholdParams`] from environment variables
/// and delegates to [`compute_effective_model_threshold_with_params`]. Use the
/// `_with_params` variant in tests to avoid mutating global env vars.
///
/// * In `AutoThresholdMode::None` (or when `orfs` is empty), returns
///   `cli_threshold` unchanged.
/// * In `AutoThresholdMode::Length`, predicts the number of genes expected from
///   `contig_length` and [`ThresholdParams::genes_per_kb`], then returns the
///   probability of the ORF at that rank when probabilities are sorted
///   descending.
/// * In `AutoThresholdMode::Percentile`, returns the probability at the
///   configured percentile of the descending probability distribution.
///
/// The returned value is always at least `cli_threshold`.
pub fn compute_effective_model_threshold<S: OrfScorer>(
    orfs: &[Orf],
    orf_model: &S,
    contig_length: usize,
    mode: AutoThresholdMode,
    cli_threshold: f64,
) -> f64 {
    let params = read_threshold_params();
    compute_effective_model_threshold_with_params(
        orfs,
        orf_model,
        contig_length,
        mode,
        cli_threshold,
        &params,
    )
}

/// Compute the effective model decision threshold for a genome.
///
/// Core implementation that accepts an explicit [`ThresholdParams`] so tests can
/// inject values without touching environment variables.
///
/// * In `AutoThresholdMode::None` (or when `orfs` is empty), returns
///   `cli_threshold` unchanged.
/// * In `AutoThresholdMode::Length`, predicts the number of genes expected from
///   `contig_length` and `params.genes_per_kb`, then returns the probability of
///   the ORF at that rank when probabilities are sorted descending.
/// * In `AutoThresholdMode::Percentile`, returns the probability at
///   `params.percentile` of the descending probability distribution.
///
/// The returned value is always at least `cli_threshold`.
pub fn compute_effective_model_threshold_with_params<S: OrfScorer>(
    orfs: &[Orf],
    orf_model: &S,
    contig_length: usize,
    mode: AutoThresholdMode,
    cli_threshold: f64,
    params: &ThresholdParams,
) -> f64 {
    if mode == AutoThresholdMode::None || orfs.is_empty() {
        return cli_threshold;
    }

    let mut probs: Vec<f64> = orfs
        .iter()
        .map(|o| orf_model.probability_for_orf(o))
        .filter(|p| p.is_finite())
        .collect();

    if probs.is_empty() {
        return cli_threshold;
    }

    probs.sort_by(|a, b| b.total_cmp(a));

    let computed = match mode {
        AutoThresholdMode::Length => {
            let target = ((contig_length as f64 / 1000.0) * params.genes_per_kb).round() as usize;
            let idx = target.saturating_sub(1).min(probs.len().saturating_sub(1));
            probs[idx]
        }
        AutoThresholdMode::Percentile => {
            let idx = ((params.percentile / 100.0) * probs.len().saturating_sub(1) as f64)
                .round()
                .min(probs.len().saturating_sub(1) as f64) as usize;
            probs[idx]
        }
        AutoThresholdMode::None => unreachable!(),
    };

    computed.max(cli_threshold)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic scorer that maps `orf.start` to a probability.
    ///
    /// `orf.start` is 1-based, so `start = n` returns `probabilities[n - 1]`.
    struct MockScorer {
        probabilities: Vec<f64>,
    }

    impl OrfScorer for MockScorer {
        fn probability_for_orf(&self, orf: &Orf) -> f64 {
            self.probabilities[(orf.start - 1) % self.probabilities.len()]
        }
    }

    fn make_orfs(count: usize) -> Vec<Orf> {
        (0..count)
            .map(|i| Orf {
                start: i + 1,
                stop: i + 100,
                ..Orf::default()
            })
            .collect()
    }

    #[test]
    fn none_mode_returns_cli_threshold() {
        let orfs = make_orfs(5);
        let scorer = MockScorer {
            probabilities: vec![0.9, 0.7, 0.5, 0.3, 0.1],
        };
        let threshold = compute_effective_model_threshold_with_params(
            &orfs,
            &scorer,
            10_000,
            AutoThresholdMode::None,
            0.4,
            &ThresholdParams::default(),
        );
        assert_eq!(threshold, 0.4);
    }

    #[test]
    fn length_mode_returns_probability_at_target_index() {
        let orfs = make_orfs(5);
        let scorer = MockScorer {
            probabilities: vec![0.1, 0.3, 0.5, 0.7, 0.9],
        };
        // 5 ORFs over 10 kb with default 1.0 gene/kb => target = 10, clamped to 5.
        let threshold = compute_effective_model_threshold_with_params(
            &orfs,
            &scorer,
            10_000,
            AutoThresholdMode::Length,
            0.0,
            &ThresholdParams::default(),
        );
        // Descending probabilities: [0.9, 0.7, 0.5, 0.3, 0.1]; idx = 4 => 0.1.
        assert_eq!(threshold, 0.1);
    }

    #[test]
    fn percentile_mode_returns_probability_at_chosen_percentile() {
        let orfs = make_orfs(5);
        let scorer = MockScorer {
            probabilities: vec![0.1, 0.2, 0.3, 0.4, 0.5],
        };
        // Default 80th percentile over 5 items => idx = (0.8 * 4).round() = 3.
        // Descending probabilities: [0.5, 0.4, 0.3, 0.2, 0.1]; idx 3 => 0.2.
        let threshold = compute_effective_model_threshold_with_params(
            &orfs,
            &scorer,
            1_000,
            AutoThresholdMode::Percentile,
            0.0,
            &ThresholdParams::default(),
        );
        assert_eq!(threshold, 0.2);
    }

    #[test]
    fn computed_threshold_is_floored_by_cli_threshold() {
        let orfs = make_orfs(5);
        let scorer = MockScorer {
            probabilities: vec![0.1, 0.2, 0.3, 0.4, 0.5],
        };
        // Computed 80th-percentile threshold is 0.2; CLI floor is 0.6.
        let threshold = compute_effective_model_threshold_with_params(
            &orfs,
            &scorer,
            1_000,
            AutoThresholdMode::Percentile,
            0.6,
            &ThresholdParams::default(),
        );
        assert_eq!(threshold, 0.6);
    }

    #[test]
    fn empty_orfs_returns_cli_threshold() {
        let scorer = MockScorer {
            probabilities: vec![0.9, 0.7, 0.5],
        };
        let threshold = compute_effective_model_threshold_with_params(
            &[],
            &scorer,
            10_000,
            AutoThresholdMode::Length,
            0.35,
            &ThresholdParams::default(),
        );
        assert_eq!(threshold, 0.35);
    }

    #[test]
    fn non_finite_probabilities_are_ignored() {
        struct NanScorer;
        impl OrfScorer for NanScorer {
            fn probability_for_orf(&self, _orf: &Orf) -> f64 {
                f64::NAN
            }
        }

        let orfs = make_orfs(3);
        let threshold = compute_effective_model_threshold_with_params(
            &orfs,
            &NanScorer,
            1_000,
            AutoThresholdMode::Percentile,
            0.25,
            &ThresholdParams::default(),
        );
        assert_eq!(threshold, 0.25);
    }

    #[test]
    fn param_override_changes_length_result() {
        let orfs = make_orfs(5);
        let scorer = MockScorer {
            probabilities: vec![0.1, 0.3, 0.5, 0.7, 0.9],
        };

        // Default 1.0 gene/kb over 5 kb => target = 5 => last probability 0.1.
        let default_threshold = compute_effective_model_threshold_with_params(
            &orfs,
            &scorer,
            5_000,
            AutoThresholdMode::Length,
            0.0,
            &ThresholdParams::default(),
        );
        assert_eq!(default_threshold, 0.1);

        // Override to 0.2 gene/kb => target = 1 => highest probability 0.9.
        let override_threshold = compute_effective_model_threshold_with_params(
            &orfs,
            &scorer,
            5_000,
            AutoThresholdMode::Length,
            0.0,
            &ThresholdParams {
                genes_per_kb: 0.2,
                ..ThresholdParams::default()
            },
        );
        assert_eq!(override_threshold, 0.9);
    }

    #[test]
    fn param_override_changes_percentile_result() {
        let orfs = make_orfs(5);
        let scorer = MockScorer {
            probabilities: vec![0.1, 0.2, 0.3, 0.4, 0.5],
        };

        // Default 80th percentile => idx 3 => 0.2.
        let default_threshold = compute_effective_model_threshold_with_params(
            &orfs,
            &scorer,
            1_000,
            AutoThresholdMode::Percentile,
            0.0,
            &ThresholdParams::default(),
        );
        assert_eq!(default_threshold, 0.2);

        // Override to 0th percentile => idx 0 => highest probability 0.5.
        let override_threshold = compute_effective_model_threshold_with_params(
            &orfs,
            &scorer,
            1_000,
            AutoThresholdMode::Percentile,
            0.0,
            &ThresholdParams {
                percentile: 0.0,
                ..ThresholdParams::default()
            },
        );
        assert_eq!(override_threshold, 0.5);
    }

    #[test]
    fn invalid_genes_per_kb_falls_back_to_default() {
        let previous = std::env::var(ENV_GENES_PER_KB).ok();
        unsafe {
            std::env::set_var(ENV_GENES_PER_KB, "not-a-number");
        }
        let params = read_threshold_params();
        assert_eq!(params.genes_per_kb, DEFAULT_GENES_PER_KB);

        unsafe {
            match previous {
                Some(v) => std::env::set_var(ENV_GENES_PER_KB, v),
                None => std::env::remove_var(ENV_GENES_PER_KB),
            }
        }
    }

    #[test]
    fn invalid_percentile_falls_back_to_default() {
        let previous = std::env::var(ENV_THRESHOLD_PERCENTILE).ok();
        unsafe {
            std::env::set_var(ENV_THRESHOLD_PERCENTILE, "not-a-number");
        }
        let params = read_threshold_params();
        assert_eq!(params.percentile, DEFAULT_THRESHOLD_PERCENTILE);

        unsafe {
            match previous {
                Some(v) => std::env::set_var(ENV_THRESHOLD_PERCENTILE, v),
                None => std::env::remove_var(ENV_THRESHOLD_PERCENTILE),
            }
        }
    }

    #[test]
    fn valid_env_vars_override_defaults() {
        let previous_genes = std::env::var(ENV_GENES_PER_KB).ok();
        let previous_pct = std::env::var(ENV_THRESHOLD_PERCENTILE).ok();

        unsafe {
            std::env::set_var(ENV_GENES_PER_KB, "2.5");
            std::env::set_var(ENV_THRESHOLD_PERCENTILE, "95.0");
        }
        let params = read_threshold_params();
        assert_eq!(params.genes_per_kb, 2.5);
        assert_eq!(params.percentile, 95.0);

        unsafe {
            match previous_genes {
                Some(v) => std::env::set_var(ENV_GENES_PER_KB, v),
                None => std::env::remove_var(ENV_GENES_PER_KB),
            }
            match previous_pct {
                Some(v) => std::env::set_var(ENV_THRESHOLD_PERCENTILE, v),
                None => std::env::remove_var(ENV_THRESHOLD_PERCENTILE),
            }
        }
    }
}
