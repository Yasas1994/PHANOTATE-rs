/// Central tuning constants for PHANOTATE-rs.
///
/// Edit the values in this file before compiling a release build. During
/// development, compile with `--features dev` to override them via CLI flags.
///
/// Probability threshold that separates ORF "reward" from "penalty".
///
/// An ORF whose predicted probability is below this value gets a positive
/// graph weight (discouraged); above it gets a negative weight (encouraged).
/// The default 0.5 is the natural logit decision boundary.
pub const DEFAULT_MODEL_THRESHOLD: f64 = 0.5;

/// Scale factor applied to the ONNX model's log-odds score before it is used
/// as a graph edge weight. A value of 1.0 leaves the score unchanged.
pub const DEFAULT_MODEL_SCALE: f64 = 1.0;

/// Minimum rescue score (probability minus overlap penalty) to consider an ORF
/// for overlap rescue. Tuned on the standard 50-genome benchmark.
pub const RESCUE_THRESHOLD: f64 = 0.70;

/// Weight applied to the overlap-ratio penalty when computing rescue scores.
pub const OVERLAP_PENALTY_WEIGHT: f64 = 0.30;

/// Minimum length (in bp, including the stop codon) for an ORF to be eligible
/// for overlap rescue.
pub const MIN_RESCUE_ORF_LEN: usize = 90;

/// Default per-rescued-gene penalty used by the overlap-rescue DP.
pub const OVERLAP_LAMBDA: f64 = 0.0;
