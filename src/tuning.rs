//! Central tuning constants for PHANOTATE-rs.
//!
//! Edit the values in this file before compiling a release build. During
//! development, compile with `--features dev` to override them via CLI flags.

/// Minimum rescue score (probability minus overlap penalty) to consider an ORF
/// for overlap rescue. Tuned on the standard 50-genome benchmark.
pub const RESCUE_THRESHOLD: f64 = 0.70;

/// Weight applied to the overlap-ratio penalty when computing rescue scores.
pub const OVERLAP_PENALTY_WEIGHT: f64 = 0.30;

/// Minimum length (in bp, including the stop codon) for an ORF to be eligible
/// for overlap rescue.
pub const MIN_RESCUE_ORF_LEN: usize = 90;
