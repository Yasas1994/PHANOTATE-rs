//! Feature extraction for ML-based ORF scoring.
//!
//! Extracts a fixed-length feature vector from each `Orf` suitable for
//! feeding into a lightweight ONNX regression model.

use crate::orf::Orf;

/// Number of features extracted per ORF.
pub const NUM_FEATURES: usize = 13;

/// Fixed-length feature vector for ML inference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrfFeatures(pub [f32; NUM_FEATURES]);

impl OrfFeatures {
    /// Return the feature vector as a slice.
    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }

    /// Return the feature vector as a fixed-size array reference.
    pub fn as_array(&self) -> &[f32; NUM_FEATURES] {
        &self.0
    }
}

/// Column names for TSV export, in order.
pub const FEATURE_NAMES: [&str; NUM_FEATURES] = [
    "log_length",
    "rbs_score_norm",
    "log_hold",
    "pstop",
    "weight_rbs_log",
    "start_codon_atg",
    "start_codon_gtg",
    "start_codon_ttg",
    "gc_content",
    "frame_fwd",
    "frame_1",
    "frame_2",
    "frame_3",
];

impl Orf {
    /// Extract a fixed-length feature vector for ML inference.
    ///
    /// Features are normalised to roughly zero-mean, unit-variance ranges
    /// where possible, and all are cast to `f32` for ONNX compatibility.
    pub fn extract_features(&self) -> OrfFeatures {
        let mut features = [0.0f32; NUM_FEATURES];

        // 0. log(ORF length in nucleotides)
        let length = if self.start <= self.stop {
            self.stop - self.start + 1
        } else {
            self.start - self.stop + 1
        };
        features[0] = (length as f32).ln();

        // 1. RBS score normalised to [0, 1]
        features[1] = (self.rbs_score as f32) / 27.0;

        // 2. log(hold) — hold is a product, so log-space is more stable
        features[2] = self.hold.ln() as f32;

        // 3. P(stop) for this ORF
        features[3] = self.pstop as f32;

        // 4. log(weight_rbs)
        features[4] = self.weight_rbs.ln() as f32;

        // 5-7. Start codon one-hot (ATG, GTG, TTG)
        let sc = self.start_codon();
        features[5] = if sc == b"atg" { 1.0 } else { 0.0 };
        features[6] = if sc == b"gtg" { 1.0 } else { 0.0 };
        features[7] = if sc == b"ttg" { 1.0 } else { 0.0 };

        // 8. GC content of the ORF sequence
        let gc_count = self
            .seq
            .iter()
            .filter(|&&b| b == b'g' || b == b'c')
            .count();
        features[8] = if !self.seq.is_empty() {
            gc_count as f32 / self.seq.len() as f32
        } else {
            0.0
        };

        // 9. Forward strand indicator
        features[9] = if self.frame > 0 { 1.0 } else { 0.0 };

        // 10-12. Frame one-hot (absolute value: 1, 2, 3)
        let abs_frame = self.frame.abs();
        features[10] = if abs_frame == 1 { 1.0 } else { 0.0 };
        features[11] = if abs_frame == 2 { 1.0 } else { 0.0 };
        features[12] = if abs_frame == 3 { 1.0 } else { 0.0 };

        OrfFeatures(features)
    }
}

/// Write feature vectors for a batch of ORFs as a TSV.
///
/// Each row corresponds to one ORF.  Columns are the feature values in
/// the order defined by [`FEATURE_NAMES`].  If `include_header` is true,
/// a header row is written first.
pub fn write_features_tsv<W: std::io::Write>(
    writer: &mut W,
    orfs: &[Orf],
    include_header: bool,
) -> std::io::Result<()> {
    if include_header {
        writeln!(writer, "{}", FEATURE_NAMES.join("\t"))?;
    }
    for orf in orfs {
        let f = orf.extract_features();
        let vals: Vec<String> = f.0.iter().map(|v| format!("{:.6}", v)).collect();
        writeln!(writer, "{}", vals.join("\t"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_orf() -> Orf {
        Orf {
            start: 100,
            stop: 300,
            frame: 1,
            seq: b"atggctagctagctagc".to_vec(),
            rbs_score: 15,
            pstop: 0.05,
            weight_rbs: 2.5,
            hold: 0.8,
            weight: -1.0,
        }
    }

    #[test]
    fn test_feature_length() {
        let orf = test_orf();
        let f = orf.extract_features();
        assert_eq!(f.0.len(), NUM_FEATURES);
    }

    #[test]
    fn test_log_length() {
        let orf = test_orf();
        let f = orf.extract_features();
        // length = 201, ln(201) ≈ 5.303
        assert!((f.0[0] - 201.0f32.ln()).abs() < 0.01);
    }

    #[test]
    fn test_rbs_score_norm() {
        let orf = test_orf();
        let f = orf.extract_features();
        assert!((f.0[1] - 15.0 / 27.0).abs() < 0.001);
    }

    #[test]
    fn test_start_codon_onehot() {
        let orf = test_orf();
        let f = orf.extract_features();
        assert_eq!(f.0[5], 1.0); // ATG
        assert_eq!(f.0[6], 0.0); // GTG
        assert_eq!(f.0[7], 0.0); // TTG
    }

    #[test]
    fn test_start_codon_gtg() {
        let mut orf = test_orf();
        orf.seq = b"gtggctagctagctagc".to_vec();
        let f = orf.extract_features();
        assert_eq!(f.0[5], 0.0); // ATG
        assert_eq!(f.0[6], 1.0); // GTG
        assert_eq!(f.0[7], 0.0); // TTG
    }

    #[test]
    fn test_gc_content() {
        let orf = test_orf();
        let f = orf.extract_features();
        // seq = "atggctagctagctagc" -> g/c count = 9, len = 17
        let expected = 9.0 / 17.0;
        assert!((f.0[8] - expected).abs() < 0.001);
    }

    #[test]
    fn test_frame_onehot() {
        let orf = test_orf();
        let f = orf.extract_features();
        assert_eq!(f.0[9], 1.0);  // fwd
        assert_eq!(f.0[10], 1.0); // frame 1
        assert_eq!(f.0[11], 0.0); // frame 2
        assert_eq!(f.0[12], 0.0); // frame 3
    }

    #[test]
    fn test_reverse_frame() {
        let mut orf = test_orf();
        orf.frame = -2;
        let f = orf.extract_features();
        assert_eq!(f.0[9], 0.0);  // not fwd
        assert_eq!(f.0[10], 0.0); // not frame 1
        assert_eq!(f.0[11], 1.0); // frame 2
        assert_eq!(f.0[12], 0.0); // not frame 3
    }

    #[test]
    fn test_tsv_export() {
        let orfs = vec![test_orf()];
        let mut buf = Vec::new();
        write_features_tsv(&mut buf, &orfs, true).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(s.starts_with("log_length"));
        let lines: Vec<&str> = s.lines().collect();
        assert_eq!(lines.len(), 2); // header + 1 data row
    }

    #[test]
    fn test_tsv_no_header() {
        let orfs = vec![test_orf()];
        let mut buf = Vec::new();
        write_features_tsv(&mut buf, &orfs, false).unwrap();
        let s = String::from_utf8(buf).unwrap();
        let lines: Vec<&str> = s.lines().collect();
        assert_eq!(lines.len(), 1); // just data row
    }
}
