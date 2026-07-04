//! Shared per-genome ORF signal computation.
//!
//! Centralises the GC-frame hold calculation and the Prodigal-style hexamer
//! cscore calculation so that the annotation pipeline and `--export-features`
//! produce consistent feature vectors.

use crate::gcfp::{max_idx, min_idx, GCframe};
use crate::hexamer::HexamerModel;
use crate::orf::Orf;

/// Compute the GC-frame "hold" multiplier for every ORF.
///
/// `hold` is the product over in-frame codons of P(not stop) raised to a
/// position-specific weight derived from the GC frame plot.  It is computed in
/// log-space for numerical stability and stored in each ORF.  The default
/// coding-potential multiplier is set to `1.0 / hold`.
pub fn compute_hold(orfs: &mut [Orf], dna: &[u8], _rc_dna: &[u8]) {
    if orfs.is_empty() {
        return;
    }
    let gc_pos_freq = gc_frame_plot(dna);
    compute_hold_with_plot(orfs, &gc_pos_freq);
}

/// Compute hold using an already-built GC frame plot.
///
/// This avoids rebuilding the GC frame plot in callers that already have it
/// available (e.g. the annotation pipeline).
pub fn compute_hold_with_plot(orfs: &mut [Orf], gc_pos_freq: &[[usize; 3]]) {
    if orfs.is_empty() {
        return;
    }
    let (pos_max, pos_min) = build_pos_max_min(orfs, gc_pos_freq);

    for orf in orfs.iter_mut() {
        let (start, stop) = (orf.start, orf.stop);
        let ln_pns = (1.0 - orf.pstop).ln();
        let mut log_hold = 0.0f64;
        if orf.frame > 0 {
            let mut base = start;
            while base < stop && base < gc_pos_freq.len() {
                let ind_max = max_idx(
                    gc_pos_freq[base][0],
                    gc_pos_freq[base][1],
                    gc_pos_freq[base][2],
                );
                let ind_min = min_idx(
                    gc_pos_freq[base][0],
                    gc_pos_freq[base][1],
                    gc_pos_freq[base][2],
                );
                log_hold += ln_pns * pos_max[ind_max] * pos_min[ind_min];
                base += 3;
            }
        } else {
            let mut base = start;
            while base > stop && base < gc_pos_freq.len() {
                let ind_max = max_idx(
                    gc_pos_freq[base][2],
                    gc_pos_freq[base][1],
                    gc_pos_freq[base][0],
                );
                let ind_min = min_idx(
                    gc_pos_freq[base][2],
                    gc_pos_freq[base][1],
                    gc_pos_freq[base][0],
                );
                log_hold += ln_pns * pos_max[ind_max] * pos_min[ind_min];
                if base >= 3 {
                    base -= 3;
                } else {
                    break;
                }
            }
        }
        orf.hold = log_hold.exp();
        // Default heuristic: GC-frame hold is the coding-potential multiplier.
        orf.coding_potential = 1.0 / orf.hold;
    }
}

/// Replace each ORF's `coding_potential` with a Prodigal-style hexamer
/// log-odds (cscore) multiplier.
///
/// `annotated` contains indices into `orfs` of trusted genes (e.g. annotated
/// CDS). If it is non-empty the model is trained from those ORFs. If it is
/// `Some(&[])` the model is trained unsupervised from all long ORFs. If it is
/// `None` the default signal-filtered training is used (requires RBS scores).
pub fn apply_hexamer_model(
    orfs: &mut [Orf],
    dna: &[u8],
    rc_dna: &[u8],
    annotated: Option<&[usize]>,
) {
    if orfs.is_empty() {
        return;
    }

    let model = match annotated {
        Some(a) if !a.is_empty() => {
            let refs: Vec<&Orf> = a.iter().map(|&i| &orfs[i]).collect();
            HexamerModel::from_annotated_orfs(&refs, dna, rc_dna)
        }
        Some(_) => HexamerModel::train_unsupervised(orfs, dna, rc_dna),
        None => HexamerModel::train(orfs, dna, rc_dna),
    };

    for orf in orfs.iter_mut() {
        orf.coding_potential = model.score_orf(orf);
        orf.cscore = model.cscore(orf);
    }
}

/// Compute GC-frame hold and, when requested, the hexamer coding potential.
pub fn compute_orf_signals(
    orfs: &mut [Orf],
    dna: &[u8],
    rc_dna: &[u8],
    use_hexamer: bool,
    annotated: Option<&[usize]>,
) {
    compute_hold(orfs, dna, rc_dna);
    if use_hexamer {
        apply_hexamer_model(orfs, dna, rc_dna, annotated);
    }
}

/// Compute GC-frame hold and, when requested, the hexamer coding potential,
/// using an already-built GC frame plot.
pub fn compute_orf_signals_with_plot(
    orfs: &mut [Orf],
    dna: &[u8],
    rc_dna: &[u8],
    gc_pos_freq: &[[usize; 3]],
    use_hexamer: bool,
    annotated: Option<&[usize]>,
) {
    compute_hold_with_plot(orfs, gc_pos_freq);
    if use_hexamer {
        apply_hexamer_model(orfs, dna, rc_dna, annotated);
    }
}

fn gc_frame_plot(dna: &[u8]) -> Vec<[usize; 3]> {
    let mut frame_plot = GCframe::new();
    for &base in dna {
        frame_plot.add_base(base);
    }
    frame_plot.get()
}

fn build_pos_max_min(orfs: &[Orf], gc_pos_freq: &[[usize; 3]]) -> ([f64; 4], [f64; 4]) {
    let mut pos_max = [1.0f64; 4];
    let mut pos_min = [1.0f64; 4];

    let mut by_stop: std::collections::BTreeMap<usize, Vec<&Orf>> =
        std::collections::BTreeMap::new();
    for orf in orfs {
        by_stop.entry(orf.stop).or_default().push(orf);
    }

    for (_, orfs_at_stop) in by_stop.iter_mut() {
        if orfs_at_stop[0].frame > 0 {
            orfs_at_stop.sort_by_key(|o| o.start);
        } else {
            orfs_at_stop.sort_by_key(|o| std::cmp::Reverse(o.start));
        }
    }

    for (_, orfs_at_stop) in by_stop {
        let mut selected = None;
        for orf in orfs_at_stop {
            if orf.start_codon() == b"atg" {
                selected = Some(orf);
                break;
            }
        }
        let orf = match selected {
            Some(o) => o,
            None => continue,
        };

        let (start, stop) = (orf.start, orf.stop);
        if start < stop {
            let n = ((stop - start) / 8) * 3;
            let mut base = start + n;
            while base + 36 < stop && base < gc_pos_freq.len() {
                let idx = max_idx(
                    gc_pos_freq[base][0],
                    gc_pos_freq[base][1],
                    gc_pos_freq[base][2],
                );
                pos_max[idx] += 1.0;
                let idx = min_idx(
                    gc_pos_freq[base][0],
                    gc_pos_freq[base][1],
                    gc_pos_freq[base][2],
                );
                pos_min[idx] += 1.0;
                base += 3;
            }
        } else {
            let n = ((start - stop) / 8) * 3;
            let mut base = start.saturating_sub(n);
            while base > stop + 36 && base < gc_pos_freq.len() {
                let idx = max_idx(
                    gc_pos_freq[base][2],
                    gc_pos_freq[base][1],
                    gc_pos_freq[base][0],
                );
                pos_max[idx] += 1.0;
                let idx = min_idx(
                    gc_pos_freq[base][2],
                    gc_pos_freq[base][1],
                    gc_pos_freq[base][0],
                );
                pos_min[idx] += 1.0;
                if base >= 3 {
                    base -= 3;
                } else {
                    break;
                }
            }
        }
    }

    let max_max = pos_max.iter().cloned().fold(0.0, f64::max);
    if max_max > 0.0 {
        for v in &mut pos_max {
            *v /= max_max;
        }
    }
    let max_min = pos_min.iter().cloned().fold(0.0, f64::max);
    if max_min > 0.0 {
        for v in &mut pos_min {
            *v /= max_min;
        }
    }

    (pos_max, pos_min)
}
