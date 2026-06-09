//! GC Frame Plot computation.
//!
//! Algorithm from the reference:
//! 1. Slide a 120 bp window (40 codons) one base at a time over all three forward-strand frames.
//!    Record the % GC content of the codon starting at each base in each frame.
//! 2. The `get()` method interleaves the three frames to produce per-base-position arrays.

use std::collections::VecDeque;

const WINDOW_CODONS: usize = 40; // 120 bp / 3

pub struct GCframe {
    window: usize,
    states: [usize; 3],
    state_idx: usize,
    bases: [VecDeque<u8>; 3],
    frequency: [[usize; 5]; 3], // a,t,c,g,-
    total: [VecDeque<usize>; 3],
}

impl Default for GCframe {
    fn default() -> Self {
        Self::new()
    }
}

impl GCframe {
    pub fn new() -> Self {
        let dash = b'-'; // placeholder
        let mut frequency = [[0usize; 5]; 3];
        for freq_row in frequency.iter_mut() {
            freq_row[4] = WINDOW_CODONS; // '-' count
        }
        GCframe {
            window: WINDOW_CODONS,
            states: [1, 2, 3],
            state_idx: 0,
            bases: [
                VecDeque::from(vec![dash; WINDOW_CODONS]),
                VecDeque::from(vec![dash; WINDOW_CODONS]),
                VecDeque::from(vec![dash; WINDOW_CODONS]),
            ],
            frequency,
            total: [VecDeque::new(), VecDeque::new(), VecDeque::new()],
        }
    }

    pub fn add_base(&mut self, base: u8) {
        let frame = self.states[self.state_idx] - 1; // 0,1,2
        self.state_idx = (self.state_idx + 1) % 3;

        let b_idx = base_to_idx(base);
        self.bases[frame].push_back(base);
        self.frequency[frame][b_idx] += 1;
        let removed = self.bases[frame].pop_front().unwrap();
        let r_idx = base_to_idx(removed);
        self.frequency[frame][r_idx] -= 1;

        let gc_count = self.frequency[frame][2] + self.frequency[frame][3]; // c + g
        self.total[frame].push_back(gc_count);
    }

    /// Finalize and return per-position [frame1, frame2, frame3] GC counts.
    /// The reference adds extra trailing entries by shifting frames.
    pub fn get(mut self) -> Vec<[usize; 3]> {
        // "close" - shift half window more times
        let half = self.window / 2;
        for _ in 0..half {
            for frame in 0..3 {
                self.total[frame].pop_front();
                let removed = self.bases[frame].pop_front().unwrap();
                let r_idx = base_to_idx(removed);
                self.frequency[frame][r_idx] -= 1;
                let gc_count = self.frequency[frame][2] + self.frequency[frame][3];
                self.total[frame].push_back(gc_count);
            }
        }

        let mut freq: Vec<[usize; 3]> = Vec::new();
        freq.push([20, 20, 20]);

        let len = self.total[2].len().saturating_sub(1);
        for i in 0..len {
            freq.push([self.total[0][i], self.total[1][i], self.total[2][i]]);
            freq.push([self.total[1][i], self.total[2][i], self.total[0][i + 1]]);
            freq.push([self.total[2][i], self.total[0][i + 1], self.total[1][i + 1]]);
        }

        let i = len;
        freq.push([self.total[0][i], self.total[1][i], self.total[2][i]]);
        if i < self.total[0].len() - 1 {
            freq.push([self.total[1][i], self.total[2][i], self.total[0][i + 1]]);
        }
        if i < self.total[1].len() - 1 {
            freq.push([self.total[2][i], self.total[0][i + 1], self.total[1][i + 1]]);
        }

        freq
    }
}

fn base_to_idx(base: u8) -> usize {
    match base {
        b'a' => 0,
        b't' => 1,
        b'c' => 2,
        b'g' => 3,
        _ => 4,
    }
}

pub fn max_idx(a: usize, b: usize, c: usize) -> usize {
    if a > b && a > c {
        1
    } else if b > c {
        2
    } else {
        3
    }
}

pub fn min_idx(a: usize, b: usize, c: usize) -> usize {
    if a > b && b > c {
        3
    } else if a > b {
        2
    } else if a > c {
        3
    } else {
        1
    }
}

#[cfg(test)]
mod debug_tests {
    use super::*;
    use crate::genome::read_fasta;

    #[test]
    fn debug_gcframe() {
        let genomes = read_fasta("../PHANOTATE/tests/phiX174.fasta").unwrap();
        for genome in genomes {
            let dna = &genome.seq;
            let mut frame_plot = GCframe::new();
            for &base in dna {
                frame_plot.add_base(base);
            }
            let gc = frame_plot.get();
            println!("len(gc): {}", gc.len());
            for i in [1, 100, 295, 1000] {
                println!("gc[{}]: {:?}", i, gc[i]);
            }
        }
    }
}

#[cfg(test)]
mod debug_tests2 {
    use super::*;
    use crate::genome::read_fasta;

    #[test]
    fn check_total_lengths() {
        let genomes = read_fasta("../PHANOTATE/tests/phiX174.fasta").unwrap();
        for genome in genomes {
            let dna = &genome.seq;
            let mut frame_plot = GCframe::new();
            for &base in dna {
                frame_plot.add_base(base);
            }
            println!(
                "total lengths before close: [{}, {}, {}]",
                frame_plot.total[0].len(),
                frame_plot.total[1].len(),
                frame_plot.total[2].len()
            );
            let gc = frame_plot.get();
            println!("len(gc): {}", gc.len());
        }
    }
}

#[cfg(test)]
mod debug_tests3 {
    use super::*;
    use crate::genome::read_fasta;

    #[test]
    fn export_gc_rust() {
        let genomes = read_fasta("../PHANOTATE/tests/phiX174.fasta").unwrap();
        for genome in genomes {
            let dna = &genome.seq;
            let mut frame_plot = GCframe::new();
            for &base in dna {
                frame_plot.add_base(base);
            }
            let gc = frame_plot.get();
            let mut out = String::new();
            for (i, vals) in gc.iter().enumerate() {
                out.push_str(&format!("{} {} {} {}\n", i, vals[0], vals[1], vals[2]));
            }
            std::fs::write("/tmp/gc_rust.txt", out).unwrap();
            println!("Wrote Rust gc_pos_freq");
        }
    }
}

#[cfg(test)]
mod debug_tests4 {
    use super::*;

    #[test]
    fn test_max_min_idx() {
        assert_eq!(max_idx(10, 20, 30), 3);
        assert_eq!(max_idx(30, 20, 10), 1);
        assert_eq!(max_idx(10, 30, 20), 2);
        assert_eq!(min_idx(10, 20, 30), 1);
        assert_eq!(min_idx(30, 20, 10), 3);
        assert_eq!(min_idx(10, 30, 20), 1);
    }
}
