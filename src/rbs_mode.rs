//! Shared RBS scoring mode selector used by the CLI and Python bindings.

use clap::ValueEnum;
use std::str::FromStr;

/// How upstream start-codon motifs should be scored.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
pub enum RbsMode {
    /// Auto-detect Shine–Dalgarno vs non-SD motifs from the ORF training set.
    #[default]
    Auto,
    /// Force legacy Shine–Dalgarno scoring.
    Sd,
    /// Force non-Shine–Dalgarno motif discovery.
    #[value(name = "non-sd", alias = "non_sd")]
    NonSd,
    /// Use Prodigal-style SD bins with a non-SD fallback for display.
    Prodigal,
}

impl RbsMode {
    /// Return true when this mode uses the Prodigal-style RBS scanner.
    pub fn is_prodigal(self) -> bool {
        self == RbsMode::Prodigal
    }
}

impl FromStr for RbsMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "auto" => Ok(RbsMode::Auto),
            "sd" => Ok(RbsMode::Sd),
            "non-sd" | "non_sd" => Ok(RbsMode::NonSd),
            "prodigal" => Ok(RbsMode::Prodigal),
            _ => Err(format!("unknown RBS mode: {}", s)),
        }
    }
}
