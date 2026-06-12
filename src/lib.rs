//! PHANOTATE-rs library — gene caller for phage genomes.
//!
//! Re-exports internal modules so that integration tests in `tests/` can
//! access crate-private items (e.g. `detect_table::CANDIDATE_TABLES`).

pub mod bellman_ford;
pub mod codon_table;
pub mod detect_table;
pub mod gcfp;
pub mod genome;
pub mod graph;
pub mod ml_features;
pub mod orf;
pub mod output;
pub mod weights;

#[cfg(feature = "ml")]
pub mod ml_scorer;

// Include Python bindings module only when the `python` feature is enabled.
// This avoids linking against libpython during `cargo test --lib`.
#[cfg(feature = "python")]
mod lib_python;
