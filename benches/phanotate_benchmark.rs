use criterion::{criterion_group, criterion_main, Criterion};
use std::process::Command;

fn benchmark_phix174(c: &mut Criterion) {
    c.bench_function("phiX174", |b| {
        b.iter(|| {
            Command::new("./target/release/phanotate-rs")
                .args(["-i", "../PHANOTATE/tests/phiX174.fasta", "-f", "sco"])
                .output()
                .expect("failed to execute")
        });
    });
}

fn benchmark_nc_001416(c: &mut Criterion) {
    c.bench_function("NC_001416.1", |b| {
        b.iter(|| {
            Command::new("./target/release/phanotate-rs")
                .args(["-i", "../PHANOTATE/tests/NC_001416.1.fasta", "-f", "sco"])
                .output()
                .expect("failed to execute")
        });
    });
}

fn benchmark_nc_000866(c: &mut Criterion) {
    c.bench_function("NC_000866.1", |b| {
        b.iter(|| {
            Command::new("./target/release/phanotate-rs")
                .args(["-i", "../PHANOTATE/tests/NC_000866.1.fasta", "-f", "sco"])
                .output()
                .expect("failed to execute")
        });
    });
}

criterion_group!(
    benches,
    benchmark_phix174,
    benchmark_nc_001416,
    benchmark_nc_000866
);
criterion_main!(benches);
