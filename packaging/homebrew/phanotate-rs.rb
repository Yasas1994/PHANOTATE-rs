class PhanotateRs < Formula
  desc "Fast Rust implementation of PHANOTATE with automatic genetic-code detection"
  homepage "https://github.com/Yasas1994/PHANOTATE-rs"
  url "https://github.com/Yasas1994/PHANOTATE-rs/archive/refs/tags/v0.1.2.tar.gz"
  sha256 "PLACEHOLDER_SHA256"
  license "GPL-3.0"

  depends_on "rust" => :build

  def install
    system "cargo", "install", *std_cargo_args
  end

  test do
    system "#{bin}/phanotate-rs", "--version"
  end
end
