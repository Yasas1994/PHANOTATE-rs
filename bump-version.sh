#!/usr/bin/env bash
# PHANOTATE-rs version bumper
# Usage: ./bump-version.sh <new-version>
# Example: ./bump-version.sh 0.1.2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NEW_VERSION="${1:-}"

if [ -z "$NEW_VERSION" ]; then
    echo "Usage: $0 <new-version>"
    echo "Example: $0 0.1.2"
    exit 1
fi

# Validate version format (semver-like: x.y.z or x.y)
if ! echo "$NEW_VERSION" | grep -qE '^[0-9]+\.[0-9]+(\.[0-9]+)?$'; then
    echo "Error: Version must be in format x.y.z or x.y (e.g., 0.1.2)"
    exit 1
fi

# Read current version
CURRENT_VERSION=$(tr -d '[:space:]' < VERSION)
echo "Bumping version: $CURRENT_VERSION -> $NEW_VERSION"

# ---------------------------------------------------------------------------
# 1. Update VERSION file
# ---------------------------------------------------------------------------
echo "$NEW_VERSION" > VERSION
echo "  ✓ VERSION"

# ---------------------------------------------------------------------------
# 2. Update Cargo.toml
# ---------------------------------------------------------------------------
# Only match "version = " at the start of a line (the package version),
# not edition or dependency versions.
perl -i -pe "s/^version = \"[^\"]+\"/version = \"$NEW_VERSION\"/" Cargo.toml
if grep -q "^version = \"$NEW_VERSION\"" Cargo.toml; then
    echo "  ✓ Cargo.toml"
else
    echo "  ✗ Failed to update Cargo.toml"
    exit 1
fi

# ---------------------------------------------------------------------------
# 3. Update packaging/conda/meta.yaml
# ---------------------------------------------------------------------------
perl -i -pe "s/{% set version = \"[^\"]+\" %}/{% set version = \"$NEW_VERSION\" %}/" packaging/conda/meta.yaml
if grep -q "{% set version = \"$NEW_VERSION\" %}" packaging/conda/meta.yaml; then
    echo "  ✓ packaging/conda/meta.yaml"
else
    echo "  ✗ Failed to update packaging/conda/meta.yaml"
    exit 1
fi

# ---------------------------------------------------------------------------
# 4. Update packaging/homebrew/phanotate-rs.rb
# ---------------------------------------------------------------------------
perl -i -pe "s|refs/tags/v[^/]+\.tar\.gz|refs/tags/v$NEW_VERSION.tar.gz|" packaging/homebrew/phanotate-rs.rb
if grep -q "refs/tags/v$NEW_VERSION.tar.gz" packaging/homebrew/phanotate-rs.rb; then
    echo "  ✓ packaging/homebrew/phanotate-rs.rb"
else
    echo "  ✗ Failed to update packaging/homebrew/phanotate-rs.rb"
    exit 1
fi

# ---------------------------------------------------------------------------
# 5. Update Cargo.lock (by running cargo update)
# ---------------------------------------------------------------------------
if command -v cargo &> /dev/null; then
    cargo update -w &> /dev/null
    echo "  ✓ Cargo.lock (via cargo update)"
else
    echo "  ! Cargo not found — skipping Cargo.lock update"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "Version bumped to $NEW_VERSION"
echo ""
echo "Next steps:"
echo "  1. Review the changes: git diff"
echo "  2. Commit: git add -A && git commit -m \"chore: bump version to $NEW_VERSION\""
echo "  3. Tag: git tag v$NEW_VERSION"
echo "  4. Push: git push origin main && git push origin v$NEW_VERSION"
echo ""
echo "The GitHub Actions release workflow will trigger automatically on the tag push."
echo "Remember to update the SHA256 placeholders in packaging files after the"
echo "release tarball is generated (or use the release workflow's computed values)."
