#!/usr/bin/env bash
# PHANOTATE-rs install script
# Usage: curl -fsSL https://raw.githubusercontent.com/deprekate/phanotate-rs/main/install.sh | bash

set -euo pipefail

REPO="Yasas1994/PHANOTATE-rs"
BINARY="phanotate-rs"

# Detect OS and architecture
detect_target() {
    local os arch
    case "$(uname -s)" in
        Linux)     os="unknown-linux" ;;
        Darwin)    os="apple-darwin" ;;
        MINGW*|MSYS*|CYGWIN*) os="pc-windows-msvc" ;;
        *) echo "Unsupported OS: $(uname -s)" >&2; exit 1 ;;
    esac

    case "$(uname -m)" in
        x86_64)  arch="x86_64" ;;
        aarch64|arm64) arch="aarch64" ;;
        *) echo "Unsupported architecture: $(uname -m)" >&2; exit 1 ;;
    esac

    if [ "$os" = "pc-windows-msvc" ]; then
        echo "${arch}-${os}"
    elif [ "$os" = "unknown-linux" ]; then
        # Prefer musl for maximum portability, fall back to gnu
        echo "${arch}-${os}-musl"
    else
        echo "${arch}-${os}"
    fi
}

# Find latest release version
get_latest_version() {
    curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" \
        | grep '"tag_name":' \
        | sed -E 's/.*"([^"]+)".*/\1/'
}

# Download and install
download_and_install() {
    local target version url tmpdir install_dir
    target="$1"
    version="$2"

    # Map musl target to actual release artifact (fallback to gnu if musl not available)
    local artifact_target="$target"
    if [ "$target" = "x86_64-unknown-linux-musl" ]; then
        # Check if musl build exists, otherwise fall back to gnu
        if ! curl -fsSL -o /dev/null -w "%{http_code}" "https://github.com/${REPO}/releases/download/${version}/phanotate-rs-${target}.tar.gz" | grep -q "200"; then
            artifact_target="x86_64-unknown-linux-gnu"
        fi
    fi

    if [[ "$target" == *"windows"* ]]; then
        url="https://github.com/${REPO}/releases/download/${version}/phanotate-rs-${artifact_target}.zip"
    else
        url="https://github.com/${REPO}/releases/download/${version}/phanotate-rs-${artifact_target}.tar.gz"
    fi

    tmpdir=$(mktemp -d)
    trap 'rm -rf "$tmpdir"' EXIT

    echo "Downloading ${BINARY} ${version} for ${target}..."
    if ! curl -fsSL "$url" -o "${tmpdir}/archive"; then
        echo "Error: Failed to download ${url}" >&2
        echo "Please check that a release exists for your platform." >&2
        exit 1
    fi

    if [[ "$url" == *.zip ]]; then
        unzip -q "${tmpdir}/archive" -d "$tmpdir"
    else
        tar -xzf "${tmpdir}/archive" -C "$tmpdir"
    fi

    # Determine install directory
    if [ -n "${INSTALL_DIR:-}" ]; then
        install_dir="$INSTALL_DIR"
    elif [ -w /usr/local/bin ]; then
        install_dir="/usr/local/bin"
    elif [ -d "$HOME/.local/bin" ]; then
        install_dir="$HOME/.local/bin"
    else
        install_dir="$HOME/.local/bin"
        mkdir -p "$install_dir"
    fi

    # Install
    if [[ "$target" == *"windows"* ]]; then
        mv "${tmpdir}/${BINARY}.exe" "${install_dir}/"
        echo "Installed ${BINARY}.exe to ${install_dir}"
    else
        mv "${tmpdir}/${BINARY}" "${install_dir}/"
        chmod +x "${install_dir}/${BINARY}"
        echo "Installed ${BINARY} to ${install_dir}"
    fi

    # Verify
    if command -v "$BINARY" >/dev/null 2>&1; then
        echo "$(${BINARY} --version)"
    else
        echo ""
        echo "WARNING: ${install_dir} is not in your PATH."
        echo "Add the following to your shell profile:"
        echo "  export PATH=\"${install_dir}:\$PATH\""
    fi
}

main() {
    local version target

    if [ -n "${VERSION:-}" ]; then
        version="$VERSION"
    else
        version=$(get_latest_version)
    fi

    target=$(detect_target)
    download_and_install "$target" "$version"
}

main "$@"
