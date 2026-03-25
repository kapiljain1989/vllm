#!/bin/bash
set -e

echo "Building vLLM wheel using Docker..."

docker run --rm -v $(pwd):/workspace \
  nvidia/cuda:12.9.1-devel-ubuntu20.04 \
  bash -c '
    set -e
    cd /workspace

    echo "Installing system dependencies..."
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -y
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        git cmake ninja-build ccache \
        gcc-10 g++-10 curl ca-certificates

    echo "Setting GCC 10 as default..."
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 110 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-10

    echo "Upgrading pip..."
    python3 -m pip install --upgrade pip

    echo "Installing build dependencies (with version workarounds)..."
    # Install specific versions that exist
    python3 -m pip install \
        "setuptools>=75.0,<81.0.0" \
        setuptools-scm>=8 \
        wheel \
        cmake>=3.26.1 \
        ninja \
        packaging>=24.2 \
        torch==2.10.0 \
        jinja2>=3.1.6 \
        regex \
        build \
        "protobuf>=5.29.6,!=6.30.*,!=6.31.*,!=6.32.*,!=6.33.0.*,!=6.33.1.*,!=6.33.2.*,!=6.33.3.*,!=6.33.4.*"

    echo "Building wheel..."
    python3 setup.py bdist_wheel --dist-dir=/workspace/dist

    echo "Renaming to manylinux..."
    cd /workspace/dist
    for w in *.whl; do
        if [[ "$w" == *"linux"* ]]; then
            new_wheel="${w/linux/manylinux_2_31}"
            mv "$w" "$new_wheel"
            echo "Renamed: $w -> $new_wheel"
        fi
    done

    echo ""
    echo "Build complete! Wheel location:"
    ls -lh *.whl
  '

echo ""
echo "Wheel built successfully in ./dist/"
