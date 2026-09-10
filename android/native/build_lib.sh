#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE_DIR="$SCRIPT_DIR/navigation_core"
OUTPUT_DIR="$SCRIPT_DIR/build"

mkdir -p "$OUTPUT_DIR"

echo "Compiling Native C++ Navigation Core..."

SRC_FILES=(
    "$CORE_DIR/strapdown_ins.cpp"
    "$CORE_DIR/eskf_core.cpp"
    "$CORE_DIR/outage_detector.cpp"
    "$CORE_DIR/zupt_detector.cpp"
    "$CORE_DIR/nav_engine.cpp"
    "$CORE_DIR/nav_core_c_api.cpp"
)


UNAME_S="$(uname -s)"
if [ "$UNAME_S" = "Darwin" ]; then
    LIB_NAME="libnav_core.dylib"
    clang++ -std=c++17 -O3 -fPIC -shared \
        -I"$CORE_DIR" \
        "${SRC_FILES[@]}" \
        -o "$OUTPUT_DIR/$LIB_NAME"
    # Also create .so symlink for portable ctypes loading
    ln -sf "$LIB_NAME" "$OUTPUT_DIR/libnav_core.so"
else
    LIB_NAME="libnav_core.so"
    g++ -std=c++17 -O3 -fPIC -shared \
        -I"$CORE_DIR" \
        "${SRC_FILES[@]}" \
        -o "$OUTPUT_DIR/$LIB_NAME"
fi

echo "Successfully built $OUTPUT_DIR/$LIB_NAME"
