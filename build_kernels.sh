#!/usr/bin/env bash
set -euo pipefail

_ea_input="${EA_BIN:-ea}"
if [[ "$_ea_input" == */* ]]; then
    EA="$(cd "$(dirname "$_ea_input")" && pwd)/$(basename "$_ea_input")"
else
    EA="$_ea_input"
fi
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LIB_DIR="$SCRIPT_DIR/src/eastack/lib"
KERNEL_DIR="$SCRIPT_DIR/kernels"

case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*|Windows_NT)
        EXT=".dll"
        PREFIX=""
        ;;
    *)
        EXT=".so"
        PREFIX="lib"
        ;;
esac

mkdir -p "$LIB_DIR"

for kernel in stack; do
    OUTNAME="${PREFIX}${kernel}${EXT}"
    echo "Compiling ${kernel}.ea -> ${OUTNAME}"
    (cd "$LIB_DIR" && "$EA" "$KERNEL_DIR/${kernel}.ea" --lib -o "$OUTNAME")
done

rm -f "$LIB_DIR"/*.o
echo "Done. Libraries in $LIB_DIR:"
ls -la "$LIB_DIR"/${PREFIX}*${EXT}
