#!/bin/sh
# conda-forge's win-64 BLAS-family packages ship DLLs without import
# libraries (conda consumers on Windows link through other means). lld-link
# needs .lib import libraries, so generate them from each DLL's export table:
# llvm-readobj reads the COFF exports, llvm-dlltool emits the .lib.
#
# Usage: gen-implibs.sh <prefix>   (expects Library/bin/*.dll under it)
set -eu

PREFIX="$1"
BIN="$PREFIX/Library/bin"
LIB="$PREFIX/Library/lib"
mkdir -p "$LIB"

for dll in "$BIN"/*.dll; do
    base=$(basename "$dll" .dll)
    if [ -f "$LIB/$base.lib" ]; then
        continue # real import library already shipped (hdf5, zlib, ...)
    fi
    echo "== generating import library for $base.dll"
    def="/tmp/$base.def"
    {
        echo "LIBRARY $base.dll"
        echo "EXPORTS"
        # Export blocks look like:  "    Name: sgemm_" - take everything
        # after the summary header (the first Name: is the DLL itself).
        llvm-readobj --coff-exports "$dll" | awk '/^ *Name: /{print $2}' | tail -n +2
    } > "$def"
    n=$(($(wc -l < "$def") - 2))
    if [ "$n" -le 0 ]; then
        echo "   no exports found, skipping"
        continue
    fi
    llvm-dlltool -m i386:x86-64 -d "$def" -D "$base.dll" -l "$LIB/$base.lib"
    echo "   $n exports -> Library/lib/$base.lib"
done
