#!/bin/sh
# Fetch conda-forge win-64 packages WITHOUT conda and extract their Library/
# trees into a single prefix. A .conda file is a zip containing zstd tars;
# a .tar.bz2 is itself the payload. We only need headers + import libraries
# + DLLs, i.e. exactly what lands under Library/ in win-64 packages.
#
# Usage: fetch-conda-win64.sh <prefix> <pkg> [pkg...]
set -eu

PREFIX="$1"
shift
mkdir -p "$PREFIX" /tmp/condapkgs
cd /tmp/condapkgs

REPODATA_URL="https://conda.anaconda.org/conda-forge/win-64"

for spec in "$@"; do
    # "pkg" or "pkg:flavor" - flavor filters build strings (e.g. the BLAS
    # mutex packages come in openblas/netlib/mkl flavors; we want the one
    # whose import library points at the openblas DLL we ship).
    pkg=${spec%%:*}
    flavor=""
    case "$spec" in *:*) flavor=${spec#*:} ;; esac
    echo "== resolving $pkg${flavor:+ (flavor: $flavor)}"
    # Ask anaconda.org's API for the latest win-64 build of the package.
    fn=$(curl -fsSL "https://api.anaconda.org/package/conda-forge/$pkg/files" |
        EINSUMS_FLAVOR="$flavor" python3 -c '
import json, os, sys
files = json.load(sys.stdin)
flavor = os.environ.get("EINSUMS_FLAVOR", "")
# Some Windows-payload packages (compiler-rt_win-64) are published under
# the noarch subdir; accept both.
cands = [f for f in files if f.get("attrs", {}).get("subdir") in ("win-64", "noarch")]
if flavor:
    cands = [f for f in cands if flavor in f.get("basename", "")]
if not cands:
    sys.exit("no matching win-64 build of package")

def version_key(f):
    # Numeric-aware version ordering: "0.3.30" > "0.3.9". Non-numeric
    # segments (rc tags etc.) sort before numbered releases.
    parts = []
    for seg in f.get("version", "0").split("."):
        try:
            parts.append((1, int(seg)))
        except ValueError:
            parts.append((0, 0))
    return (parts, f.get("attrs", {}).get("build_number", 0))

cands.sort(key=version_key, reverse=True)
# The API listing can contain removed files; emit the newest few so the
# shell can fall back on a 404.
for f in cands[:5]:
    print(f["basename"])
')
    got=""
    for cand in $fn; do
        echo "   fetching $cand"
        if curl -fsSL -o pkg_archive "https://conda.anaconda.org/conda-forge/${cand}"; then
            got="$cand"
            break
        fi
        echo "   (unavailable, trying next)"
    done
    [ -n "$got" ] || { echo "no downloadable win-64 build of $pkg" >&2; exit 1; }
    fn="$got"

    rm -rf extract && mkdir extract
    case "$fn" in
    *.conda)
        # zip of {info,pkg}-*.tar.zst
        unzip -q pkg_archive -d conda_zip
        for t in conda_zip/pkg-*.tar.zst; do
            zstd -dc "$t" | tar -x -C extract
        done
        rm -rf conda_zip
        ;;
    *.tar.bz2)
        tar -xjf pkg_archive -C extract
        ;;
    *)
        echo "unknown package format: $fn" >&2
        exit 1
        ;;
    esac

    # win-64 packages put everything under Library/; merge into the prefix.
    if [ -d extract/Library ]; then
        rsync -a extract/Library/ "$PREFIX/Library/"
    fi
    rm -rf extract pkg_archive
done

echo "== winlibs prefix ready:"
ls "$PREFIX/Library" 2>/dev/null || true
