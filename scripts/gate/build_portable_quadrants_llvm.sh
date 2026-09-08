#!/usr/bin/env bash
# Build the LLVM 22.1.0 install tree Quadrants links into Linux aarch64 wheels.
#
# This runs *inside* the pinned manylinux_2_34_aarch64 image selected by
# resolve_wheel_matrix.py.  The critical part is using AlmaLinux 9's GCC 11,
# not the image's gcc-toolset-14: the latter's static unwind support may refer
# to _dl_find_object@GLIBC_2.35 even though the userspace itself is glibc 2.34.
set -euo pipefail

: "${PORTABLE_LLVM_VERSION:?set PORTABLE_LLVM_VERSION}"
: "${PORTABLE_LLVM_COMMIT:?set PORTABLE_LLVM_COMMIT}"
: "${PORTABLE_LLVM_BUILD_IMAGE:?set PORTABLE_LLVM_BUILD_IMAGE}"
: "${PORTABLE_LLVM_GCC_NVR:?set PORTABLE_LLVM_GCC_NVR}"

if [[ $# -ne 2 ]]; then
  echo "usage: $0 OUTPUT_ZIP PROVENANCE_FILE" >&2
  exit 2
fi

output_zip=$(realpath -m "$1")
provenance_file=$(realpath -m "$2")
work=${RUNNER_TEMP:-/tmp}/quadrants-portable-llvm
src=$work/llvm-project
build=$work/build
prefix=$work/install

rm -rf "$work"
mkdir -p "$work" "$(dirname "$output_zip")" "$(dirname "$provenance_file")"

arch=$(uname -m)
[[ "$arch" == aarch64 || "$arch" == arm64 ]] || {
  echo "portable Quadrants LLVM must be built natively on aarch64, got $arch" >&2
  exit 1
}

# Avoid a version-probe pipeline here. This script runs with pipefail, and the
# manylinux ldd probe can otherwise surface SIGPIPE as exit 141 before the
# actual LLVM build starts. Capture all output, then take the first line in
# Bash so the compatibility check itself cannot fail because of pipe plumbing.
glibc_output=$(ldd --version 2>&1)
glibc_line=${glibc_output%%$'\n'*}
glibc_version=$(sed -nE 's/.* ([0-9]+\.[0-9]+)$/\1/p' <<<"$glibc_line")
[[ "$glibc_version" == "2.34" ]] || {
  echo "expected a genuine glibc 2.34 userspace, got: $glibc_line" >&2
  exit 1
}

# The manylinux image deliberately puts a newer gcc-toolset first on PATH.
# Install and select the distro compiler explicitly so every LLVM object is
# built with the compiler whose unwind/static libraries are valid on glibc 2.34.
dnf install -y \
  "gcc-${PORTABLE_LLVM_GCC_NVR}" \
  "gcc-c++-${PORTABLE_LLVM_GCC_NVR}" \
  "libstdc++-static-${PORTABLE_LLVM_GCC_NVR}" zip \
  || dnf install -y --enablerepo=crb \
       "gcc-${PORTABLE_LLVM_GCC_NVR}" \
       "gcc-c++-${PORTABLE_LLVM_GCC_NVR}" \
       "libstdc++-static-${PORTABLE_LLVM_GCC_NVR}" zip
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

gcc_version=$($CC -dumpfullversion -dumpversion)
gxx_version=$($CXX -dumpfullversion -dumpversion)
[[ "$gcc_version" == 11.* && "$gxx_version" == 11.* ]] || {
  echo "expected AlmaLinux 9 GCC 11, got gcc=$gcc_version g++=$gxx_version" >&2
  exit 1
}
installed_gcc_nvr=$(rpm -q --qf '%{VERSION}-%{RELEASE}' gcc)
installed_gxx_nvr=$(rpm -q --qf '%{VERSION}-%{RELEASE}' gcc-c++)
[[ "$installed_gcc_nvr" == "$PORTABLE_LLVM_GCC_NVR" && \
   "$installed_gxx_nvr" == "$PORTABLE_LLVM_GCC_NVR" ]] || {
  echo "GCC package provenance mismatch: wanted $PORTABLE_LLVM_GCC_NVR, got gcc=$installed_gcc_nvr g++=$installed_gxx_nvr" >&2
  exit 1
}
static_libstdcxx=$($CXX -print-file-name=libstdc++.a)
[[ -f "$static_libstdcxx" ]] || {
  echo "GCC 11 static libstdc++ is missing: $static_libstdcxx" >&2
  exit 1
}

# Pin the build front-end too. These are build tools only; the resulting LLVM
# libraries are compiled by the GCC 11 selected above.
py=/opt/python/cp311-cp311/bin/python
[[ -x "$py" ]] || { echo "manylinux cp311 interpreter is missing" >&2; exit 1; }
"$py" -m pip install --disable-pip-version-check "cmake==3.31.10" "ninja==1.13.2"
export PATH="$(dirname "$py"):$PATH"

git init "$src"
git -C "$src" remote add origin https://github.com/llvm/llvm-project.git
git -C "$src" fetch --depth 1 origin "$PORTABLE_LLVM_COMMIT"
git -C "$src" checkout --detach FETCH_HEAD
actual_commit=$(git -C "$src" rev-parse HEAD)
[[ "$actual_commit" == "$PORTABLE_LLVM_COMMIT" ]] || {
  echo "LLVM provenance mismatch: wanted $PORTABLE_LLVM_COMMIT, got $actual_commit" >&2
  exit 1
}

cmake_args=(
  -G Ninja
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_C_COMPILER=/usr/bin/gcc
  -DCMAKE_CXX_COMPILER=/usr/bin/g++
  "-DCMAKE_INSTALL_PREFIX=$prefix"
  -DLLVM_ENABLE_PROJECTS=clang
  -DLLVM_ENABLE_RTTI:BOOL=ON
  -DLLVM_ENABLE_LIBXML2=OFF
  -DLLVM_ENABLE_ZLIB=OFF
  -DLLVM_ENABLE_ZSTD=OFF
  '-DLLVM_TARGETS_TO_BUILD=host;NVPTX;AMDGPU'
  -DLLVM_ENABLE_TERMINFO=OFF
  -DLLVM_INCLUDE_TESTS=OFF
  -DLLVM_INCLUDE_EXAMPLES=OFF
  -DLLVM_INCLUDE_BENCHMARKS=OFF
  -DLLVM_INCLUDE_DOCS=OFF
  -DCLANG_INCLUDE_TESTS=OFF
  -DCLANG_ENABLE_STATIC_ANALYZER=OFF
  -DCLANG_ENABLE_ARCMT=OFF
  -DLLVM_PARALLEL_LINK_JOBS=1
)

cmake -S "$src/llvm" -B "$build" "${cmake_args[@]}"
cmake --build "$build" --target install -- -k 0

"$prefix/bin/llvm-config" --version | grep -Fx "$PORTABLE_LLVM_VERSION"
"$prefix/bin/clang" --version | sed -n '1,4p'

# Match the upstream SDK packaging recipe: stripping is not required for ABI
# correctness, but it keeps this cached/uploaded install tree near the size of
# the archive Quadrants normally downloads instead of carrying debug sections.
find "$prefix/bin" -type f -exec file {} + \
  | grep -E 'ELF.*executable|ELF.*shared object' \
  | cut -d: -f1 \
  | xargs --no-run-if-empty "$prefix/bin/llvm-strip" --strip-unneeded
find "$prefix/lib" -type f \
  \( -name '*.so*' -o -name '*.a' \) \
  -exec "$prefix/bin/llvm-strip" --strip-unneeded {} +

# Linked ELF files carry concrete GLIBC version requirements, so measure those
# exactly. Keep the complete symbol line alongside the candidate path as well
# as the version-only list: if the gate fires, the log must identify the ELF
# and import that raised the floor rather than forcing another blind rebuild.
# Static archive members are earlier in the link pipeline: they can contain an
# unresolved `_dl_find_object` without an @GLIBC_2.35 suffix yet. Reject that
# raw symbol too; it is the object-level fingerprint of the bug that the old
# prebuilt LLVM contributed to the final Quadrants extension.
versions_file=$work/glibc-versions.txt
references_file=$work/glibc-references.txt
: > "$versions_file"
: > "$references_file"
while IFS= read -r -d '' candidate; do
  if readelf --wide --dyn-syms "$candidate" >/tmp/algan-readelf.$$ 2>/dev/null; then
    sed -nE 's/.*@GLIBC_([0-9]+\.[0-9]+).*/\1/p' /tmp/algan-readelf.$$ >> "$versions_file"
    while IFS= read -r reference; do
      printf '%s\t%s\n' "$candidate" "$reference" >> "$references_file"
    done < <(grep -E '@GLIBC_[0-9]+\.[0-9]+' /tmp/algan-readelf.$$ || true)
  fi
done < <(find "$prefix/bin" "$prefix/lib" -type f -print0)
rm -f /tmp/algan-readelf.$$

max_glibc=$(sort -Vu "$versions_file" | tail -1)
[[ -n "$max_glibc" ]] || {
  echo "readelf found no versioned GLIBC references in the LLVM install tree" >&2
  exit 1
}
newest=$(printf '%s\n' "$max_glibc" 2.34 | sort -V | tail -1)
[[ "$newest" == "2.34" ]] || {
  echo "portable LLVM requires GLIBC_$max_glibc, above the 2.34 target" >&2
  echo "Offending GLIBC_$max_glibc references:" >&2
  grep -F "@GLIBC_$max_glibc" "$references_file" >&2 || true
  echo "All GLIBC versions found:" >&2
  sort -Vu "$versions_file" >&2
  exit 1
}

archive_find_object=$work/archive-dl-find-object.txt
: > "$archive_find_object"
while IFS= read -r -d '' archive; do
  "$prefix/bin/llvm-nm" -A -u "$archive" 2>/dev/null \
    | grep -F '_dl_find_object' >> "$archive_find_object" || true
done < <(find "$prefix/lib" -type f -name '*.a' -print0)
if [[ -s "$archive_find_object" ]]; then
  echo "portable LLVM archives still contain an unresolved _dl_find_object:" >&2
  cat "$archive_find_object" >&2
  exit 1
fi

rm -f "$output_zip" "$output_zip.sha256"
(
  cd "$prefix"
  zip -qr "$output_zip" .
)
(
  cd "$(dirname "$output_zip")"
  sha256sum "$(basename "$output_zip")" > "$(basename "$output_zip").sha256"
)
archive_sha=$(sha256sum "$output_zip" | awk '{print $1}')

cat > "$provenance_file" <<EOF
Quadrants portable LLVM artifact
LLVM version: $PORTABLE_LLVM_VERSION
LLVM commit: $actual_commit
Build image: $PORTABLE_LLVM_BUILD_IMAGE
Architecture: $arch
Userspace: $glibc_line
C compiler: $($CC --version | sed -n '1p')
C++ compiler: $($CXX --version | sed -n '1p')
GCC package: $(rpm -q gcc)
G++ package: $(rpm -q gcc-c++)
Static libstdc++ package: $(rpm -q libstdc++-static)
Static libstdc++: $static_libstdcxx
CMake: $(cmake --version | sed -n '1p')
Ninja: $(ninja --version)
CMake arguments: ${cmake_args[*]}
Measured maximum GLIBC symbol in install tree/archive members: GLIBC_$max_glibc
Archive SHA-256: $archive_sha
EOF

cat "$provenance_file"
