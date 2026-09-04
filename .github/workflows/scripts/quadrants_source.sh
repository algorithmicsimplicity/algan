#!/usr/bin/env bash
# =============================================================================
# The one place `quadrants_build.yaml` gets its Quadrants source tree from.
#
# Clones `Genesis-Embodied-AI/quadrants` at `$QUADRANTS_REF` into
# `$QUADRANTS_SRC` (default `quadrants-src/`) and, unless `APPLY_PATCHES` is
# `false`, applies every `quadrants_patches/[0-9]*.patch` in numeric order with
# a strict `git apply` -- no fuzz, no 3-way merge -- so a patch that has
# drifted from the tag fails HERE, loudly, rather than half applying and
# producing a wheel whose behaviour nobody can account for. It refuses to
# "apply" an empty directory for the same reason: a wheel that looks patched
# and is not is worse than no wheel.
#
# Every leg of the workflow runs this (bash on macOS, in the manylinux
# container, and under Git Bash on Windows) and so does the `plan` job, which
# runs it without submodules as a fast pre-check. Keeping it in one file is
# what guarantees the three wheels of one run were cut from the same tree.
#
# Writes `PATCHED=0|1` and `QUADRANTS_SHA=<commit>` to `$GITHUB_ENV` (and to
# `$GITHUB_OUTPUT` when set), and a Markdown `$PATCH_REPORT` (default
# `patch-report.md`) naming the base commit, the patches and their diffstat --
# the release job pastes that into the release body.
#
# Knobs, all environment variables:
#   QUADRANTS_REF     tag/branch/sha to clone (required)
#   APPLY_PATCHES     "true" (default) or "false" for a stock build
#   WITH_SUBMODULES   "true" (default) or "false"; the patches touch no
#                     submodule, so the pre-check skips ~160 MB of them
#   QUADRANTS_SRC     destination directory (default quadrants-src)
#   PATCH_DIR         where the patches are (default quadrants_patches)
#   PATCH_REPORT      report path (default patch-report.md)
#
# Paths are relative to the current directory on purpose: on Windows
# `$GITHUB_WORKSPACE` is a backslashed `D:\a\...` that Git Bash's globbing does
# not take kindly to, and every step that calls this runs from the workspace.
# =============================================================================
set -euo pipefail

ref="${QUADRANTS_REF:?QUADRANTS_REF is required}"
apply="${APPLY_PATCHES:-true}"
submodules="${WITH_SUBMODULES:-true}"
repo="${QUADRANTS_REPO:-https://github.com/Genesis-Embodied-AI/quadrants.git}"
dest="${QUADRANTS_SRC:-quadrants-src}"
patch_dir="${PATCH_DIR:-quadrants_patches}"
report="${PATCH_REPORT:-patch-report.md}"

emit() {
  # `$1=$2` to GITHUB_ENV and GITHUB_OUTPUT, when the workflow provides them.
  [ -n "${GITHUB_ENV:-}" ] && echo "$1=$2" >> "$GITHUB_ENV"
  [ -n "${GITHUB_OUTPUT:-}" ] && echo "$1=$2" >> "$GITHUB_OUTPUT"
  return 0
}

echo "--- clone $repo @ $ref -> $dest (submodules: $submodules)"
rm -rf "$dest"
# `core.autocrlf=false` is for the Windows leg: the patches are LF, Quadrants
# ships no .gitattributes, and a checkout converted to CRLF would make a strict
# `git apply` fail on every hunk for a reason that has nothing to do with the
# patch. A no-op everywhere else.
clone_args=(-c core.autocrlf=false --depth 1 --branch "$ref")
if [ "$submodules" = "true" ]; then
  clone_args+=(--recurse-submodules --shallow-submodules)
fi
git clone "${clone_args[@]}" "$repo" "$dest"
sha="$(git -C "$dest" rev-parse HEAD)"
git -C "$dest" log -1 --format='%H %ci %s'
emit QUADRANTS_SHA "$sha"

# setuptools_scm derives the wheel version from `git describe`; Quadrants' own
# CI clones with fetch-depth 0 for exactly this. A `--depth 1` clone of a tag
# still describes, but check rather than ship a wheel called 0.1
# (`scripts/gate/quadrants_macos_build.sh` does the same).
if git -C "$dest" describe --tags >/dev/null 2>&1; then
  echo "git describe --tags: $(git -C "$dest" describe --tags)"
else
  echo "WARNING: git describe failed on the shallow clone; fetching the tag explicitly"
  git -C "$dest" fetch --depth 1 origin "refs/tags/$ref:refs/tags/$ref" || true
  echo "git describe --tags: $(git -C "$dest" describe --tags 2>&1 || true)"
fi

{
  echo "Base: \`$ref\` (\`$sha\`)"
  echo
} > "$report"

if [ "$apply" != "true" ]; then
  echo "--- APPLY_PATCHES=$apply: stock $ref, nothing applied"
  echo "Patches: **none** (stock build)" >> "$report"
  emit PATCHED 0
  exit 0
fi

# A newline list rather than a bash array: macOS's /bin/bash is 3.2, where
# `"${arr[@]}"` on an empty array is an unbound-variable error under `set -u`.
patch_list="$(ls -1 "$patch_dir"/[0-9]*.patch 2>/dev/null || true)"
if [ -z "$patch_list" ]; then
  echo "no patches in $patch_dir/ -- refusing to build a wheel that would look"
  echo "patched and not be. Pass apply_patches=false for a stock build."
  exit 1
fi

echo "--- applying, in order:"
printf '  %s\n' $patch_list
echo "Patches, in the order applied:" >> "$report"
echo >> "$report"
while IFS= read -r patch; do
  [ -n "$patch" ] || continue
  name="$(basename "$patch")"
  echo "--- git apply $name"
  # Strict: no fuzz, no 3-way. A failure here means the patch has drifted from
  # the tag; fix the patch, do not loosen the apply (quadrants_patches/README.md).
  git -C "$dest" apply --verbose "../$patch"
  {
    echo "* \`$name\`"
    echo
    echo '```'
    git -C "$dest" apply --numstat "../$patch" 2>/dev/null \
      | awk '{printf "  +%s/-%s  %s\n", $1, $2, $3}' || true
    echo '```'
  } >> "$report"
done <<EOF
$patch_list
EOF

echo "--- resulting diffstat"
git -C "$dest" -c core.pager=cat diff --stat | tee -a /dev/stderr | {
  echo
  echo "Combined \`git diff --stat\` against \`$ref\`:"
  echo
  echo '```'
  cat
  echo '```'
} >> "$report"
emit PATCHED 1
