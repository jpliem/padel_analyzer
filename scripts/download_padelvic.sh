#!/usr/bin/env bash
#
# Download + organize the PADELVIC dataset (multi-camera padel match + mocap GT).
# Source: https://github.com/UPC-ViRVIG/PadelVic  (CC-BY, research use only)
#
# Usage:
#   ./scripts/download_padelvic.sh              # download everything
#   ./scripts/download_padelvic.sh cameras      # only real-camera videos
#   ./scripts/download_padelvic.sh synthetic    # only synthetic clips + CSV GT
#   ./scripts/download_padelvic.sh mocap derived
#
# Idempotent: skips files already fully downloaded, resumes partial ones (curl -C -).

set -euo pipefail

# Resolve repo root regardless of where script is called from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEST="$ROOT/data/datasets/padelvic"

# Each entry: "subdir|filename|url"  (Dropbox dl=1 = direct download)
ENTRIES=(
  # --- real cameras (multi-cam) ---
  "cameras|panasonic_final.mp4|https://www.dropbox.com/scl/fi/hrzytt71wc92zaq8oq0fh/panasonic_final.mp4?rlkey=vhj3rp9xbmxhemmx24bkfnu55&dl=1"
  "cameras|gopro.mp4|https://www.dropbox.com/scl/fi/pnzk74zghdxtybd9zb62a/gopro.mp4?rlkey=k3agl6p3ntigncx0dbu0jwu2e&dl=1"
  "cameras|samsung.mp4|https://www.dropbox.com/s/hn9ub1qmjurcf5b/samsung.mp4?dl=1"
  "cameras|iphone.mp4|https://www.dropbox.com/scl/fi/6n7ahs5az28sos3pws7dg/iphone.mp4?rlkey=vn6hfwoz84k60w5ug1nyo2uiz&dl=1"

  # --- ground truth ---
  "mocap|bvh.zip|https://www.dropbox.com/scl/fi/sl5hdlr6bal69v2acdbei/bvh.zip?rlkey=13ttllqg65ayoujhozpy75yzd&dl=1"

  # --- synthetic Xsens clips (paired with projected positional GT; not ball labels) ---
  "synthetic|1630-16300.mp4|https://www.dropbox.com/scl/fi/clsjqiwvby6w8byv9g0ap/1630-16300.mp4?rlkey=aqecxe72tasojmoqi721l6mdv&dl=1"
  "synthetic|001-2200-3960.mkv|https://www.dropbox.com/scl/fi/lv1itp4vpnwywudwe6t6b/001-2200-3960.mkv?rlkey=fm1544hw7gwzzcw8ly9ulfyku&dl=1"
  "synthetic|001-1250-17462.mkv|https://www.dropbox.com/scl/fi/hhw2ag7nfligagrgu01nm/001-1250-17462.mkv?rlkey=9pcmp205da9ifanqxr1uc99ox&dl=1"
  "synthetic|001-1250-17462.csv|https://www.dropbox.com/scl/fi/84pppfrg8h263q91wr9vv/001-1250-17462.csv?rlkey=48cs2qs15cdq7pt28qzxfkfjt&dl=1"
  "synthetic|002-0275-11541.mkv|https://www.dropbox.com/scl/fi/1v5gi1sz331f6vwwaqer2/002-0275-11541.mkv?rlkey=fva50ssaofd5c2v0xpdxxn4va&dl=1"
  "synthetic|002-0275-11541.csv|https://www.dropbox.com/scl/fi/1rea3wuis3cv12pf6fwts/002-0275-11541.csv?rlkey=wk9yvlikw2jzw19if6ckc3bzw&dl=1"
  "synthetic|002B-0275-11541.mkv|https://www.dropbox.com/scl/fi/tb94z4sm8s2oeufwtb9fy/002B-0275-11541.mkv?rlkey=91ddt8bxhl41gfj33ynw3e52x&dl=1"
  "synthetic|002B-0275-11541.csv|https://www.dropbox.com/scl/fi/qku3xtfekqia0ar1mlgdv/002B-0275-11541.csv?rlkey=c0bgzjtwf0zcdspqrkbtbnbzo&dl=1"

  # --- derived / reference outputs ---
  "derived|game2_vic_panasonic.mp4|https://www.dropbox.com/scl/fi/oelg2ildoat14ttf4asv0/game2_vic_panasonic.mp4?rlkey=snv6msx10usvaq6nnl47v4cij&dl=1"
  "derived|PadelVic_Panasonic_labeling.xlsx|https://www.dropbox.com/scl/fi/g8usouvynhxb8jkoumqop/PadelVic_Panasonic_labeling-all-shared.xlsx?rlkey=5zs59cg3q4doiio75bak5255h&dl=1"
  "derived|Panasonic_Poses.zip|https://www.dropbox.com/scl/fi/tkbeu8ndxl4axy9iijcwt/Panasonic_Poses.zip?rlkey=q9c60m5ln85nrihum1tlq4q53&dl=1"
)

# Optional category filter from CLI args.
WANT=("$@")
want_category() {
  [ ${#WANT[@]} -eq 0 ] && return 0
  local c="$1"; for w in "${WANT[@]}"; do [ "$w" = "$c" ] && return 0; done
  return 1
}

echo "PADELVIC -> $DEST"
echo

downloaded=0 skipped=0
for entry in "${ENTRIES[@]}"; do
  IFS='|' read -r subdir fname url <<< "$entry"
  want_category "$subdir" || continue

  dir="$DEST/$subdir"
  out="$dir/$fname"
  part="$out.part"
  mkdir -p "$dir"

  if [ -f "$out" ] && [ -s "$out" ]; then
    echo "skip  $subdir/$fname (exists)"
    skipped=$((skipped+1))
    continue
  fi

  echo "get   $subdir/$fname"
  # -L follow redirects, -C - resume, --fail error on 4xx/5xx, retry on transient errors.
  curl -L -C - --fail --retry 3 --retry-delay 2 -o "$part" "$url" || {
    echo "FAILED $subdir/$fname — leaving partial for resume" >&2
    continue
  }
  mv "$part" "$out"
  downloaded=$((downloaded+1))
done

echo
echo "done. downloaded=$downloaded skipped=$skipped"
echo "tree:"
find "$DEST" -maxdepth 2 -type f -exec ls -lh {} \; 2>/dev/null | awk '{print $5"\t"$NF}' || true
