#!/usr/bin/env bash
# Populate models/hailo8l/ with the HEFs used by the Pi 5 + Hailo-8L pipeline.
#
# HEFs already installed by hailo-apps / TAPPAS under /usr/local/hailo are
# symlinked; the rest are downloaded from the Hailo Model Zoo. Every file is
# checked with `hailortcli parse-hef` afterwards.
#
# Usage: scripts/fetch_hailo_models.sh [--with-optional]
#   --with-optional  also fetch arcface_r50 (higher-accuracy face embedder)
#                    and yolov8m (larger object detector)
set -euo pipefail

ARCH="hailo8l"
ZOO_VERSION="${HAILO_MODEL_ZOO_VERSION:-v2.17.0}"
ZOO_URL="https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/${ZOO_VERSION}/${ARCH}"
SYSTEM_DIR="${HAILO_SYSTEM_MODELS:-/usr/local/hailo/resources/models/${ARCH}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${SCRIPT_DIR}/../models/${ARCH}"

REQUIRED=(yolov8s_pose scrfd_2.5g arcface_mobilefacenet repvgg_a0_person_reid_512 yolov8s)
OPTIONAL=(arcface_r50 yolov8m)

models=("${REQUIRED[@]}")
if [[ "${1:-}" == "--with-optional" ]]; then
  models+=("${OPTIONAL[@]}")
fi

mkdir -p "${DEST}"
cd "${DEST}"

for name in "${models[@]}"; do
  file="${name}.hef"
  if [[ -s "${file}" ]]; then
    echo "[ok]       ${file} (already present)"
  elif [[ -s "${SYSTEM_DIR}/${file}" ]]; then
    ln -sf "${SYSTEM_DIR}/${file}" "${file}"
    echo "[linked]   ${file} -> ${SYSTEM_DIR}/${file}"
  else
    echo "[download] ${file} from Model Zoo ${ZOO_VERSION}"
    curl -fL --retry 3 -o "${file}.part" "${ZOO_URL}/${file}"
    mv "${file}.part" "${file}"
  fi
done

if command -v hailortcli >/dev/null 2>&1; then
  echo
  for name in "${models[@]}"; do
    arch_line="$(hailortcli parse-hef "${name}.hef" 2>&1 | grep -m1 'Architecture' || true)"
    if [[ "${arch_line}" != *HAILO8L* ]]; then
      echo "[error] ${name}.hef did not parse as a HAILO8L HEF: ${arch_line}" >&2
      exit 1
    fi
    echo "[verified] ${name}.hef  (${arch_line##*: })"
  done
fi

echo
echo "Models ready in $(cd "${DEST}" && pwd)"
echo "Rebuild (colcon build) so they are installed into the package share directory."
