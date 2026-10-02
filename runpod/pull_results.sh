#!/usr/bin/env bash
# 팟 results/ 를 로컬 results/ 로 회수 (같은 이름은 팟 쪽이 덮어씀, 로컬에만 있는 파일은 유지).
#   bash runpod/pull_results.sh            # core_pretrained exp3 core_resnet
#   bash runpod/pull_results.sh exp3       # 일부만
set -u
HOST=root@<pod-ip>; PORT=22153; KEY=/c/Users/KDI/.ssh/runpod_key
REMOTE=/workspace/subtractive-intelligence/results
LOCAL="$(cd "$(dirname "$0")/.." && pwd)/results"
dirs=("$@"); [ ${#dirs[@]} -eq 0 ] && dirs=(core_pretrained exp3 core_resnet)
for d in "${dirs[@]}"; do
  echo "== $d"
  ssh -o StrictHostKeyChecking=no -o BatchMode=yes -i "$KEY" -p "$PORT" "$HOST" "cd $REMOTE && tar czf - $d" | tar xzf - -C "$LOCAL"
done
mkdir -p "$LOCAL/_pod_logs"
scp -q -o StrictHostKeyChecking=no -i "$KEY" -P "$PORT" "$HOST:$REMOTE/_pod_*" "$LOCAL/_pod_logs/" 2>/dev/null || true
echo "pulled: $(find "$LOCAL/core_pretrained" -name 'seed*.json' 2>/dev/null | wc -l) core_pretrained, $(find "$LOCAL/exp3" -name 'seed*.json' 2>/dev/null | wc -l) exp3, $(find "$LOCAL/core_resnet" -name 'seed*.json' 2>/dev/null | wc -l) core_resnet json files"
