#!/usr/bin/env bash
# Upload a pre-rendered client .webm from your laptop to homelab.
# Usage: ./docker/upload_clip.sh showcase0 /tmp/showcase0.webm skg@homelab
set -euo pipefail

NAME="${1:?name e.g. showcase0}"
FILE="${2:?path to .webm}"
HOST="${3:-skg@homelab}"

scp "$FILE" "$HOST:/tmp/${NAME}.webm"
ssh "$HOST" "sudo mkdir -p /var/lib/openfront-eval/data/clips && \
  sudo cp /tmp/${NAME}.webm /var/lib/openfront-eval/data/clips/${NAME}.webm && \
  docker exec openfront-eval python3 -c \"
import json
from pathlib import Path
p = Path('/data/state.json')
s = json.loads(p.read_text()) if p.exists() else {}
urls = s.setdefault('hero_clips', [])
u = '/archive/clips/${NAME}.webm'
if u not in urls: urls.append(u)
p.write_text(json.dumps(s, indent=2) + '\n')
print('hero_clips', urls)
\""

echo "uploaded ${NAME}.webm - refresh https://openfrontai.skg.gg/"
