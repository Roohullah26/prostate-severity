#!/usr/bin/env bash
# Simple example curl request to the FastAPI server (assumes server running on port 8000)
# Example: ./example_request.sh ./data/example.jpg ./models/prototype_toy.pth

IMG=${1:-./data/example.jpg}
STATE=${2:-./models/prototype_toy.pth}
HOST=${3:-127.0.0.1:8000}
API_KEY=${4:-}

if [ ! -f "$IMG" ]; then
  echo "image not found: $IMG"
  exit 1
fi

if [ -n "$API_KEY" ]; then
  curl -s -X POST "http://$HOST/predict?state_dict=$(python -c "import sys,urllib.parse; print(urllib.parse.quote(sys.argv[1]))" "$STATE")&in_channels=3" -H "X-API-Key: $API_KEY" -F "file=@$IMG" | jq
else
  curl -s -X POST "http://$HOST/predict?state_dict=$(python -c "import sys,urllib.parse; print(urllib.parse.quote(sys.argv[1]))" "$STATE")&in_channels=3" -F "file=@$IMG" | jq
fi
