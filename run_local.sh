#!/bin/bash
source .env

cargo run --release -p gw-server -- \
  --port 11434 \
  --override-host-ip 127.0.0.1 \
  --no-https \
  --api-key $MAI_SERVER_NO_APIKEY \
  --log-request-info
