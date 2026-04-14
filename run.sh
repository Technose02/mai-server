#!/bin/bash
source .env
RUST_LOG=reqwest=info \
MAISERVER_LOG=info \
cargo run --release -p gw-server -- \
  --api-key $MAI_SERVER_APIKEY \
  --log-request-info
