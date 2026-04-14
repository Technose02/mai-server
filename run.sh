#!/bin/bash
source .env
RUST_LOG=reqwest=trace \
MAISERVER_MAX_LOGLEVEL=$MAISERVER_MAX_LOGLEVEL \
cargo run --release -p gw-server -- \
  --api-key $MAI_SERVER_APIKEY \
  --log-request-info
