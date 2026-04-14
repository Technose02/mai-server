#!/bin/bash
source .env
#RUST_LOG=reqwest=trace \
cargo run --release -p gw-server -- \
  --api-key $MAI_SERVER_APIKEY \
  --log-request-info
