#!/bin/bash
source .env

cargo run --release -p gw-server -- \
  --api-key $MAI_SERVER_APIKEY \
  --log-request-info
