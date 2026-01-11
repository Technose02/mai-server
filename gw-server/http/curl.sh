#!/bin/bash

working_dir="$(dirname $0)"

source $working_dir/env.secret

ENDPOINT_URL="https://${MAI_SERVER_HOST}:${MAI_SERVER_PORT}/admin/llamacpp"
AUTH_HEADER="Authorization: Bearer ${MAI_SERVER_APIKEY}"

function start-process()
{
  curl -X PUT ${ENDPOINT_URL} -H "${AUTH_HEADER}" \
  --json @$working_dir/$1
}

function get-process-state()
{
  curl -X GET $ENDPOINT_URL -H "${AUTH_HEADER}"
}

function stop-process()
{
  curl -X DELETE $ENDPOINT_URL -H "${AUTH_HEADER}"
}

#start-process devstral-small-2-24B-instruct-2512.json
#start-process gemma-3-12b-it-qat-Q8_0.json
#start-process gemma-3-27b-it-qat-Q8_0.json
#start-process glm-4.6v-flash.json
#start-process gpt-oss-120b-Q8_0.json
#start-process granite-4.0-h-small.json
#start-process nemotron-3-nano-30b-a3b.json
#start-process phi-4-reasoning-plus.json
#start-process qwen3-vl-30b-a3b-instruct.json
#start-process qwen3-vl-30b-a3b-thinking.json

stop-process
get-process-state
