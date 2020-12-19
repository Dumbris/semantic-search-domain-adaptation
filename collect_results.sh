#!/usr/bin/env bash

##################################################################
# Purpose: Create virtualenv dir, setup python packages
##################################################################

set -ex

###Global vars declaration
readonly HOMEDIR="${PWD}"
readonly REPORTSDIR="${1:=outputs}"
readonly DATE=$(date '+%Y_%m_%d__%H_%M_%S')

[ ! -d "$REPORTSDIR" ] && echo "No reports dir" && exit 1

function extract_info()
{
    local dir="$1"
    local config
    local overrides
    if [[ -d "$dir" && -f "$dir/output.json" && -d "$dir/.hydra" ]];then
        config=$(yq -c . "$dir/.hydra/config.yaml")
        overrides=$(yq -c . "$dir/.hydra/overrides.yaml")
        jq -c --argjson config "${config}" \
            --argjson overrides "${overrides}" \
            --arg dir "${dir}" \
            '. + {config: $config, overrides: $overrides, dir: $dir}' \
            "$dir/output.json"
    fi
}

dirlist=$(find "${REPORTSDIR}" -mindepth 1 -maxdepth 2 -type d)

for dir in $dirlist
do
  (
      extract_info $dir
  )
done > "report_${DATE}.jsonl"