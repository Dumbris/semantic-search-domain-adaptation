#!/usr/bin/env bash

##################################################################
# Purpose: Create virtualenv dir, setup python packages
##################################################################

set -ex

###Global vars declaration
readonly HOMEDIR="${PWD}"

function extract_info()
{
    local dir="$1"
    #cd $dir
    local config=$(yq -c . "$dir/.hydra/config.yaml")
    local overrides=$(yq -c . "$dir/.hydra/overrides.yaml")
    if [ -f "$dir/output.json" ];then
        jq -c --argjson config "${config}" \
            --argjson overrides "${overrides}" \
            --arg dir "${dir}" \
            '. + {config: $config, overrides: $overrides, dir: $dir}' \
            "$dir/output.json"
    fi
}

dirlist=$(find "${HOMEDIR}/outputs" -mindepth 2 -maxdepth 2 -type d)

for dir in $dirlist
do
  (
      extract_info $dir
  )
done > output.json