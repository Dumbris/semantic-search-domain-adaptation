#!/usr/bin/env bash

##################################################################
# Purpose: Start experiment
##################################################################

set -ex

###Global vars declaration
readonly HOMEDIR="${PWD}"
readonly VIRT_ENV_DIR=${VIRT_ENV_DIR:-".pyenv"}
readonly REPORTSDIR="${1:=outputs}"
readonly EXPRIMENT_DIR="${REPORTSDIR}/eval_01"

[ ! -d "${VIRT_ENV_DIR}" ] &&  echo "Not found ${VIRT_ENV_DIR}" && exit 1

source "${VIRT_ENV_DIR}/bin/activate"

eval_reranker -m reranker.base_model=distilroberta-base,roberta-base hydra.sweep.dir="${EXPRIMENT_DIR}"
./collect_results.sh "${EXPRIMENT_DIR}"
