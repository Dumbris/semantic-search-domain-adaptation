#!/usr/bin/env bash

##################################################################
# Purpose: Start experiment
##################################################################

set -ex

###Global vars declaration
readonly HOMEDIR="${PWD}"
readonly VIRT_ENV_DIR=${VIRT_ENV_DIR:-".pyenv"}
readonly REPORTSDIR="${1:=outputs}"
readonly EXPRIMENT_DIR="${REPORTSDIR}/eval_encoder"

[ ! -d "${VIRT_ENV_DIR}" ] &&  echo "Not found ${VIRT_ENV_DIR}" && exit 1

source "${VIRT_ENV_DIR}/bin/activate"

#eval_encoder -m models.senttrans.base_model=distilroberta-base-msmarco-v2,roberta-base-nli-stsb-mean-tokens,distilbert-base-nli-stsb-quora-ranking hydra.sweep.dir="${EXPRIMENT_DIR}"
eval_encoder -m models.senttrans.base_model=roberta-base-nli-stsb-mean-tokens,distilbert-base-nli-stsb-quora-ranking hydra.sweep.dir="${EXPRIMENT_DIR}"
./collect_results.sh "${EXPRIMENT_DIR}"
