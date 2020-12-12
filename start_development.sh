#!/usr/bin/env bash

##################################################################
# Purpose: Create virtualenv dir, setup python packages
##################################################################

set -ex

###Global vars declaration
readonly HOMEDIR="${PWD}"
readonly VIRT_ENV_DIR=${VIRT_ENV_DIR:-".pyenv"}

if [ ! -d "${VIRT_ENV_DIR}" ];then
    python3 -m virtualenv -p python3 "${VIRT_ENV_DIR}"
else
    echo "Found ${VIRT_ENV_DIR}, skip virtualenv creation."
fi

echo "Install packages for develop..."

for pkg_dir in "search_eval" "experiment"
do
    cd "${HOMEDIR}/src/${pkg_dir}"
    "${HOMEDIR}/${VIRT_ENV_DIR}/bin/python" -m pip install -e .
    cd "${HOMEDIR}"
done

#"${HOMEDIR}/${VIRT_ENV_DIR}/bin/python" -m spacy download en_core_web_sm