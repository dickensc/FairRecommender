#!/usr/bin/env bash

readonly BASE_DIR=$(realpath "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"/..)

readonly RAW_DATA_URL='https://files.grouplens.org/datasets/movielens/ml-1m.zip'
readonly DATA_FILE="${BASE_DIR}/psl-datasets/movielens/data/ml-1m.zip"
readonly DATA_DIR="${BASE_DIR}/psl-datasets/movielens/data/ml-1m"

readonly PSL_DATA_CONSTRUCTOR="${BASE_DIR}/data_construction/write_predicates.py"

function fetch_raw_data() {
   fetch_file "${RAW_DATA_URL}" "${DATA_FILE}" 'data'
   extract_zip "${DATA_FILE}" "${DATA_DIR}" 'data'
}

function check_requirements() {
   local hasWget
   local hasCurl

   type wget > /dev/null 2> /dev/null
   hasWget=$?

   type curl > /dev/null 2> /dev/null
   hasCurl=$?

   if [[ "${hasWget}" -ne 0 ]] && [[ "${hasCurl}" -ne 0 ]]; then
      echo 'ERROR: wget or curl required to download dataset'
      exit 10
   fi

   type tar > /dev/null 2> /dev/null
   if [[ "$?" -ne 0 ]]; then
      echo 'ERROR: tar required to extract dataset'
      exit 11
   fi
}

function get_fetch_command() {
   type curl > /dev/null 2> /dev/null
   if [[ "$?" -eq 0 ]]; then
      echo "curl -o"
      return
   fi

   type wget > /dev/null 2> /dev/null
   if [[ "$?" -eq 0 ]]; then
      echo "wget -O"
      return
   fi

   echo 'ERROR: wget or curl not found'
   exit 20
}

function fetch_file() {
   local url=$1
   local path=$2
   local name=$3

   if [[ -e "${path}" ]]; then
      echo "${name} file found cached, skipping download."
      return
   fi

   echo "Downloading ${name} file with command: $FETCH_COMMAND"
   `get_fetch_command` "${path}" "${url}"
   if [[ "$?" -ne 0 ]]; then
      echo "ERROR: Failed to download ${name} file"
      exit 30
   fi
}

# Works for tarballs too.
function extract_zip() {
   local path=$1
   local expectedDir=$2
   local name=$3

   if [[ -e "${expectedDir}" ]]; then
      echo "Extracted ${name} zip found cached, skipping extract."
      return
   fi

   echo "Extracting the ${name} zip"
   unzip "${path}" -d "$(dirname "${expectedDir}")"
   if [[ "$?" -ne 0 ]]; then
      echo "ERROR: Failed to extract ${name} zip"
      exit 40
   fi
}

function main() {
   trap exit SIGINT

   check_requirements

   fetch_raw_data

   # Construct PSL predicates.
   python3 "${PSL_DATA_CONSTRUCTOR}"

   exit 0
}

[[ "${BASH_SOURCE[0]}" == "${0}" ]] && main "$@"