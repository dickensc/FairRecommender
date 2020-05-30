#!/bin/bash

readonly THIS_DIR=$(realpath "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )")
readonly BASE_DIR="${THIS_DIR}/../.."

# Tuffy path variables
readonly PSL_TO_TUFFY_HELPER_PATH="${BASE_DIR}/psl_to_tuffy_examples"
readonly TUFFY_EXAMPLES="${BASE_DIR}/tuffy-examples"
readonly TUFFY_URL="http://i.stanford.edu/hazy/tuffy/download/tuffy-0.4-july2014.zip"
readonly TUFFY_BIN="${BASE_DIR}/tuffy-0.3-jun2014"
readonly TUFFY_ZIP="${BASE_DIR}/tuffy-0.4-july2014.zip"
readonly TUFFY_RESOURCES_DIR="${BASE_DIR}/tuffy_resources"

function main() {
  trap exit SIGINT

  if [[ $# -eq 0 ]]; then
    echo "USAGE: $0 <example dir> ..."
    echo "USAGE: Example Directories can be among: ${SUPPORTED_EXAMPLES}"
    exit 1
  fi

  echo "INFO: Working on setting up tuffy ${experiment}"

  # Make sure we can run psl_to_tuffy_examples.
  check_requirements

  # fetch psl_to_tuffy_examples
  tuffy_load

  # First begin by creating a postgreSQL database and user for psl_to_tuffy_examples
  tuffy_create_postgres_db

  # Create Tuffy Experiment Directory if it does not already exist
  [ -d "${TUFFY_EXAMPLES}" ] || mkdir -p "${TUFFY_EXAMPLES}"

  # Verify psl_to_tuffy_examples directory exists and contains the helpers for
  # each of the models we are running
  for dataset_path in "$@"; do
    experiment=$(basename "${dataset_path}")
    check_psl_to_tuffy_example "${PSL_TO_TUFFY_HELPER_PATH}/${experiment}"
  done
}

function tuffy_create_postgres_db() {
  echo "INFO: Creating tuffy postgres user and db..."
  psql postgres -tAc "SELECT 1 FROM pg_roles WHERE rolname='tuffy'" | grep -q 1 || createuser -s tuffy
  psql postgres -lqt | cut -d \| -f 1 | grep -qw tuffy || createdb tuffy
}

function tuffy_load() {
   echo "INFO: Fetching Tuffy..."
   if [ -f "${TUFFY_RESOURCES_DIR}/tuffy.jar" ] ; then
      echo "Jar exists, skipping request"
      return
   fi

   curl -O ${TUFFY_URL}
   unzip ${TUFFY_ZIP}
   mv ${TUFFY_BIN}/tuffy.jar ${TUFFY_RESOURCES_DIR}/tuffy.jar
   rm -r ${TUFFY_BIN}
   rm ${TUFFY_ZIP}
}

function check_psl_to_tuffy_example() {
   local path=$1

   if [ ! -d "$path" ] ; then
       echo "ERROR: missing $path"
       exit 1
   fi
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

   type java > /dev/null 2> /dev/null
   if [[ "$?" -ne 0 ]]; then
      echo 'ERROR: java required to run project'
      exit 13
   fi

   type psql > /dev/null 2> /dev/null
   if [[ "$?" -ne 0 ]]; then
      echo 'ERROR: postgres required to run project'
      exit 13
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


main "$@"
