#!/bin/bash

# Options can also be passed on the command line.
# These options are blind-passed to the CLI.
# Ex: ./run.sh -D log4j.threshold=DEBUG

readonly PSL_VERSION='2.3.0-SNAPSHOT'
readonly JAR_PATH="./psl-cli-${PSL_VERSION}.jar"
readonly FETCH_DATA_SCRIPT='../data/fetchData.sh'
readonly BASE_NAME='movielens'
readonly POSTGRES_DB='psl'
readonly STANDARD_PSL_OPTIONS="--postgres ${POSTGRES_DB}"

#readonly ADDITIONAL_PSL_OPTIONS='--int-ids'
readonly ADDITIONAL_PSL_OPTIONS=' --postgres psl -D admmreasoner.initialconsensusvalue=ZERO -D log4j.threshold=TRACE'
readonly ADDITIONAL_LEARN_OPTIONS='--learn'
readonly ADDITIONAL_EVAL_OPTIONS='--infer --eval org.linqs.psl.evaluation.statistics.ContinuousEvaluator '

readonly JAVA_MEM_GB=24

function main() {
   trap exit SIGINT
   local model=$1

   shift 1

   echo "${model}"

   # Get the data
   # getData

   # Make sure we can run PSL.
   check_requirements
   fetch_psl

   # Run PSL
   # runWeightLearning "$@"
   runEvaluation "$model" "$@"
}

function getData() {
   pushd . > /dev/null

   cd "$(dirname $FETCH_DATA_SCRIPT)"
   bash "$(basename $FETCH_DATA_SCRIPT)"

   popd > /dev/null
}

function runWeightLearning() {
   echo "Running PSL Weight Learning"

   java -XmxG -XmsG -jar "${JAR_PATH}" --model "${BASE_NAME}.psl" --data "${BASE_NAME}-learned.data" ${ADDITIONAL_LEARN_OPTIONS} ${ADDITIONAL_PSL_OPTIONS} "$@"
   if [[ "$?" -ne 0 ]]; then
      echo 'ERROR: Failed to run weight learning'
      exit 60
   fi
}

function runEvaluation() {
   local model=$1
   local extra_options=$2
   echo "Running PSL Inference"

   java -Xmx${JAVA_MEM_GB}G -Xms${JAVA_MEM_GB}G -jar "${JAR_PATH}" --model "${BASE_NAME}_${model}-learned.psl" --data "${BASE_NAME}-eval.data" --output inferred-predicates ${ADDITIONAL_EVAL_OPTIONS} ${ADDITIONAL_PSL_OPTIONS} ${extra_options}
   if [[ "$?" -ne 0 ]]; then
      echo 'ERROR: Failed to run infernce'
      exit 70
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

   echo "Downloading ${name} file located at: '${url}'."
   `get_fetch_command` "${path}" "${url}"
   if [[ "$?" -ne 0 ]]; then
      echo "ERROR: Failed to download ${name} file"
      exit 30
   fi
}

# Fetch the jar from a remote or local location and put it in this directory.
# Snapshots are fetched from the local maven repo and other builds are fetched remotely.
function fetch_psl() {
   if [[ $PSL_VERSION == *'SNAPSHOT'* ]]; then
      local snapshotJARPath="$HOME/.m2/repository/org/linqs/psl-cli/${PSL_VERSION}/psl-cli-${PSL_VERSION}.jar"
      cp "${snapshotJARPath}" "${JAR_PATH}"
   else
      local remoteJARURL="https://repo1.maven.org/maven2/org/linqs/psl-cli/${PSL_VERSION}/psl-cli-${PSL_VERSION}.jar"
      fetch_file "${remoteJARURL}" "${JAR_PATH}" 'psl-jar'
   fi
}

main "$@"