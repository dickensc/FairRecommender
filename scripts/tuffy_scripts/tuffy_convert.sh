#!/bin/bash

readonly THIS_DIR=$(realpath "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )")
readonly BASE_DIR="${THIS_DIR}/../.."

readonly PSL_TO_TUFFY_HELPER_PATH="${BASE_DIR}/psl_to_tuffy_examples"
readonly TUFFY_EXAMPLES_PATH="${BASE_DIR}/tuffy-datasets"
readonly PSL_EXAMPLES_PATH="${BASE_DIR}/psl-datasets"

function main() {
   trap exit SIGINT

   if [[ $# -eq 0 ]]; then
      echo "USAGE: $0 <example dir> ..."
      echo "USAGE: Example Directories can be among: ${SUPPORTED_EXAMPLES}"
      exit 1
   fi

   echo "$@"

   for dataset in "$@"; do
      dataset_path="${BASE_DIR}/${dataset}"
      experiment=$(basename "${dataset_path}")

      if [ ! -d "${TUFFY_EXAMPLES_PATH}/${experiment}" ]; then
        echo "INFO: Converting data for ${experiment}"
        # make the example directory
        mkdir -p "${dataset_path}"
        mkdir -p "${dataset_path}/data"
        mkdir -p "${dataset_path}/cli"
        copy_tuffy_model "${TUFFY_EXAMPLES_PATH}/${experiment}" "${PSL_TO_TUFFY_HELPER_PATH}/${experiment}"
        convert_data_tuffy "$PSL_TO_TUFFY_HELPER_PATH" "$TUFFY_EXAMPLES_PATH" "$PSL_EXAMPLES_PATH" "$experiment"
      else
        echo "INFO: Data for ${experiment} has already been converted from PSL to Tuffy. Skipping"
      fi
   done
}

function copy_tuffy_model() {
   local example_path=$1
   local model_path=$2
   local experiment
   experiment=$(basename "${example_path}")

   # copy the mln model to the example directory
   # We should have verified this exists in the init script
   cp "${model_path}/prog.mln" "${example_path}/cli/prog.mln"
}

function convert_data_tuffy() {
   local psl_to_tuffy_helper_path=$1
   local tuffy_experiment_path=$2
   local psl_experiment_path=$3
   local experiment=$4

   pushd . > /dev/null

     cd ${BASE_DIR}/scripts || exit 1
     python3 "${THIS_DIR}"/prepare_tuffy.py "${psl_to_tuffy_helper_path}" "${tuffy_experiment_path}" "${psl_experiment_path}" "${experiment}" || exit 1

   popd > /dev/null
}

main "$@"
