#!/usr/bin/env bash

# run weight learning performance experiments,
#i.e. collects runtime and evaluation statistics of various weight learning methods

readonly THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
readonly BASE_DIR="${THIS_DIR}/.."
readonly BASE_OUT_DIR="${BASE_DIR}/results/fairness"

readonly STUDY_NAME='fairness_study'

readonly WL_METHODS='UNIFORM BOWLSS'
readonly SEED=4
readonly TRACE_LEVEL='TRACE'

readonly SUPPORTED_DATASETS='movielens'

# Evaluators to be use for each example
declare -A DATASET_EVALUATORS
DATASET_EVALUATORS[movielens]='Continuous'

# Evaluators to be use for each example
# todo: (Charles D.) just read this information from psl example data directory rather than hardcoding
declare -A DATASET_FOLDS
DATASET_FOLDS[movielens]=5

function run_example() {
    local example_directory=$1
    local wl_method=$2
    local fold=$3

    local example_name
    example_name=$(basename "${example_directory}")

    local cli_directory="${BASE_DIR}/${example_directory}/cli"

    out_directory="${BASE_OUT_DIR}/psl/${STUDY_NAME}/${example_name}/${wl_method}/${evaluator}/FairPSL/LEARNED/${fold}"

    # Only make a new out directory if it does not already exist
    [[ -d "$out_directory" ]] || mkdir -p "$out_directory"

    # Setup experiment cli and data directory
    setup_fairness_experiment "${example_directory}" "${cli_directory}"

    ##### WEIGHT LEARNING #####
    run_weight_learning "${example_name}" "${evaluator}" "${wl_method}" "${fold}" "${cli_directory}" "${out_directory}"

    ##### EVALUATION #####
    run_evaluation "${example_name}" "${evaluator}" "${fold}" "${out_directory}"

    return 0
}

function setup_fairness_experiment() {
      local example_directory=$1
      local cli_directory=$2

      local example_name
      example_name=$(basename "${example_directory}")

      local fairness_model_directory="${BASE_DIR}/${example_directory}/${example_name}_FairPSL"
      echo "fairness_model_directory"
      echo "$fairness_model_directory"

      # copy the .data and .psl files into the cli directory
      cp "${fairness_model_directory}/${example_name}.psl" "${cli_directory}/${example_name}.psl"
      cp "${fairness_model_directory}/${example_name}-eval.data" "${cli_directory}/${example_name}-eval.data"
      cp "${fairness_model_directory}/${example_name}-learn.data" "${cli_directory}/${example_name}-learn.data"
}

function run_evaluation() {
    local example_name=$1
    local evaluator=$2
    local fold=$3
    local out_directory=$4

    # path to output files
    local out_path="${out_directory}/eval_out.txt"
    local err_path="${out_directory}/eval_out.err"

    if [[ -e "${out_path}" ]]; then
        echo "Output file already exists, skipping: ${out_path}"
    else
        echo "Running ${example_name} ${evaluator} FairPSL (#${fold}) -- Evaluation."
        # call inference script for SRL model type
        pushd . > /dev/null
            cd "psl_scripts" || exit
            ./run_inference.sh "${example_name}" "${evaluator}" "FairPSL" "${fold}" "${out_directory}" > "$out_path" 2> "$err_path"
        popd > /dev/null
    fi
}

function run_weight_learning() {
    local example_name=$1
    local evaluator=$2
    local wl_method=$3
    local fold=$4
    local cli_directory=$5
    local out_directory=$6

    # path to output files
    local out_path="${out_directory}/learn_out.txt"
    local err_path="${out_directory}/learn_out.err"

    if [[ -e "${out_path}" ]]; then
        echo "Output file already exists, skipping: ${out_path}"
        echo "Copying cached learned model from earlier run into cli"
        # copy the learned weights into the cli directory for inference
        cp "${out_directory}/${example_name}-learned.psl" "${cli_directory}/"
    else
        echo "Running ${example_name} ${evaluator} (#${fold}) -- ${wl_method}."
        # call weight learning script for SRL model type
        pushd . > /dev/null
            cd "psl_scripts" || exit
            ./run_wl.sh "${example_name}" "${evaluator}" "${wl_method}" "FairPSL" "${fold}" "${SEED}" "${out_directory}" "${TRACE_LEVEL}" > "$out_path" 2> "$err_path"
        popd > /dev/null
    fi
}

function main() {
    trap exit SIGINT

    if [[ $# -eq 0 ]]; then
        echo "USAGE: $0 <example dir> ..."
        exit 1
    fi

    local example_name

    for example_directory in "$@"; do
        for wl_method in ${WL_METHODS}; do
           example_name=$(basename "${example_directory}")
           for evaluator in ${DATASET_EVALUATORS[${example_name}]}; do
              for ((fold=0; fold<${DATASET_FOLDS[${example_name}]}; fold++)) do
                  if [[ "${SUPPORTED_DATASETS}" == *"${example_name}"* ]]; then
                      run_example "${example_directory}" "${wl_method}" "${fold}"
                  fi
               done
            done
        done
    done

    return 0
}

main "$@"
