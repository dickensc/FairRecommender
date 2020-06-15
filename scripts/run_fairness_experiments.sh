#!/usr/bin/env bash

# run weight learning performance experiments,
#i.e. collects runtime and evaluation statistics of various weight learning methods

readonly THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
readonly BASE_DIR="${THIS_DIR}/.."
readonly BASE_OUT_DIR="${BASE_DIR}/results/fairness"

readonly FAIRNESS_MODELS='base non_parity value'
readonly WL_METHODS='UNIFORM'
readonly SEED=22
readonly TRACE_LEVEL='TRACE'

declare -A SUPPORTED_DATASETS
SUPPORTED_DATASETS[psl]='movielens'

declare -A SUPPORTED_FAIRNESS_MODELS
SUPPORTED_FAIRNESS_MODELS[psl]='base non_parity value'

# set of currently supported examples
readonly SUPPORTED_MODEL_TYPES='psl'

# Evaluators to be use for each example
declare -A DATASET_EVALUATORS
DATASET_EVALUATORS[movielens]='Continuous'
DATASET_EVALUATORS[movielens_non_parity]='Continuous'
DATASET_EVALUATORS[movielens_value]='Continuous'

# Evaluators to be use for each example
# todo: (Charles D.) just read this information from psl example data directory rather than hardcoding
declare -A DATASET_FOLDS
DATASET_FOLDS[movielens]=1
DATASET_FOLDS[movielens_non_parity]=1
DATASET_FOLDS[movielens_value]=1

declare -A MODEL_TYPE_TO_FILE_EXTENSION
MODEL_TYPE_TO_FILE_EXTENSION[psl]="psl"


function run_example() {
    local srl_model_type=$1
    local example_directory=$2
    local wl_method=$3
    local fairness_model=$4
    local fold=$5

    local example_name
    example_name=$(basename "${example_directory}")

    local cli_directory="${BASE_DIR}/${example_directory}/cli"

    out_directory="${BASE_OUT_DIR}/${srl_model_type}/performance_study/${example_name}/${wl_method}/${evaluator}/${fold}/${fairness_model}"

    # Only make a new out directory if it does not already exist
    [[ -d "$out_directory" ]] || mkdir -p "$out_directory"

    ##### WEIGHT LEARNING #####
    echo "Running ${example_name} ${evaluator} (#${fold}) -- ${wl_method}."

    # path to output files
    local out_path="${out_directory}/learn_out.txt"
    local err_path="${out_directory}/learn_out.err"
    local time_path="${out_directory}/learn_time.txt"

    if [[ -e "${out_path}" ]]; then
        echo "Output file already exists, skipping: ${out_path}"
        echo "Copying cached learned model from earlier run into cli"
        # copy the learned weights into the cli directory for inference
        cp "${out_directory}/${example_name}_${fairness_model}-learned.${MODEL_TYPE_TO_FILE_EXTENSION[${srl_model_type}]}" "${cli_directory}/"
    else
        # call weight learning script for SRL model type
        pushd . > /dev/null
            cd "${srl_model_type}_scripts" || exit
            ./run_wl.sh "${example_name}" "${fold}" "${SEED}" "performance_study" "${wl_method}" "${evaluator}" "${out_directory}" "${TRACE_LEVEL}" "${fairness_model}" > "$out_path" 2> "$err_path"
        popd > /dev/null
    fi

    ##### EVALUATION #####
    echo "Running ${example_name} ${evaluator} ${fairness_model} model (#${fold}) -- Evaluation."

    # path to output files
    local out_path="${out_directory}/eval_out.txt"
    local err_path="${out_directory}/eval_out.err"
    local time_path="${out_directory}/eval_time.txt"

    if [[ -e "${out_path}" ]]; then
        echo "Output file already exists, skipping: ${out_path}"
    else
        # call inference script for SRL model type
        pushd . > /dev/null
            cd "${srl_model_type}_scripts" || exit
            ./run_inference.sh "${example_name}" "eval" "${fold}" "${evaluator}" "${out_directory}" "${fairness_model}" > "$out_path" 2> "$err_path"
        popd > /dev/null
    fi

    return 0
}

function main() {
    trap exit SIGINT

    if [[ $# -le 1 ]]; then
        echo "USAGE: $0 <srl modeltype> <example dir> ..."
        exit 1
    fi

    local srl_modeltype=$1
    shift

    local example_name

    for example_directory in "$@"; do
        echo "$example_directory"
        for wl_method in ${WL_METHODS}; do
           example_name=$(basename "${example_directory}")
           for evaluator in ${DATASET_EVALUATORS[${example_name}]}; do
              for ((fold=0; fold<${DATASET_FOLDS[${example_name}]}; fold++)) do
                 for fairness_model in ${FAIRNESS_MODELS}; do
                    #echo "$example_name"
                    if [[ "${SUPPORTED_DATASETS[${srl_modeltype}]}" == *"${example_name}"* ]]; then
                        if [[ "${SUPPORTED_FAIRNESS_MODELS[${srl_modeltype}]}" == *"${fairness_model}"* ]]; then
                            run_example "${srl_modeltype}" "${example_directory}" "${wl_method}" "${fairness_model}" "${fold}"
                        fi
                    fi
                 done
              done
            done
        done
    done

    return 0
}

main "$@"
