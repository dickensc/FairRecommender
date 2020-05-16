#!/usr/bin/env bash

# run weight learning performance experiments,
#i.e. collects runtime and evaluation statistics of various weight learning methods

readonly THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
readonly BASE_DIR="${THIS_DIR}/.."
readonly BASE_OUT_DIR="${BASE_DIR}/results/weightlearning"

readonly FAIRNESS_MODELS='BASELINE non_parity'

declare -A SUPPORTED_FAIRNESS_MODELS
SUPPORTED_FAIRNESS_MODELS[psl]='BASELINE non_parity'
SUPPORTED_FAIRNESS_MODELS[tuffy]='BASELINE non_parity'

# set of currently supported examples
readonly SUPPORTED_DATASETS='movielens'
readonly SUPPORTED_MODEL_TYPES='psl tuffy'

# Evaluators to be use for each example
declare -A DATASET_EVALUATORS
DATASET_EVALUATORS[movielens]='Continuous Ranking'

# Evaluators to be use for each example
# todo: (Charles D.) just read this information from psl example data directory rather than hardcoding
declare -A DATASET_FOLDS
DATASET_FOLDS[movielens]=1

declare -A MODEL_TYPE_TO_FILE_EXTENSION
MODEL_TYPE_TO_FILE_EXTENSION[psl]="psl"
MODEL_TYPE_TO_FILE_EXTENSION[tuffy]="mln"


function run_example() {
    local srl_model_type=$1
    local example_directory=$2
    local wl_method=$3

    local example_name
    example_name=$(basename "${example_directory}")

    local cli_directory="${BASE_DIR}/${example_directory}/cli"

    for evaluator in ${EXAMPLE_EVALUATORS[${example_name}]}; do
        for ((fold=0; fold<${EXAMPLE_FOLDS[${example_name}]}; fold++)) do

            out_directory="${BASE_OUT_DIR}/${srl_model_type}/performance_study/${example_name}/${wl_method}/${evaluator}/${fold}"

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
                cp "${out_directory}/${example_name}-learned.${MODEL_TYPE_TO_FILE_EXTENSION[${srl_model_type}]}" "${cli_directory}/"
            else
                # call weight learning script for SRL model type
                pushd . > /dev/null
                    cd "${srl_model_type}_scripts" || exit
                    /usr/bin/time -v --output="${time_path}" ./run_wl.sh "${example_name}" "${fold}" "${SEED}" "${ALPHA}" "performance_study" "${wl_method}" "${evaluator}" "${out_directory}" > "$out_path" 2> "$err_path"
#                    ./run_wl.sh "${example_name}" "${fold}" "${SEED}" "${ALPHA}" "performance_study" "${wl_method}" "${evaluator}" "${out_directory}" > "$out_path" 2> "$err_path"
                popd > /dev/null
            fi

            ##### EVALUATION #####
            echo "Running ${example_name} ${evaluator} (#${fold}) -- Evaluation."

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
                    /usr/bin/time -v --output="${time_path}" ./run_inference.sh "${example_name}" "eval" "${fold}" "${evaluator}" "${out_directory}" > "$out_path" 2> "$err_path"
#                    ./run_inference.sh "${example_name}" "eval" "${fold}" "${evaluator}" "${out_directory}" > "$out_path" 2> "$err_path"
                popd > /dev/null
            fi
        done
    done

    return 0
}

function main() {
    trap exit SIGINT

    if [[ $# -le 1 ]]; then
        echo "USAGE: $0 <srl modeltype> <example dir> ..."
        echo "USAGE: SRL model types may be among: ${SUPPORTED_MODEL_TYPES}"
        echo "USAGE: Example Directories can be among: ${SUPPORTED_EXAMPLES}"
        exit 1
    fi

    local srl_modeltype=$1
    shift

    for example_directory in "$@"; do
        for wl_method in ${WL_METHODS}; do
            if [[ "${SUPPORTED_WL_METHODS[${srl_modeltype}]}" == *"${wl_method}"* ]]; then
              run_example "${srl_modeltype}" "${example_directory}" "${wl_method}"
            fi
        done
    done

    return 0
}

main "$@"
