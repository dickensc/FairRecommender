#!/usr/bin/env bash

# run weight learning performance experiments,
#i.e. collects runtime and evaluation statistics of various weight learning methods

readonly THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
readonly BASE_DIR="${THIS_DIR}/.."
readonly BASE_OUT_DIR="${BASE_DIR}/results/fairness"

readonly STUDY_NAME='fairness_study'

readonly FAIRNESS_WEIGHTS='0.001'
readonly FAIRNESS_MODELS='None Absolute Over+Under Overestimation Parity Underestimation Value'
readonly SEED=22

readonly SUPPORTED_DATASETS='movielens'

# Evaluators to be use for each example
declare -A DATASET_MAIN_SCRIPTS
DATASET_MAIN_SCRIPTS[movielens]='./mainML.py'

# Evaluators to be use for each example
# todo: (Charles D.) just read this information from psl example data directory rather than hardcoding
declare -A DATASET_FOLDS
DATASET_FOLDS[movielens]=5


function run_example() {
    local example_directory=$1
    local fold=$2
    local fair_weight=$3

    local example_name
    example_name=$(basename "${example_directory}")


    ##### EVALUATION #####
    run_evaluation "${example_name}" "${fold}" "${fair_weight}"

    return 0
}

function run_evaluation() {
    local example_name=$1
    local fold=$2
    local fair_weight=$3

    # Only make a new out directory if it does not already exist
    local out_directory="${BASE_OUT_DIR}/mf/${STUDY_NAME}/${example_name}/${fair_weight}/${fold}"
    [[ -d "$out_directory" ]] || mkdir -p "$out_directory"

    # path to output files
    local out_path="${out_directory}/eval_out.txt"
    local err_path="${out_directory}/eval_out.err"

    # path to data files
    local train_data_path="../../psl-datasets/${example_name}/data/${example_name}/${fold}/eval/rating_obs_mf.txt"
    local test_data_path="../../psl-datasets/${example_name}/data/${example_name}/${fold}/eval/rating_truth_mf.txt"
    local protected_group_data_path="../../psl-datasets/${example_name}/data/${example_name}/${fold}/eval/group_member_mf.txt"

    if [[ -e "${out_path}" ]]; then
        echo "Output file already exists, skipping: ${out_path}"
    else
        echo "Running MF ${example_name} (#${fold})"
        pushd . > /dev/null
            cd "beyond_parity_scripts" || exit
            python2.7 "${DATASET_MAIN_SCRIPTS[${example_name}]}" "${train_data_path}" "${test_data_path}" "${protected_group_data_path}" "${out_directory}" "${fair_weight}" > "$out_path" 2> "$err_path"
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
      for fair_weight in ${FAIRNESS_WEIGHTS}; do
         example_name=$(basename "${example_directory}")
         for ((fold=0; fold<${DATASET_FOLDS[${example_name}]}; fold++)) do
            if [[ "${SUPPORTED_DATASETS}" == *"${example_name}"* ]]; then
                run_example "${example_directory}" "${fold}" "${fair_weight}"
            fi
         done
      done
    done

    return 0
}

main "$@"
