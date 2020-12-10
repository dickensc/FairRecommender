#!/usr/bin/env bash

# run weight learning performance experiments,
#i.e. collects runtime and evaluation statistics of various weight learning methods

readonly THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
readonly BASE_DIR="${THIS_DIR}/.."
readonly BASE_OUT_DIR="${BASE_DIR}/results/fairness"

readonly STUDY_NAME='fairness_study'

#readonly FAIRNESS_MODELS='base non_parity value non_parity_value nb nmf non_parity_nmf_retro_fit value_nmf_retro_fit mutual_information'
readonly FAIRNESS_MODELS='mutual_information'
#readonly FAIRNESS_WEIGHTS='LEARNED 0.00001 0.0001 0.001 0.01 0.1 1.0 10.0 100.0 1000.0 10000.0 100000.0 1000000.0 10000000.0'
readonly FAIRNESS_WEIGHTS='1000000.0 10000000.0'
readonly WL_METHODS='UNIFORM'
readonly SEED=4
readonly TRACE_LEVEL='TRACE'

readonly SUPPORTED_DATASETS='movielens'
readonly SUPPORTED_FAIRNESS_MODELS='base non_parity value non_parity_value nmf nb non_parity_nmf_retro_fit value_nmf_retro_fit mutual_information'

# Evaluators to be use for each example
declare -A DATASET_EVALUATORS
DATASET_EVALUATORS[movielens]='Continuous'

# Weight support by fairness model
declare -A SUPPORTED_FAIR_WEIGHTS
SUPPORTED_FAIR_WEIGHTS[base]='LEARNED'
SUPPORTED_FAIR_WEIGHTS[non_parity]='LEARNED'
SUPPORTED_FAIR_WEIGHTS[value]='LEARNED'
SUPPORTED_FAIR_WEIGHTS[non_parity_value]='LEARNED'
SUPPORTED_FAIR_WEIGHTS[nb]='LEARNED'
SUPPORTED_FAIR_WEIGHTS[nmf]='LEARNED'
SUPPORTED_FAIR_WEIGHTS[mutual_information]='0.01 0.1 1.0 10.0 100.0 1000.0 10000.0 10000.0 100000.0 1000000.0 10000000.0'
SUPPORTED_FAIR_WEIGHTS[non_parity_nmf_retro_fit]='0.01 0.1 1.0 10.0 100.0 1000.0 10000.0'
SUPPORTED_FAIR_WEIGHTS[value_nmf_retro_fit]='0.00001 0.0001 0.001 0.01 0.1 1.0 10.0'

# Fair weight learning rate
declare -A LEARNING_RATES
LEARNING_RATES['LEARNED']='-D sgd.learningrate=1.0'
LEARNING_RATES['0.00001']='-D sgd.learningrate=1.0'
LEARNING_RATES['0.0001']='-D sgd.learningrate=1.0'
LEARNING_RATES['0.001']='-D sgd.learningrate=1.0'
LEARNING_RATES['0.01']='-D sgd.learningrate=10.0'
LEARNING_RATES['0.1']='-D sgd.learningrate=10.0'
LEARNING_RATES['1.0']='-D sgd.learningrate=10.0'
LEARNING_RATES['10.0']='-D sgd.learningrate=100.0'
LEARNING_RATES['100.0']='-D sgd.learningrate=100.0'
LEARNING_RATES['1000.0']='-D sgd.learningrate=100.0'
LEARNING_RATES['10000.0']='-D sgd.learningrate=1000.0'
LEARNING_RATES['100000.0']='-D sgd.learningrate=1000.0'
LEARNING_RATES['1000000.0']='-D sgd.learningrate=10000.0'
LEARNING_RATES['10000000.0']='-D sgd.learningrate=10000.0 -D reasoner.tolerance=1e-15f'

# Evaluators to be use for each example
# todo: (Charles D.) just read this information from psl example data directory rather than hardcoding
declare -A DATASET_FOLDS
DATASET_FOLDS[movielens]=5

function run_example() {
    local example_directory=$1
    local wl_method=$2
    local fairness_model=$3
    local fold=$4
    local fair_weight=$5

    local example_name
    example_name=$(basename "${example_directory}")

    echo "Running example ${example_name} : ${fairness_model} : ${fold} : ${wl_method} : lambda=${fair_weight}"

    local cli_directory="${BASE_DIR}/${example_directory}/cli"

    out_directory="${BASE_OUT_DIR}/psl/${STUDY_NAME}/${example_name}/${wl_method}/${evaluator}/${fairness_model}/${fair_weight}"/${fold}

    # Only make a new out directory if it does not already exist
    [[ -d "$out_directory" ]] || mkdir -p "$out_directory"

    # Setup experiment cli and data directory
    setup_fairness_experiment "${example_directory}" "${fairness_model}" "${cli_directory}"

    ##### WEIGHT LEARNING #####
    run_weight_learning "${example_name}" "${evaluator}" "${wl_method}" "${fairness_model}" "${fair_weight}" "${fold}" "${cli_directory}" "${out_directory}"

    ##### EVALUATION #####
    run_evaluation "${example_name}" "${evaluator}" "${fairness_model}" "${fair_weight}" "${fold}" "${out_directory}"

    return 0
}

function setup_fairness_experiment() {
      local example_directory=$1
      local fairness_model=$2
      local cli_directory=$3

      local example_name
      example_name=$(basename "${example_directory}")

      local fairness_model_directory="${BASE_DIR}/${example_directory}/${example_name}_${fairness_model}"
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
    local fairness_model=$3
    local fairness_weight=$4
    local fold=$5
    local out_directory=$6

    # path to output files
    local out_path="${out_directory}/eval_out.txt"
    local err_path="${out_directory}/eval_out.err"

    if [[ -e "${out_path}" ]]; then
        echo "Output file already exists, skipping: ${out_path}"
    else
        echo "Running ${example_name} ${evaluator} ${fairness_model} model (#${fold}) -- Evaluation."
        # call inference script for SRL model type
        pushd . > /dev/null
            cd "psl_scripts" || exit
            ./run_inference.sh "${example_name}" "${evaluator}" "${fairness_model}" "${fold}" "${out_directory}" ${LEARNING_RATES[${fairness_weight}]}> "$out_path" 2> "$err_path"
        popd > /dev/null
    fi
}

function run_weight_learning() {
    local example_name=$1
    local evaluator=$2
    local wl_method=$3
    local fairness_model=$4
    local fairness_weight=$5
    local fold=$6
    local cli_directory=$7
    local out_directory=$8

    # path to output files
    local out_path="${out_directory}/learn_out.txt"
    local err_path="${out_directory}/learn_out.err"

    local first_fair_weight
    first_fair_weight=$(echo ${FAIRNESS_WEIGHTS} | cut -d " " -f 1)

    if [ "${fairness_weight}" == "${first_fair_weight}" ] || [ ! -e "${out_directory}/../../${first_fair_weight}/${fold}/${example_name}-learned.psl" ]; then
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
              ./run_wl.sh "${example_name}" "${evaluator}" "${wl_method}" "${fairness_model}" "${fold}" "${SEED}" "${out_directory}" "${TRACE_LEVEL}" > "$out_path" 2> "$err_path"
          popd > /dev/null
      fi
    else
        echo "Using learned weights from fairness weight: ${first_fair_weight} run"
        echo "Copying cached learned model from earlier run into cli"
        # copy the learned weights into the cli and out directory for inference
        cp "${out_directory}/../../${first_fair_weight}/${fold}/${example_name}-learned.psl" "${out_directory}/"
        cp "${out_directory}/../../${first_fair_weight}/${fold}/${example_name}-learned.psl" "${cli_directory}/"
    fi

    # write the fairness weight
    write_fairness_weight "$fairness_weight" "$fairness_model" "$example_name" "$wl_method" "$cli_directory"
}

function write_fairness_weight() {
    local fairness_weight=$1
    local fairness_model=$2
    local example_name=$3
    local wl_method=$4
    local cli_directory=$5

    echo "${fairness_model}"

    # write fairness weights in learned psl file
    pushd . > /dev/null
        cd "${cli_directory}" || exit

        local rule

        if [[ ${fairness_model} != 'base' && ${fairness_weight} != 'LEARNED' ]]; then
          if [[ ${wl_method} == 'UNIFORM' ]]; then
            # set the fairness related rule weights in the learned file to the fairness_weight value and write to learned.psl file
            if [[ ${fairness_model} == 'non_parity' || ${fairness_model} == 'non_parity_nmf_retro_fit' ]]; then
              rule="group1_avg_rating\(c\) = group2_avg_rating\(c\)"
            elif [[ ${fairness_model} == 'value' || ${fairness_model} == 'value_nmf_retro_fit' ]]; then
              rule="pred_group_average_item_rating\(G1, I\) - obs_group_average_item_rating\(G1, I\) = pred_group_average_item_rating\(G2, I\) - obs_group_average_item_rating\(G2, I\)"
            elif [[ ${fairness_model} == 'non_parity_value' ]]; then
              rule="group1_avg_rating\(c\) = group2_avg_rating\(c\)"
              rule="pred_group_average_item_rating\(G1, I\) - obs_group_average_item_rating\(G1, I\) = pred_group_average_item_rating\(G2, I\) - obs_group_average_item_rating\(G2, I\)"
            elif [[ ${fairness_model} == 'mutual_information' ]]; then
              rule="@MI\[rating\(\+U1, I\), group_member\(\+U2, \+G\)\] \{U1: rated\(U1, I\)\}"
            fi
            sed -i -r "s/^[0-9]+.[0-9]+ : ${rule}|^[0-9]+ : ${rule}/${fairness_weight} : ${rule}/g"  "${example_name}-learned.psl"
          else
            if [[ ${fairness_model} == 'non_parity' || ${fairness_model} == 'non_parity_nmf_retro_fit' ]]; then
              rule="1.0 \* GROUP1_AVG_RATING\(c\) \+ -1.0 \* GROUP2_AVG_RATING\(c\) = 0.0"
            elif [[ ${fairness_model} == 'value' || ${fairness_model} == 'value_nmf_retro_fit' ]]; then
              rule="1.0 \* PRED_GROUP_AVERAGE_ITEM_RATING\(G1, I\) \+ -1.0 \* OBS_GROUP_AVERAGE_ITEM_RATING\(G1, I\) \+ -1.0 \* PRED_GROUP_AVERAGE_ITEM_RATING\(G2, I\) \+ 1.0 \* OBS_GROUP_AVERAGE_ITEM_RATING\(G2, I\) = 0.0"
            elif [[ ${fairness_model} == 'non_parity_value' ]]; then
              rule="group1_avg_rating\(c\) = group2_avg_rating\(c\)"
              rule="pred_group_average_item_rating\(G1, I\) - obs_group_average_item_rating\(G1, I\) = pred_group_average_item_rating\(G2, I\) - obs_group_average_item_rating\(G2, I\)"
            elif [[ ${fairness_model} == 'mutual_information' ]]; then
              rule="@MI\[rating\(\+U1, I\), group_member\(\+U2, \+G\)\] \{U1: rated\(U1, I\)\}"
            fi

            sed -i -r "s/^[0-9]+.[0-9]+ : ${rule}|^[0-9]+ : ${rule}/${fairness_weight} : ${rule}/g"  "${example_name}-learned.psl"
          fi
        fi

    popd > /dev/null
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
        for fair_weight in ${FAIRNESS_WEIGHTS}; do
             example_name=$(basename "${example_directory}")
             for evaluator in ${DATASET_EVALUATORS[${example_name}]}; do
                for ((fold=0; fold<${DATASET_FOLDS[${example_name}]}; fold++)) do
                   for fairness_model in ${FAIRNESS_MODELS}; do
                      if [[ "${SUPPORTED_DATASETS}" == *"${example_name}"* ]]; then
                          if [[ "${SUPPORTED_FAIRNESS_MODELS}" == *"${fairness_model}"* ]]; then
                             if [[ "${SUPPORTED_FAIR_WEIGHTS[${fairness_model}]}" == *"${fair_weight}"* ]]; then
                              run_example "${example_directory}" "${wl_method}" "${fairness_model}" "${fold}" "${fair_weight}"
                             fi
                          fi
                      fi
                   done
                done
              done
          done
        done
    done

    return 0
}

main "$@"
