#!/usr/bin/env bash

# runs psl weight learning,

readonly THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
readonly BASE_EXAMPLE_DIR="${THIS_DIR}/../../psl-datasets"

readonly SUPPORTED_EXAMPLES='movielens'

# Examples that cannot use int ids.
readonly STRING_IDS='FairPSL'

# Standard options for all examples and models
# note that this is assuming that we are only using datasets that have int-ids
# todo: (Charles D.) break this assumption
readonly POSTGRES_DB='psl'
readonly STANDARD_PSL_OPTIONS="--postgres ${POSTGRES_DB} -D log4j.threshold=TRACE"

# Options specific to each example (missing keys yield empty strings).
declare -A EXAMPLE_OPTIONS
EXAMPLE_OPTIONS[citeseer]='-D categoricalevaluator.defaultpredicate=hasCat'
EXAMPLE_OPTIONS[cora]='-D categoricalevaluator.defaultpredicate=hasCat'
EXAMPLE_OPTIONS[epinions]=''
EXAMPLE_OPTIONS[jester]=''
EXAMPLE_OPTIONS[lastfm]=''

# Options specific to each model (missing keys yield empty strings).
declare -A MODEL_OPTIONS
MODEL_OPTIONS[base]='-D sgd.maxiterations=500'
MODEL_OPTIONS[non_parity]='-D sgd.maxiterations=500'
MODEL_OPTIONS[value]='-D sgd.maxiterations=500'
MODEL_OPTIONS[non_parity_value]='-D sgd.maxiterations=500'
MODEL_OPTIONS[nb]='-D sgd.maxiterations=500'
MODEL_OPTIONS[nmf]='-D sgd.maxiterations=500'
MODEL_OPTIONS[mutual_information]='-D sgd.maxiterations=500'
MODEL_OPTIONS[non_parity_nmf_retro_fit]='-D sgd.maxiterations=500'
MODEL_OPTIONS[value_nmf_retro_fit]='-D sgd.maxiterations=500'

readonly PSL_VERSION='2.3.0-SNAPSHOT'

function run() {
    local cli_directory=$1

    shift 1

    pushd . > /dev/null
        cd "${cli_directory}" || exit
        ./run.sh "$@"
    popd > /dev/null
}

function run_inference() {
    local example_name=$1
    local evaluator=$2
    local fairness_model=$3
    local fold=$4
    local out_directory=$5

    shift 5

    local example_directory="${BASE_EXAMPLE_DIR}/${example_name}"
    local cli_directory="${example_directory}/cli"

    # deactivate weight learning step in run script
    deactivate_weight_learning "$example_directory"

    # reactivate evaluation step in run script
    reactivate_evaluation "$example_directory"

    # modify runscript to run with the options for this study
    modify_run_script_options "$example_directory" "$evaluator" "$fairness_model"

    # modify data files to point to the fold
    modify_data_files "$example_directory" "$fold"

    # modify the model file if necessary for the fairness intervention
    modify_model_file "$example_directory" "$fairness_model" "$fold"

    # set the psl version for WL experiment
    set_psl_version "$PSL_VERSION" "$example_directory"

    # run evaluation
    run "${cli_directory}" "$@"

    # modify data files to point back to the 0'th fold
    modify_data_files "$example_directory" 0

    # Copy the original model file back into the cli directory
    cp "${BASE_EXAMPLE_DIR}/${example_name}/${example_name}_${fairness_model}/${example_name}.psl" "${cli_directory}/${example_name}.psl"

    # reactivate weight learning step in run script
    reactivate_weight_learning "$example_directory"

    # save inferred predicates
    mv "${cli_directory}/inferred-predicates" "${out_directory}/inferred-predicates"

    return 0
}

function reactivate_evaluation() {
    local example_directory=$1
    local example_name
    example_name=$(basename "${example_directory}")

    # reactivate evaluation step in run script
    pushd . > /dev/null
        cd "${example_directory}/cli" || exit

        # reactivate evaluation.
        sed -i 's/^\(\s\+\)# runEvaluation/\1runEvaluation/' run.sh

    popd > /dev/null
}

function set_psl_version() {
    local psl_version=$1
    local example_directory=$2

    pushd . > /dev/null
      cd "${example_directory}/cli"

      # Set the PSL version.
      sed -i "s/^readonly PSL_VERSION='.*'$/readonly PSL_VERSION='${psl_version}'/" run.sh

    popd > /dev/null
}

function deactivate_weight_learning() {
    local example_directory=$1
    local example_name
    example_name=$(basename "${example_directory}")

    # deactivate weight learning step in run script
    pushd . > /dev/null
        cd "${example_directory}/cli" || exit

        # deactivate weight learning.
        sed -i 's/^\(\s\+\)runWeightLearning/\1# runWeightLearning/' run.sh

    popd > /dev/null
}

function reactivate_weight_learning() {
    local example_directory=$1
    local example_name
    example_name=$(basename "${example_directory}")

    # reactivate weight learning step in run script
    pushd . > /dev/null
        cd "${example_directory}/cli" || exit

        # reactivate weight learning.
        sed -i 's/^\(\s\+\)# runWeightLearning/\1runWeightLearning/' run.sh

    popd > /dev/null
}

function modify_run_script_options() {
    local example_directory=$1
    local objective=$2
    local fairness_model=$3

    local example_name
    example_name=$(basename "${example_directory}")

    local int_ids_options=''
    # Check for int ids.
    if [[ "${STRING_IDS}" != *"${fairness_model}"* ]]; then
        int_ids_options="--int-ids ${int_ids_options}"
    fi

    pushd . > /dev/null
        cd "${example_directory}/cli" || exit

        # set the ADDITIONAL_PSL_OPTIONS
        sed -i "s/^readonly ADDITIONAL_PSL_OPTIONS='.*'$/readonly ADDITIONAL_PSL_OPTIONS='${int_ids_options} ${STANDARD_PSL_OPTIONS}'/" run.sh

        # set the ADDITIONAL_EVAL_OPTIONS
        sed -i "s/^readonly ADDITIONAL_EVAL_OPTIONS='.*'$/readonly ADDITIONAL_EVAL_OPTIONS='--infer SGDInference ${MODEL_OPTIONS[${fairness_model}]} --eval org.linqs.psl.evaluation.statistics.${objective}Evaluator ${EXAMPLE_OPTIONS[${example_name}]}'/" run.sh
    popd > /dev/null
}

function modify_data_files() {
    local example_directory=$1
    local new_fold=$2

    local example_name
    example_name=$(basename "${example_directory}")

    pushd . > /dev/null
        cd "${example_directory}/cli" || exit

        # update the fold in the .data file
        sed -i -E "s/${example_name}\/[0-9]+\/eval/${example_name}\/${new_fold}\/eval/g" "${example_name}"-eval.data
    popd > /dev/null
}

function modify_model_file() {
    local example_directory=$1
    local fairness_metric=$2
    local fold=$3

    local example_name
    example_name=$(basename "${example_directory}")

    if [[ "${fairness_metric}" == "non_parity" || "${fairness_metric}" == "non_parity_nmf_retro_fit" || ${fairness_model} == 'non_parity_value' ]]; then
      pushd . > /dev/null
          cd "${example_directory}/cli" || exit

          # replace the denominator in the model file with the learn fold specific value
          local group_1_denominator
          local group_2_denominator
          group_1_denominator=$( grep F "../data/${example_name}/${fold}/eval/group_denominators_obs.txt" | cut -f 2 )
          group_2_denominator=$( grep M "../data/${example_name}/${fold}/eval/group_denominators_obs.txt" | cut -f 2 )
          sed -i -E "s/DENOMINATOR_1/${group_1_denominator}/g" "${example_name}"-learned.psl
          sed -i -E "s/DENOMINATOR_2/${group_2_denominator}/g" "${example_name}"-learned.psl
      popd > /dev/null
    fi
}

function main() {
    if [[ $# -le 4 ]]; then
        echo "USAGE: $0 <example_name> <evaluator> <fairness_model> <fold> <out_directory>"
        echo "USAGE: Examples can be among: ${SUPPORTED_EXAMPLES}"
        exit 1
    fi

    trap exit SIGINT

    run_inference "$@"
}

main "$@"