#!/usr/bin/env bash

# runs psl weight learning,

readonly THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
readonly BASE_EXAMPLE_DIR="${THIS_DIR}/../../psl-datasets"

readonly SUPPORTED_EXAMPLES='movielens'

# Examples that cannot use int ids.
readonly STRING_IDS='entity-resolution simple-acquaintances user-modeling'

# Standard options for all examples and models
# note that this is assuming that we are only using datasets that have int-ids
# todo: (Charles D.) break this assumption
readonly POSTGRES_DB='psl'
readonly STANDARD_PSL_OPTIONS="--postgres ${POSTGRES_DB} -D admmreasoner.initialconsensusvalue=ZERO -D log4j.threshold=TRACE"

# Options specific to each example (missing keys yield empty strings).
declare -A EXAMPLE_OPTIONS
EXAMPLE_OPTIONS[citeseer]='-D categoricalevaluator.defaultpredicate=hasCat'
EXAMPLE_OPTIONS[cora]='-D categoricalevaluator.defaultpredicate=hasCat'
EXAMPLE_OPTIONS[epinions]=''
EXAMPLE_OPTIONS[jester]=''
EXAMPLE_OPTIONS[lastfm]=''

readonly PSL_VERSION='2.3.0-SNAPSHOT'

function run() {
    local cli_directory=$1
    local model=$2

    pushd . > /dev/null
        cd "${cli_directory}" || exit
        ./run.sh "$model" "$@"
    popd > /dev/null
}

function run_inference() {
    local example_name=$1
    # TODO: modify run script so phase is considered. Important for wrappers
    local phase=$2
    local fold=$3
    local evaluator=$4
    local out_directory=$5
    local model=$6

    shift 6

    local example_directory="${BASE_EXAMPLE_DIR}/${example_name}"
    local cli_directory="${example_directory}/cli"

    # deactivate weight learning step in run script
    deactivate_weight_learning "$example_directory"

    # modify runscript to run with the options for this study
    modify_run_script_options "$example_directory" "$evaluator"

    # modify data files to point to the fold
    modify_data_files "$example_directory" "$fold"

    # set the psl version for WL experiment
    set_psl_version "$PSL_VERSION" "$example_directory"

    # run evaluation
    run "${cli_directory}" "${model}" "$@"

    # modify data files to point back to the 0'th fold
    modify_data_files "$example_directory" 0

    # reactivate weight learning step in run script
    reactivate_weight_learning "$example_directory"

    # save inferred predicates
    mv "${cli_directory}/inferred-predicates" "${out_directory}/inferred-predicates"

    return 0
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

    local example_name
    example_name=$(basename "${example_directory}")

    local int_ids_options=''
    # Check for int ids.
    if [[ "${STRING_IDS}" != *"${example_name}"* ]]; then
        int_ids_options="--int-ids ${int_ids_options}"
    fi

    pushd . > /dev/null
        cd "${example_directory}/cli" || exit

        # set the ADDITIONAL_PSL_OPTIONS
        sed -i "s/^readonly ADDITIONAL_PSL_OPTIONS='.*'$/readonly ADDITIONAL_PSL_OPTIONS='${int_ids_options} ${STANDARD_PSL_OPTIONS}'/" run.sh

        # set the ADDITIONAL_EVAL_OPTIONS
        sed -i "s/^readonly ADDITIONAL_EVAL_OPTIONS='.*'$/readonly ADDITIONAL_EVAL_OPTIONS='--infer --eval org.linqs.psl.evaluation.statistics.${objective}Evaluator ${EXAMPLE_OPTIONS[${example_name}]}'/" run.sh
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

function main() {
    if [[ $# -le 5 ]]; then
        echo "USAGE: $0 <example name> <phase> <fold> <evaluator> <out directory> <model>"
        echo "USAGE: Examples can be among: ${SUPPORTED_EXAMPLES}"
        exit 1
    fi

    trap exit SIGINT

    run_inference "$@"
}

main "$@"