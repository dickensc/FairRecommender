#!/usr/bin/env bash

# runs psl weight learning,

readonly THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
readonly BASE_EXAMPLE_DIR="${THIS_DIR}/../../psl-datasets"

readonly SUPPORTED_WL_METHODS='UNIFORM BOWLOS BOWLSS CRGS HB RGS LME MLE MPLE'
readonly SUPPORTED_EXAMPLES='movielens'

# Examples that cannot use int ids.
readonly STRING_IDS='FairPSL'

# Standard options for all examples and models
# note that this is assuming that we are only using datasets that have int-ids
# todo: (Charles D.) break this assumption
readonly POSTGRES_DB='psl'
readonly STANDARD_PSL_OPTIONS="--postgres ${POSTGRES_DB}"
# Random Seed option
readonly WEIGHT_LEARNING_SEED='-D random.seed='
# Inference reasoner
readonly WEIGHT_LEARNING_INFERENCE='-D weightlearning.inference=SGDInference'

# The weight learning classes for each method
declare -A WEIGHT_LEARNING_METHODS
WEIGHT_LEARNING_METHODS[BOWLOS]='--learn org.linqs.psl.application.learning.weight.bayesian.GaussianProcessPrior'
WEIGHT_LEARNING_METHODS[BOWLSS]='--learn org.linqs.psl.application.learning.weight.bayesian.GaussianProcessPrior'
WEIGHT_LEARNING_METHODS[CRGS]='--learn org.linqs.psl.application.learning.weight.search.grid.ContinuousRandomGridSearch'
WEIGHT_LEARNING_METHODS[HB]='--learn org.linqs.psl.application.learning.weight.search.Hyperband'
WEIGHT_LEARNING_METHODS[RGS]='--learn org.linqs.psl.application.learning.weight.search.grid.RandomGridSearch'
WEIGHT_LEARNING_METHODS[LME]='--learn'
WEIGHT_LEARNING_METHODS[MLE]='--learn'
WEIGHT_LEARNING_METHODS[MPLE]='--learn org.linqs.psl.application.learning.weight.maxlikelihood.MaxPseudoLikelihood'
WEIGHT_LEARNING_METHODS[UNIFORM]=''

# Options specific to each method (missing keys yield empty strings).
declare -A WEIGHT_LEARNING_METHOD_OPTIONS
WEIGHT_LEARNING_METHOD_OPTIONS[BOWLOS]='-D admmreasoner.initialconsensusvalue=ZERO -D gppker.reldep=1 -D gpp.explore=1 -D gpp.maxiterations=50 -D gppker.space=OS -D gpp.initialweightstd=0.5 -D gpp.initialweightvalue=0.5'
WEIGHT_LEARNING_METHOD_OPTIONS[BOWLSS]='-D admmreasoner.initialconsensusvalue=ZERO -D gppker.reldep=1 -D gpp.explore=1 -D gpp.maxiterations=25 -D gppker.space=SS -D gpp.initialweightstd=0.5 -D gpp.initialweightvalue=0.5'
WEIGHT_LEARNING_METHOD_OPTIONS[CRGS]='-D admmreasoner.initialconsensusvalue=ZERO -D continuousrandomgridsearch.maxlocations=25'
WEIGHT_LEARNING_METHOD_OPTIONS[HB]='-D admmreasoner.initialconsensusvalue=ZERO'
WEIGHT_LEARNING_METHOD_OPTIONS[RGS]='-D admmreasoner.initialconsensusvalue=ZERO -D randomgridsearch.maxlocations=50'
WEIGHT_LEARNING_METHOD_OPTIONS[LME]='-D admmreasoner.initialconsensusvalue=ZERO -D frankwolfe.maxiter=100 -D weightlearning.randomweights=true'
WEIGHT_LEARNING_METHOD_OPTIONS[MLE]='-D admmreasoner.initialconsensusvalue=ZERO -D votedperceptron.numsteps=100 -D votedperceptron.stepsize=0.5 -D votedperceptron.inertia=0.5 -D weightlearning.randomweights=true'
WEIGHT_LEARNING_METHOD_OPTIONS[MPLE]='-D votedperceptron.zeroinitialweights=true -D votedperceptron.numsteps=100 -D votedperceptron.stepsize=1.0 -D weightlearning.randomweights=true'
WEIGHT_LEARNING_METHOD_OPTIONS[UNIFORM]=''

# Options specific to each method (missing keys yield empty strings).
declare -A WEIGHT_LEARNING_METHOD_PSL_PSL_VERSION
WEIGHT_LEARNING_METHOD_PSL_PSL_VERSION[BOWLOS]='2.3.0-SNAPSHOT'
WEIGHT_LEARNING_METHOD_PSL_PSL_VERSION[BOWLSS]='2.3.0-SNAPSHOT'
WEIGHT_LEARNING_METHOD_PSL_PSL_VERSION[CRGS]='2.3.0-SNAPSHOT'
WEIGHT_LEARNING_METHOD_PSL_PSL_VERSION[HB]='2.3.0-SNAPSHOT'
WEIGHT_LEARNING_METHOD_PSL_PSL_VERSION[RGS]='2.3.0-SNAPSHOT'
WEIGHT_LEARNING_METHOD_PSL_PSL_VERSION[LME]='max-margin'
WEIGHT_LEARNING_METHOD_PSL_PSL_VERSION[MLE]='2.3.0-SNAPSHOT'
WEIGHT_LEARNING_METHOD_PSL_PSL_VERSION[MPLE]='2.2.1'
WEIGHT_LEARNING_METHOD_PSL_PSL_VERSION[UNIFORM]='2.2.1'

# Weight learning methods that can optimize an arbitrary objective
readonly SEARCH_BASED_LEARNERS='BOWLOS BOWLSS CRGS HB RGS'

# Options specific to each example (missing keys yield empty strings).
declare -A EXAMPLE_OPTIONS
EXAMPLE_OPTIONS[movielens]=''

function run() {
    local cli_directory=$1

    shift 1

    pushd . > /dev/null
        cd "${cli_directory}" || exit
        ./run.sh "$@"
    popd > /dev/null
}

function run_weight_learning() {
    local example_name=$1
    local evaluator=$2
    local wl_method=$3
    local fairness_model=$4
    local fold=$5
    local seed=$6
    local out_directory=$7
    local trace_level=$8

    shift 8

    local example_directory="${BASE_EXAMPLE_DIR}/${example_name}"
    local cli_directory="${example_directory}/cli"

    # Check if uniform weight run
    if [[ "${wl_method}" == "UNIFORM" ]]; then
        # if so, write uniform weights to -learned.psl file for evaluation
        write_uniform_learned_psl_file "$example_directory"

    elif [[ "${SUPPORTED_WL_METHODS}" == *"${wl_method}"* ]]; then
        # deactivate evaluation step in run script
        deactivate_evaluation "$example_directory"

        # reactivate weight learning step in run script
        reactivate_weight_learning "$example_directory"

        # modify runscript to run with the options for this study
        modify_run_script_options "$example_directory" "$wl_method" "$evaluator" "$seed" "$trace_level"

        # modify data files to point to the fold
        modify_data_files "$example_directory" "$fold"

        # modify the model file if necessary for the fairness intervention
        modify_model_file "$example_directory" "$fairness_model" "$fold"

        # set the psl version for WL experiment
        set_psl_version "${WEIGHT_LEARNING_METHOD_PSL_PSL_VERSION[${wl_method}]}" "$example_directory"

        # run weight learning
        run  "${cli_directory}" "$@"

        # modify data files to point back to the 0'th fold
        modify_data_files "$example_directory" 0

        # Copy the original model file back into the cli directory
        cp "${BASE_EXAMPLE_DIR}/${example_name}/${example_name}_${fairness_model}/${example_name}.psl" "${cli_directory}/${example_name}.psl"

        # reactivate evaluation step in run script
        reactivate_evaluation "$example_directory"
    else
        echo "USAGE: Weight learning method: ${wl_method} not supported can be among: ${SUPPORTED_WL_METHODS}"
    fi

    # save learned model
    cp "${cli_directory}/${example_name}-learned.psl" "${out_directory}/${example_name}-learned.psl"

    return 0
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

function deactivate_evaluation() {
    local example_directory=$1
    local example_name
    example_name=$(basename "${example_directory}")

    # deactivate evaluation step in run script
    pushd . > /dev/null
        cd "${example_directory}/cli" || exit

        # deactivate evaluation.
        sed -i 's/^\(\s\+\)runEvaluation/\1# runEvaluation/' run.sh

    popd > /dev/null
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

function write_uniform_learned_psl_file() {
    local example_directory=$1
    local example_name
    example_name=$(basename "${example_directory}")

    # write uniform weights as learned psl file
    pushd . > /dev/null
        cd "${example_directory}/cli" || exit

#        # set the weights in the learned file to 1 and write to learned.psl file
#        sed -r "s/^[0-9]+.[0-9]+ :|^[0-9]+ :/1.0 :/g"  "${example_name}.psl" > "${example_name}-learned.psl"

        cp "${example_name}.psl" "${example_name}-learned.psl"
    popd > /dev/null
}

function modify_run_script_options() {
    local example_directory=$1
    local wl_method=$2
    local objective=$3
    local seed=$4
    local trace_level=$5

    local example_name
    example_name=$(basename "${example_directory}")

    local evaluator_options=''
    local int_ids_options=''
    local search_options=''

    # Check for SEARCH_BASED_LEARNERS.
    if [[ "${SEARCH_BASED_LEARNERS}" == *"${wl_method}"* ]]; then
        evaluator_options="-D weightlearning.evaluator=org.linqs.psl.evaluation.statistics.${objective}Evaluator"
    fi

    # Check for int ids.
    if [[ "${STRING_IDS}" != *"${fairness_model}"* ]]; then
        int_ids_options="--int-ids ${int_ids_options}"
    fi

    pushd . > /dev/null
        cd "${example_directory}/cli" || exit

        # set the ADDITIONAL_LEARN_OPTIONS
        sed -i "s/^readonly ADDITIONAL_LEARN_OPTIONS='.*'$/readonly ADDITIONAL_LEARN_OPTIONS='${WEIGHT_LEARNING_METHODS[${wl_method}]} ${WEIGHT_LEARNING_INFERENCE} ${WEIGHT_LEARNING_SEED}${seed} ${WEIGHT_LEARNING_METHOD_OPTIONS[${wl_method}]} -D log4j.threshold=${trace_level} ${EXAMPLE_OPTIONS[${example_name}]} ${evaluator_options} ${search_options}'/" run.sh

        # set the ADDITIONAL_PSL_OPTIONS
        sed -i "s/^readonly ADDITIONAL_PSL_OPTIONS='.*'$/readonly ADDITIONAL_PSL_OPTIONS='${int_ids_options} ${STANDARD_PSL_OPTIONS}'/" run.sh
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

function modify_data_files() {
    local example_directory=$1
    local new_fold=$2

    local example_name
    example_name=$(basename "${example_directory}")

    pushd . > /dev/null
        cd "${example_directory}/cli" || exit

        # update the fold in the .data file
        sed -i -E "s/${example_name}\/[0-9]+\/learn/${example_name}\/${new_fold}\/learn/g" "${example_name}"-learn.data
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
          group_1_denominator=$( grep F "../data/${example_name}/${fold}/learn/group_denominators_obs.txt" | cut -f 2 )
          group_2_denominator=$( grep M "../data/${example_name}/${fold}/learn/group_denominators_obs.txt" | cut -f 2 )
          sed -i -E "s/DENOMINATOR_1/${group_1_denominator}/g" "${example_name}".psl
          sed -i -E "s/DENOMINATOR_2/${group_2_denominator}/g" "${example_name}".psl
      popd > /dev/null
    fi
}

function main() {
    trap exit SIGINT
    if [[ $# -le 7 ]]; then
        echo "USAGE: $0 <example_name> <evaluator> <wl_method> <fairness_model> <fold> <seed> <outDir> <trace_level>"
        echo "USAGE: Examples can be among: ${SUPPORTED_EXAMPLES}"
        echo "USAGE: Weight Learning methods can be among: ${SUPPORTED_WL_METHODS}"
        exit 1
    fi

    run_weight_learning "$@"
}

main "$@"