#!/usr/bin/env bash

# run weight learning performance experiments,
#i.e. collects runtime and evaluation statistics of various weight learning methods

readonly THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
readonly BASE_DIR="${THIS_DIR}/../.."

# the misc tuffy resources
readonly TUFFY_RESOURCES_DIR="${BASE_DIR}/tuffy_resources"
readonly TUFFY_CONFIG="${TUFFY_RESOURCES_DIR}/tuffy.conf"
readonly TUFFY_JAR="${TUFFY_RESOURCES_DIR}/tuffy.jar"

# the tuffy examples path
readonly TUFFY_EXAMPLES="${BASE_DIR}/tuffy-datasets"

# the weight learning wrapper script paths
readonly RGS_WRAPPER="${BASE_DIR}/scripts/weight_learning_wrappers/rgs.py"
readonly CRGS_WRAPPER="${BASE_DIR}/scripts/weight_learning_wrappers/crgs.py"
readonly HB_WRAPPER="${BASE_DIR}/scripts/weight_learning_wrappers/hb.py"
readonly BOWLOS_WRAPPER="${BASE_DIR}/scripts/weight_learning_wrappers/bowlos.py"

# set of currently supported PSL examples
readonly SUPPORTED_EXAMPLES='movielens'
readonly SUPPORTED_WL_METHODS='UNIFORM DiagonalNewton CRGS RGS'

# Weight learning methods that are built in to Tuffy
readonly BUILT_IN_LEARNERS='DiagonalNewton'

# Options specific to each example (missing keys yield empty strings).
declare -A EXAMPLE_OPTIONS
EXAMPLE_OPTIONS[movielens]=''

#readonly AVAILABLE_MEM_KB=$(cat /proc/meminfo | grep 'MemTotal' | sed 's/^[^0-9]\+\([0-9]\+\)[^0-9]\+$/\1/')
## Floor by multiples of 5 and then reserve an additional 5 GB.
#readonly JAVA_MEM_GB=$((${AVAILABLE_MEM_KB} / 1024 / 1024 / 5 * 5 - 5))
readonly JAVA_MEM_GB=8

function run_weight_learning() {
    local example_name=$1
    local fold=$2
    local seed=$3
    local study=$4
    local wl_method=$5
    local evaluator=$6
    local out_directory=$7
    local trace_level=$8
    local model=$9



    local example_directory="${TUFFY_EXAMPLES}/${example_name}"

    # run tuffy weight learning
    local prog_file="${example_directory}/cli/prog.mln"
    local results_file="${example_directory}/cli/wl_results.txt"

    if [[ "${wl_method}" == "UNIFORM" ]]; then
        # if so, write uniform weights to -learned.psl file for evaluation
        write_uniform_learned_tuffy_file "$example_directory/cli" "$example_name"
    elif [[ "${BUILT_IN_LEARNERS}" == *"${wl_method}"* ]]; then
        local evidence_file="${example_directory}/data/${example_name}/${fold}/built_in_learn/evidence.db"
        local query_file="${example_directory}/data/${example_name}/${fold}/built_in_learn/query.db"

        java -Xmx${JAVA_MEM_GB}G -Xms${JAVA_MEM_GB}G -jar "$TUFFY_JAR" -learnwt -mln "$prog_file" -evidence "$evidence_file" -queryFile "$query_file" -r "$results_file" -conf "$TUFFY_CONFIG" ${EXAMPLE_OPTIONS[${example_name}]} -verbose 3
        write_average_weights "$results_file" "$example_name"

        # save weight learning results
        mv "$results_file" "${out_directory}/wl_results.txt"
    else
        if [[ "${wl_method}" == "RGS" ]]; then
            python3 "$RGS_WRAPPER" "tuffy" "${evaluator}" "${example_name}" "${fold}" "${seed}" "${alpha}" "${study}" "${out_directory}"
        elif [[ "${wl_method}" == "CRGS" ]]; then
            python3 "$CRGS_WRAPPER" "tuffy" "${evaluator}" "${example_name}" "${fold}" "${seed}" "${alpha}" "${study}" "${out_directory}"
        elif [[ "${wl_method}" == "HB" ]]; then
            python3 "$HB_WRAPPER" "tuffy" "${evaluator}" "${example_name}" "${fold}" "${seed}" "${alpha}" "${study}" "${out_directory}"
        elif [[ "${wl_method}" == "BOWLOS" ]]; then
            python3 "$BOWLOS_WRAPPER" "tuffy" "${evaluator}" "${example_name}" "${fold}" "${seed}" "${alpha}" "${study}" "${out_directory}"
        else
            echo "Method: ${wl_method} not yet supported"
            return 1
        fi
    fi

    # save learned model
    cp "${example_directory}/cli/${example_name}-learned.mln" "${out_directory}/${example_name}-learned.mln"

    return 0
}

function write_average_weights () {
    local tuffy_wl_results=$1
    local example_name=$2

    local example_directory
    example_directory=$(dirname "${tuffy_wl_results}")

    local learned_prog_file
    learned_prog_file=$(basename "${tuffy_wl_results}")

    # write the average weight learning step in run script
    pushd . > /dev/null
        cd "${example_directory}" || exit

        # since tuffy outputs models that are not readable by its own parser, we need to make some changes to the output

        # only keep the AVERAGE WEIGHT OF ALL THE ITERATIONS which is reccomended in the Tuffy user manual
        sed -r '1,/\/{14}AVERAGE WEIGHT OF ALL THE ITERATIONS\/{14}/{}; /\/{14}AVERAGE WEIGHT OF ALL THE ITERATIONS\/{14}/,/\/{14}WEIGHT OF LAST ITERATION\/{14}/{}; /\/{14}WEIGHT OF LAST ITERATION\/{14}/,${d}' "${learned_prog_file}" > 'prog-avg-results.txt'

        # first copy over original prog.mln
        cp  "prog.mln" "$example_name-learned.mln"

        # The rules are not in the same order as input but there are keys
        local weights
        local rule_keys
        local rule_keys
        local i
        i=1
        weights=$(grep -o -E '^-?[0-9]+.[0-9]+' "prog-avg-results.txt")
        rule_keys=$(grep -o -E '//[0-9]+.[0-9]+$' "prog-avg-results.txt" | sed -r 's/\/\/|.[0-9]+$//g')
        for weight in $weights; do
            # incrementally set the weights in the learned file to the learned weight and write to prog-learned.mln file
            key=$(echo "$rule_keys" | head -n $i | tail -n 1)
            awk -v inc="$key" -v new_weight="${weight} " '/^-?[0-9]+.[0-9]+ |^-?[0-9]+ /{c+=1}{if(c==inc){sub(/^-?[0-9]+.[0-9]+ |^-?[0-9]+ /, new_weight, $0)};print}' "$example_name-learned.mln"> "tmp" && mv "tmp" "$example_name-learned.mln"
            i=$((i+1))
        done
    popd > /dev/null
}

function write_uniform_learned_tuffy_file() {
    local example_directory=$1
    local example_name=$2

    # write uniform weights as learned psl file
    pushd . > /dev/null
        cd "${example_directory}" || exit

#        # set the weights in the learned file to 1 and write to learned.psl file
#        sed -r "s/^[0-9]+.[0-9]+ |^[0-9]+ /1.0 /g"  "prog.mln" > "$example_name-learned.mln"

        # set the weights in the learned file to 1 and write to learned.psl file
        cp "prog.mln" "$example_name-learned.mln"
    popd > /dev/null
}

function main() {
    if [[ $# -ne 9 ]]; then
        echo "USAGE: $0 <example name> <fold> <seed> <study> <wl_method> <evaluator> <outDir> <trace_level> <model>"
        echo "USAGE: Examples can be among: ${SUPPORTED_EXAMPLES}"
        echo "USAGE: Weight Learning methods can be among: ${SUPPORTED_WL_METHODS}"
        exit 1
    fi

    trap exit SIGINT

    run_weight_learning "$@"
}

main "$@"