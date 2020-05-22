#!/usr/bin/env bash

# Fetch the PSL examples and modify the CLI configuration for these experiments.
# Note that you can change the version of PSL used with the PSL_VERSION option in the run inference and run wl scripts.

readonly BASE_DIR=$(realpath "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"/../..)

readonly PSL_DATASET_DIR="${BASE_DIR}/psl-datasets"

#readonly AVAILABLE_MEM_KB=$(cat /proc/meminfo | grep 'MemTotal' | sed 's/^[^0-9]\+\([0-9]\+\)[^0-9]\+$/\1/')
## Floor by multiples of 5 and then reserve an additional 5 GB.
#readonly JAVA_MEM_GB=$((${AVAILABLE_MEM_KB} / 1024 / 1024 / 5 * 5 - 5))
readonly JAVA_MEM_GB=24


# Common to all examples.
function standard_fixes() {
    for exampleDir in `find ${PSL_DATASET_DIR} -maxdepth 1 -mindepth 1 -type d -not -name '.*' -not -name '_scripts'`; do
        local baseName=`basename ${exampleDir}`

        pushd . > /dev/null
            cd "${exampleDir}/cli" || exit

            # Increase memory allocation.
            sed -i "s/java -jar/java -Xmx${JAVA_MEM_GB}G -Xms${JAVA_MEM_GB}G -jar/" run.sh

            # Deactivate get data step
            # TODO: (Charles) For now until data construction is complete and we host the
            #       final version of the data on the linqs server
            sed -i 's/^\(\s\+\)getData/\1# getData/' run.sh

            # Deactivate weight learning psl step
            # TODO: (Charles) For now until we make splits
            sed -i 's/^\(\s\+\)runWeightLearning/\1# runWeightLearning/' run.sh

        popd > /dev/null

    done
}

function main() {
   trap exit SIGINT

   standard_fixes

   exit 0
}

[[ "${BASH_SOURCE[0]}" == "${0}" ]] && main "$@"