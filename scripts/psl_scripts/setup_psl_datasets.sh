#!/usr/bin/env bash

# Fetch the PSL examples and modify the CLI configuration for these experiments.
# Note that you can change the version of PSL used with the PSL_VERSION option in the run inference and run wl scripts.

readonly BASE_DIR=$(realpath "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"/../..)

readonly PSL_DATASET_DIR="${BASE_DIR}/psl-datasets"

readonly AVAILABLE_MEM_KB=$(cat /proc/meminfo | grep 'MemTotal' | sed 's/^[^0-9]\+\([0-9]\+\)[^0-9]\+$/\1/')
# Floor by multiples of 5 and then reserve an additional 5 GB.
readonly JAVA_MEM_GB=$((${AVAILABLE_MEM_KB} / 1024 / 1024 / 5 * 5 - 5))
#readonly JAVA_MEM_GB=24


# Common to all examples.
function standard_fixes() {
    for exampleDir in `find ${PSL_DATASET_DIR} -maxdepth 1 -mindepth 1 -type d -not -name '.*' -not -name '_scripts'`; do
        local baseName=`basename ${exampleDir}`

        pushd . > /dev/null
            cd "${exampleDir}/cli" || exit

            # Increase memory allocation.
            sed "s/readonly JAVA_MEM_GB=.*/readonly JAVA_MEM_GB=${JAVA_MEM_GB}/" run.sh

            # Deactivate get data step
            # TODO: (Charles) For now until data construction is complete and we host the
            #       final version of the data on the linqs server
            sed -i 's/^\(\s\+\)getData/\1# getData/' run.sh

            # cp 2.3.0 snapshot into the cli directory
            cp ../../../psl_resources/psl-cli-2.3.0-SNAPSHOT.jar ./

            # Deactivate fetch psl step
            sed -i 's/^\(\s\+\)fetch_psl/\1# fetch_psl/' run.sh

        popd > /dev/null

    done
}

function main() {
   trap exit SIGINT

   standard_fixes

   exit 0
}

[[ "${BASH_SOURCE[0]}" == "${0}" ]] && main "$@"