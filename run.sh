#!/usr/bin/env bash

# Run all the experiments.

DATASETS='movielens'

function main() {
    trap exit SIGINT

    # dataset paths to pass to scripts
    dataset_paths=''
    for dataset in $DATASETS; do
        dataset_paths="${dataset_paths}psl-datasets/${dataset} "
    done

    # PSL Experiments
    # Fetch the data and models if they are not already present and make some
    # modifactions to the run scripts and models.
    # required for both Tuffy and PSL experiments
    ./scripts/psl_scripts/setup_psl_datasets.sh

    echo "Running psl fairness experiments on datasets: [${DATASETS}]."
    pushd . > /dev/null
        cd "./scripts" || exit
        # shellcheck disable=SC2086
        ./run_psl_fairness_experiments.sh ${dataset_paths}
    popd > /dev/null

#    echo "Running mf fairness experiments on datasets: [${DATASETS}]."
#      pushd . > /dev/null
#        cd "./scripts" || exit
#        # shellcheck disable=SC2086
#        ./run_mf_fairness_experiments.sh ${dataset_paths}
#    popd > /dev/null
}

main "$@"