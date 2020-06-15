#!/usr/bin/env bash

# Run all the experiments.

PSL_DATASETS='movielens'
TUFFY_DATASETS='movielens movielens_non_parity movielens_value'

function main() {
    trap exit SIGINT

    # dataset paths to pass to scripts
    psl_dataset_paths=''
    for dataset in $PSL_DATASETS; do
        psl_dataset_paths="${psl_dataset_paths}psl-datasets/${dataset} "
    done

    # PSL Experiments
    # Fetch the data and models if they are not already present and make some
    # modifactions to the run scripts and models.
    # required for both Tuffy and PSL experiments
    ./scripts/psl_scripts/setup_psl_datasets.sh

    echo "Running psl fairness experiments on datasets: [${PSL_DATASETS}]."
    pushd . > /dev/null
        cd "./scripts" || exit
        # shellcheck disable=SC2086
        ./run_fairness_experiments.sh "psl" ${psl_dataset_paths}
    popd > /dev/null
}

main "$@"