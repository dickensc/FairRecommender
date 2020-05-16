#!/usr/bin/env bash

# Run all the experiments.

PSL_DATASETS='movielens'
TUFFY_DATASETS='movielens'

function main() {
    trap exit SIGINT

    # dataset paths to pass to scripts
    psl_dataset_paths=''
    tuffy_dataset_paths=''
    for dataset in $PSL_DATASETS; do
        psl_dataset_paths="${psl_dataset_paths}psl-datasets/${dataset} "
    done

    for dataset in $TUFFY_DATASETS; do
        tuffy_dataset_paths="${tuffy_dataset_paths}tuffy-datasets/${dataset} "
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

     # Tuffy Experiments
     # Initialize Tuffy environment
     # shellcheck disable=SC2086
     ./scripts/tuffy_scripts/tuffy_init.sh ${tuffy_dataset_paths}

     # Convert psl formatted data into tuffy formatted data
     # shellcheck disable=SC2086
     ./scripts/tuffy_scripts/tuffy_convert.sh ${tuffy_dataset_paths}

     # run tuffy performance experiments
     echo "Running tuffy performance experiments on datasets: [${TUFFY_DATASETS}]."
     pushd . > /dev/null
         cd "./scripts" || exit
         # shellcheck disable=SC2086
         ./run_weight_learning_performance_experiments.sh "tuffy" ${tuffy_dataset_paths}
     popd > /dev/null

    echo "Running tuffy robustness experiments on datasets: [${TUFFY_DATASETS}]."
    # shellcheck disable=SC2086
    pushd . > /dev/null
        cd "./scripts" || exit
        # shellcheck disable=SC2086
        ./run_weight_learning_robustness_experiments.sh "tuffy" ${tuffy_dataset_paths}
    popd > /dev/null

    echo "Running tuffy sampling experiments on datasets: [${TUFFY_DATASETS}]."
    # shellcheck disable=SC2086
    pushd . > /dev/null
        cd "./scripts" || exit
        # shellcheck disable=SC2086
        ./run_weight_learning_sampling_experiments.sh "tuffy" ${tuffy_dataset_paths}
    popd > /dev/null
}

main "$@"