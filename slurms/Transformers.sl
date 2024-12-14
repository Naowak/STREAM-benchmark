#!/bin/bash
#SBATCH --job-name=Transformers
#SBATCH --output=logs/Transformers_%N_%j.out
#SBATCH --error=logs/Transformers_%N_%j.err
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -C a100
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread
#SBATCH --account=rvc@a100

# Charger les modules nécessaires
module purge
module load arch/a100
module load miniforge/24.9.0
conda activate stream_venv
set -x

# Define an array of arguments for the different runs
ARGS=(
    "--model Transformers --n_trials 200 --n_seeds 10 --device cuda --tasks sin_forecasting chaotic_forecasting discrete_postcasting continue_postcasting"
    "--model Transformers --n_trials 200 --n_seeds 10 --device cuda --tasks copy_task"
    "--model Transformers --n_trials 200 --n_seeds 10 --device cuda --tasks selective_copy_task"
    "--model Transformers --n_trials 200 --n_seeds 10 --device cuda --tasks discrete_pattern_completion continue_pattern_completion"
    "--model Transformers --n_trials 200 --n_seeds 10 --device cuda --tasks mnist_classification"
    "--model Transformers --n_trials 200 --n_seeds 10 --device cuda --tasks adding_problem"
    "--model Transformers --n_trials 200 --n_seeds 10 --device cuda --tasks sorting_problem"
    "--model Transformers --n_trials 200 --n_seeds 10 --device cuda --tasks bracket_matching"
)

# Boucle pour lancer chaque instance du script avec les arguments correspondants
for ((i=0; i<8; i++)); do
    srun --exclusive -N1 -n1 python ../run_evaluation.py ${ARGS[i]} &
done

wait
