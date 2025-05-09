#!/bin/bash
#SBATCH --job-name=T-Decoder
#SBATCH --output=logs/T-Decoder_%N_%j.out
#SBATCH --error=logs/T-Decoder_%N_%j.err
#SBATCH --nodes=3
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
module load pytorch-gpu/py3/2.3.0
set -x

# Define an array of arguments for the different runs
ARGS=(
    "--model TransformerDecoderOnly --n_trials 200 --n_seeds 10 --device cuda --tasks sin_forecasting chaotic_forecasting discrete_postcasting continue_postcasting"
    "--model TransformerDecoderOnly --n_trials 200 --n_seeds 10 --device cuda --tasks copy_task selective_copy_task discrete_pattern_completion continue_pattern_completion"
    "--model TransformerDecoderOnly --n_trials 200 --n_seeds 10 --device cuda --tasks adding_problem sorting_problem bracket_matching mnist_classification"
)

# Boucle pour lancer chaque instance du script avec les arguments correspondants
for ((i=0; i<3; i++)); do
    srun --exclusive -N1 -n1 python ../run_evaluation.py ${ARGS[i]} &
done

wait
