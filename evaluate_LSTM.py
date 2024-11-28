from src.benchmark import Benchmark, Task
from src.tasks import *
from src.models.LSTM import LSTM

NB_SEEDS = 1
N_TRIALS = 1

tasks = {
    'discrete_postcasting': {
        'generator': generate_discrete_postcasting,
        'is_classification': True,
        'generator_params': {"sequence_length": 1000, "delay": 100, "n_symbols": 30},
    },
    'continue_postcasting': {
        'generator': generate_continue_postcasting,
        'is_classification': False,
        'generator_params': {"sequence_length": 1000, "delay": 100},
    },
    'copy_task': {
        'generator': generate_copy_task,
        'is_classification': True,
        'generator_params': {"n_samples": 1000, "sequence_length": 100, "delay": 10, "n_symbols": 10},
    },
    'selective_copy_task': {
        'generator': generate_selective_copy_task,
        'is_classification': True,
        'generator_params': {"n_samples": 1000, "sequence_length": 100, "delay": 10, "n_markers": 20, "n_symbols": 10},
    },
    'adding_problem': {
        'generator': generate_adding_problem,
        'is_classification': True,
        'generator_params': {"n_samples": 1000, "sequence_length": 100, "max_number": 20},
    },
    'sorting_problem': {
        'generator': generate_sorting_problem,
        'is_classification': True,
        'generator_params': {"n_samples": 1000, "sequence_length": 50, "n_symbols": 10},
    },
    'mnist_classification': {
        'generator': generate_mnist_classification,
        'is_classification': True,
        'generator_params': {"n_samples": 1000, "path": "datasets/mnist"},
    },
    'discrete_pattern_completion': {
        'generator': generate_discrete_pattern_completion,
        'is_classification': True,
        'generator_params': {"n_samples": 1000, "sequence_length": 100, "n_symbols": 12, "base_length": 20, "mask_ratio": 0.2},
    },
    'continuous_pattern_completion': {
        'generator': generate_continuous_pattern_completion,
        'is_classification': False,
        'generator_params': {"n_samples": 1000, "sequence_length": 100, "base_length": 10, "mask_ratio": 0.2},
    },
    'bracket_matching': {
        'generator': generate_bracket_matching,
        'is_classification': True,
        'generator_params': {"n_samples": 1000, "sequence_length": 200, "max_depth": 20},
    },
    'sin_forecasting': {
        'generator': generate_sin_forecasting,
        'is_classification': False,
        'generator_params': {"sequence_length": 1000, "forecast_length": 10},
    },
    'chaotic_forecasting': {
        'generator': generate_chaotic_forecasting,
        'is_classification': False,
        'generator_params': {"sequence_length": 1000, "forecast_length": 10},
    },
}

if __name__ == "__main__":

    # Création du benchmark
    benchmark = Benchmark(
        model_class=LSTM, 
        model_name="LSTM", 
        seeds=list(range(NB_SEEDS))
    )

    # Args pour le modèle et l'entraînement
    model_args = {
        "hidden_size": [8, 256], 
        "num_layers": [1, 10], 
        "learning_rate": 1e-3
    }
    training_args = {
        "epochs": 10, 
        "batch_size": 10
    }

    # Ajout des tâches
    for task_name, task_params in tasks.items():
        benchmark.add_task(Task(
            name=task_name,
            generator=task_params['generator'],
            is_classification=task_params['is_classification'],
            model_args=model_args,
            generator_params=task_params['generator_params'],
            training_args=training_args,
            n_trials=N_TRIALS
        ))
    
    # Evaluation du modèle
    benchmark.run()

    # Génération du rapport
    benchmark.generate_report()