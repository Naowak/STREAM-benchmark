from src.tasks import *

evaluation = {
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