from src.benchmark import Benchmark, Task
from src.tasks import *
from models.ESN import ESN as RC


NB_SEEDS = 2
N_TRIALS = 25

if __name__ == "__main__":

    # Création du benchmark
    seeds = list(range(NB_SEEDS))
    benchmark = Benchmark(model_class=RC, model_name="ESN-test", seeds=seeds)
    n_trials = N_TRIALS
    model_args = {"n_units": [10, 500], "spectral_radius": [0., 1.], "leak_rate": [0., 1.], "ridge": 1e-6}

    # Ajout des tâches
    # benchmark.add_task(Task(
    #     name="discrete_postcasting",
    #     generator=generate_discrete_postcasting,
    #     is_classification=True,
    #     model_args=model_args,
    #     generator_params={"sequence_length": 1000, "delay": 10, "n_symbols": 30},
    #     n_trials=n_trials
    # ))

    benchmark.add_task(Task(
        name="continue_postcasting",
        generator=generate_continue_postcasting,
        is_classification=False,
        model_args=model_args,
        generator_params={"sequence_length": 1000, "delay": 10},
        n_trials=n_trials
    ))

    # benchmark.add_task(Task(
    #     name="copy_task",
    #     generator=generate_copy_task,
    #     is_classification=True,
    #     model_args=model_args,
    #     generator_params={"sequence_length": 100, "delay": 10, "n_symbols": 10},
    #     n_trials=n_trials
    # ))

    # benchmark.add_task(Task(
    #     name="selective_copy_task",
    #     generator=generate_selective_copy_task,
    #     is_classification=True,
    #     model_args=model_args,
    #     generator_params={"sequence_length": 100, "delay": 10, "n_symbols": 10},
    #     n_trials=n_trials
    # ))

    # benchmark.add_task(Task(
    #     name="adding_problem",
    #     generator=generate_adding_problem,
    #     is_classification=True,
    #     model_args=model_args,
    #     generator_params={"sequence_length": 50, "max_number": 20},
    #     n_trials=n_trials
    # ))

    # benchmark.add_task(Task(
    #     name="sorting_problem",
    #     generator=generate_sorting_problem,
    #     is_classification=True,
    #     model_args=model_args,
    #     generator_params={"sequence_length": 50, "n_symbols": 10},
    #     n_trials=n_trials
    # ))

    # # benchmark.add_task(Task(
    # #     name="mnist_classification",
    # #     generator=generate_mnist_classification,
    # #     is_classification=True,
    # #     model_args=model_args,
    # #     generator_params={"n_samples": 1000},
    # #     n_trials=n_trials
    # # ))

    # benchmark.add_task(Task(
    #     name="discrete_pattern_completion",
    #     generator=generate_discrete_pattern_completion,
    #     is_classification=True,
    #     model_args=model_args,
    #     generator_params={"sequence_length": 100, "n_symbols": 12, "base_length": 20},
    #     n_trials=n_trials
    # ))

    # benchmark.add_task(Task(
    #     name="continuous_pattern_completion",
    #     generator=generate_continuous_pattern_completion,
    #     is_classification=False,
    #     model_args=model_args,
    #     generator_params={"sequence_length": 100, "base_length": 10},
    #     n_trials=n_trials
    # ))

    # benchmark.add_task(Task(
    #     name="bracket_matching",
    #     generator=generate_bracket_matching,
    #     is_classification=True,
    #     model_args=model_args,
    #     generator_params={"sequence_length": 100, "max_depth": 10},
    #     n_trials=n_trials
    # ))

    # benchmark.add_task(Task(
    #     name="sin_forecasting",
    #     generator=generate_sin_forecasting,
    #     is_classification=False,
    #     model_args=model_args,
    #     generator_params={"sequence_length": 1000, "forecast_length": 10},
    #     n_trials=n_trials
    # ))

    # benchmark.add_task(Task(
    #     name="chaotic_forecasting",
    #     generator=generate_chaotic_forecasting,
    #     is_classification=False,
    #     model_args=model_args,
    #     generator_params={"sequence_length": 1000, "forecast_length": 10},
    #     n_trials=n_trials
    # ))

    # Evaluation du modèle
    benchmark.run()

    # Génération du rapport
    benchmark.generate_report()