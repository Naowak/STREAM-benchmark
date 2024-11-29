from src.benchmark import Benchmark, Task
from src.models.ESN import ESN
from src.evaluation import evaluation

NB_SEEDS = 1
N_TRIALS = 1

if __name__ == "__main__":

    # Création du benchmark
    benchmark = Benchmark(
        model_class=ESN, 
        model_name="ESN", 
        seeds=list(range(NB_SEEDS))
    )

    # Args pour le modèle et l'entraînement
    model_args = {
        "n_units": [8, 512], 
        "spectral_radius": [0., 1.], 
        "leak_rate": [0., 1.], 
    }
    training_args = {}

        # Ajout des tâches
    for task_name, task_params in evaluation.items():
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