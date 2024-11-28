from src.benchmark import Benchmark, Task
from src.models.TransformerDecoderOnly import TransformerDecoderOnly
from src.evaluation import evaluation

NB_SEEDS = 1
N_TRIALS = 1

if __name__ == "__main__":

    # Création du benchmark
    benchmark = Benchmark(
        model_class=TransformerDecoderOnly, 
        model_name="TransformersDecoderOnly", 
        seeds=list(range(NB_SEEDS))
    )

    # Args pour le modèle et l'entraînement
    model_args = {
        "d_model": (16, 32, 64),
        "nhead": (2, 4, 8),
        "num_layers": [1, 8],
        "dim_feedforward": (64, 128, 256),
        "dropout": 0.1,
        "learning_rate": 1e-3
    }
    training_args = {
        "epochs": 10, 
        "batch_size": 10, 
    }

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