from src.benchmark import Benchmark, Task
from src.models.ESN import ESN
from src.models.LSTM import LSTM
from src.models.Transformers import Transformers
from src.models.TransformerDecoderOnly import TransformerDecoderOnly
from src.models.EchoStateTransformer import EchoStateTransformer
from src.evaluation import evaluation
import argparse

# List of available models & their hyperparameters
MODELS = {
    'ESN': {
        'model': ESN,
        'args': {
            "n_units": [8, 512], 
            "spectral_radius": [0., 1.], 
            "leak_rate": [0., 1.], 
        },
        'training_args': {},
    },
    'LSTM': {
        'model': LSTM,
        'args': {
            "hidden_size": [8, 256], 
            "num_layers": [1, 10], 
            "learning_rate": 1e-3
        },
        'training_args': {
            "epochs": 10, 
            "batch_size": 10
        },
    },
    'Transformers': {
        'model': Transformers,
        'args': {
            "d_model": (16, 32, 64),
            "nhead": (2, 4, 8),
            "num_encoder_layers": [1, 8],
            "num_decoder_layers": [1, 8],
            "dim_feedforward": (64, 128, 256),
            "dropout": 0.1,
            "learning_rate": 1e-3
        },
        'training_args': {
            "epochs": 10, 
            "batch_size": 10
        },
    },
    'TransformerDecoderOnly': {
        'model': TransformerDecoderOnly,
        'args': {
            "d_model": (16, 32, 64),
            "nhead": (2, 4, 8),
            "num_layers": [1, 8],
            "dim_feedforward": (64, 128, 256),
            "dropout": 0.1,
            "learning_rate": 1e-3
        },
        'training_args': {
            "epochs": 10, 
            "batch_size": 10
        },
    },
    'EchoStateTransformer': {
        'model': EchoStateTransformer,
        'args': {
            "memory_units": 4,
            "memory_dim": 50,
            "attention_dim": 3,
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
        },
        'training_args': {
            "epochs": 10, 
            "batch_size": 10
        },
    },
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ESN", choices=MODELS.keys())
    parser.add_argument("--n_trials", type=int, default=1)
    parser.add_argument("--n_seeds", type=int, default=1)
    parser.add_argument("--tasks", type=str, default="all", choices=['all', 'none'] + list(evaluation.keys()), nargs='+')
    parser.add_argument("--report", type=bool, default=True)
    parser.add_argument("--device", type=str, default="")
    return parser.parse_args()

if __name__ == "__main__":

    # Parsing des arguments
    args = parse_args()

    # Création du benchmark
    benchmark = Benchmark(
        model_class=MODELS[args.model]['model'], 
        model_name=args.model, 
        seeds=list(range(args.n_seeds))
    )

    # Args pour le modèle et l'entraînement
    model_args = MODELS[args.model]['args']
    training_args = MODELS[args.model]['training_args']

    # Add device if specified
    if args.device:
        model_args['device'] = args.device

    # Ajout des tâches
    if args.tasks == 'all':
        # Ajout de toutes les taches
        for task_name, task_params in evaluation.items():
            benchmark.add_task(Task(
                name=task_name,
                generator=task_params['generator'],
                is_classification=task_params['is_classification'],
                model_args=model_args,
                generator_params=task_params['generator_params'],
                training_args=training_args,
                n_trials=args.n_trials
            ))
    elif args.tasks == 'none':
        # Aucune tâche spécifiée
        pass
    elif type(args.tasks) == list:
        # Check if 'all' or 'none' are combined with other tasks
        if 'all' in args.tasks and len(args.tasks) > 1:
            raise ValueError("'all' cannot be combined with other tasks.")
        if 'none' in args.tasks and len(args.tasks) > 1:
            raise ValueError("'none' cannot be combined with other tasks.")
        
        # If 'all' or 'none' are specified, replace them with the corresponding list
        if 'all' in args.tasks:
            args.tasks = list(evaluation.keys())
        if 'none' in args.tasks:
            args.tasks = []

        # Ajout des tâches spécifiées
        for task_name in args.tasks:
            task_params = evaluation[task_name]
            benchmark.add_task(Task(
                name=task_name,
                generator=task_params['generator'],
                is_classification=task_params['is_classification'],
                model_args=model_args,
                generator_params=task_params['generator_params'],
                training_args=training_args,
                n_trials=args.n_trials
            ))
    else:
        raise ValueError("Invalid task name")
    
    # Evaluation du modèle
    benchmark.run()

    # Génération du rapport
    if args.report:
        benchmark.generate_report()

