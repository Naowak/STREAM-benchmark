import numpy as np
from typing import Dict, Any, Callable, List, Tuple, Optional
import time
import psutil
import torch
from dataclasses import dataclass
from collections import defaultdict
import logging
from enum import Enum
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, root_mean_squared_error, precision_score, recall_score

class TaskType(Enum):
    SINGLE_SEQUENCE = "single_sequence"
    MULTI_SEQUENCE = "multi_sequence"

@dataclass
class Task:
    name: str
    generator: Callable
    task_type: TaskType
    is_classification: bool
    params: Dict[str, Any] = None

@dataclass
class TaskResult:
    task_name: str
    accuracy: float  # Pour les tâches de classification
    mse: float      # Pour les tâches de régression
    precision: Optional[float]  # Pour les tâches de classification
    recall: Optional[float]     # Pour les tâches de classification
    training_time: float
    inference_time: float
    memory_usage: float
    training_iterations: int
    sequence_length: int
    feature_dim: int

class BenchmarkSuite:
    def __init__(self, seed: int = 42):
        self.tasks = {}
        self.results = defaultdict(list)
        self.seed = seed
        np.random.seed(seed)
        self._setup_logging()
        
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def add_task(self, task: Task):
        """Ajoute une tâche au benchmark"""
        self.tasks[task.name] = task
        self.logger.info(f"Added task: {task.name}")

    def _generate_dataset(self, task: Task, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Génère un dataset d'entraînement et de test selon le type de tâche"""
        if task.task_type == TaskType.SINGLE_SEQUENCE:
            # Pour les tâches single_sequence, on génère une seule longue séquence
            X, y = task.generator(**(task.params or {}))
            train_size = int(0.8 * len(X))
            return (X[:train_size], y[:train_size]), (X[train_size:], y[train_size:])
        else:
            # Pour les tâches multi_sequence, on génère plusieurs séquences
            X_train, y_train = zip(*[task.generator(**(task.params or {})) for _ in range(n_samples)])
            X_test, y_test = zip(*[task.generator(**(task.params or {})) for _ in range(n_samples//5)])
            return (np.array(X_train), np.array(y_train)), (np.array(X_test), np.array(y_test))

    def _evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, is_classification: bool) -> Dict[str, float]:
        """Évalue les prédictions selon le type de tâche"""
        if is_classification:
            # Convert to class indices if needed
            if len(y_true.shape) > 2:
                y_true = np.argmax(y_true, axis=-1)
            if len(y_pred.shape) > 2:
                y_pred = np.argmax(y_pred, axis=-1)
                
            return {
                'accuracy': accuracy_score(y_true.flatten(), y_pred.flatten()),
                'precision': precision_score(y_true.flatten(), y_pred.flatten(), average='weighted', zero_division=0),
                'recall': recall_score(y_true.flatten(), y_pred.flatten(), average='weighted', zero_division=0),
                'mse': root_mean_squared_error(y_true, y_pred)
            }
        else:
            return {
                'accuracy': 0.0,  # Non applicable pour la régression
                'precision': None,
                'recall': None,
                'mse': root_mean_squared_error(y_true, y_pred)
            }

    def evaluate_model(self, model: Any, model_name: str, n_samples: int = 1000):
        """Évalue un modèle sur toutes les tâches enregistrées"""
        self.logger.info(f"Starting evaluation of model: {model_name}")
        
        for task_name, task in self.tasks.items():
            self.logger.info(f"Evaluating task: {task_name}")
            
            # Génération des données
            try:
                (X_train, y_train), (X_test, y_test) = self._generate_dataset(task, n_samples)
            except Exception as e:
                self.logger.error(f"Error generating dataset for task {task_name}: {e}")
                continue
                
            # Mesures de performance
            start_memory = psutil.Process().memory_info().rss
            start_time = time.time()
            
            # Entraînement
            try:
                model.train(X_train, y_train)
            except Exception as e:
                self.logger.error(f"Error training model on task {task_name}: {e}")
                continue
                
            training_time = time.time() - start_time
            memory_used = (psutil.Process().memory_info().rss - start_memory) / 1024 / 1024  # MB
            
            # Inférence
            start_time = time.time()
            try:
                predictions = model.run(X_test)
            except Exception as e:
                self.logger.error(f"Error during inference on task {task_name}: {e}")
                continue
            inference_time = time.time() - start_time
            
            # Évaluation
            metrics = self._evaluate_predictions(y_test, predictions, task.is_classification)
            
            # Enregistrement des résultats
            result = TaskResult(
                task_name=task_name,
                accuracy=metrics['accuracy'],
                mse=metrics['mse'],
                precision=metrics['precision'],
                recall=metrics['recall'],
                training_time=training_time,
                inference_time=inference_time,
                memory_usage=memory_used,
                training_iterations=getattr(model, 'n_iterations', 0),
                sequence_length=X_train.shape[1],
                feature_dim=X_train.shape[2]
            )
            
            self.results[model_name].append(result)
            self.logger.info(f"Completed evaluation for task: {task_name}")
            
        return self.results[model_name]

    def generate_report(self, output_path: str = 'benchmark_results'):
        """Génère un rapport détaillé des résultats"""
        import pandas as pd
        
        # Création du dossier de sortie
        import os
        os.makedirs(output_path, exist_ok=True)
        
        # Conversion des résultats en DataFrame
        results_data = []
        for model_name, task_results in self.results.items():
            for result in task_results:
                results_data.append({
                    'Model': model_name,
                    'Task': result.task_name,
                    'Accuracy': result.accuracy,
                    'MSE': result.mse,
                    'Precision': result.precision,
                    'Recall': result.recall,
                    'Training Time (s)': result.training_time,
                    'Inference Time (s)': result.inference_time,
                    'Memory Usage (MB)': result.memory_usage,
                    'Training Iterations': result.training_iterations,
                    'Sequence Length': result.sequence_length,
                    'Feature Dimension': result.feature_dim
                })
        
        df = pd.DataFrame(results_data)
        
        # Sauvegarde des résultats bruts
        df.to_csv(f'{output_path}/results.csv', index=False)
        
        # Génération des graphiques
        self._generate_plots(df, output_path)
        
        # Génération du rapport markdown
        with open(f'{output_path}/report.md', 'w') as f:
            f.write('# Benchmark Results\n\n')
            
            # Résumé global
            f.write('## Global Performance Summary\n\n')
            summary = df.groupby('Model').mean()
            f.write(summary.to_markdown() + '\n\n')
            
            # Détails par tâche
            for task_name in df['Task'].unique():
                f.write(f'## Task: {task_name}\n\n')
                task_df = df[df['Task'] == task_name]
                f.write(task_df.to_markdown() + '\n\n')

    def _generate_plots(self, df: 'pd.DataFrame', output_path: str):
        """Génère des visualisations des résultats"""
        # Performance plot
        plt.figure(figsize=(12, 6))
        for metric in ['Accuracy', 'MSE']:
            if df[metric].sum() > 0:  # Only plot if metric is used
                plt.subplot(1, 2, 1 if metric == 'Accuracy' else 2)
                df.boxplot(column=metric, by='Model')
                plt.title(f'{metric} by Model')
                plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_path}/performance.png')
        
        # Time and Memory plot
        plt.figure(figsize=(12, 6))
        df.boxplot(column=['Training Time (s)', 'Memory Usage (MB)'], by='Model')
        plt.title('Computational Resources by Model')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_path}/resources.png')

# Exemple d'utilisation
if __name__ == '__main__':
    # Création du benchmark
    benchmark = BenchmarkSuite()
    
    # Ajout des tâches
    from tasks import *  # Importer toutes les fonctions de génération
    
    benchmark.add_task(Task(
        name="copy_discrete",
        generator=generate_discrete_postcasting,
        task_type=TaskType.SINGLE_SEQUENCE,
        is_classification=True,
        params={"sequence_length": 1000, "delay": 10, "n_symbols": 8}
    ))
    
    benchmark.add_task(Task(
        name="adding_problem",
        generator=generate_adding_problem,
        task_type=TaskType.MULTI_SEQUENCE,
        is_classification=True,
        params={"sequence_length": 100, "max_number": 9}
    ))
    
    # Évaluation des modèles
    class DummyModel:
        def train(self, X, y):
            self.n_iterations = 100
            pass
            
        def run(self, X):
            return np.random.rand(*X.shape)
    
    model = DummyModel()
    benchmark.evaluate_model(model, "DummyModel")
    
    # Génération du rapport
    benchmark.generate_report()