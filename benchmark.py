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

@dataclass
class Task:
    name: str
    generator: Callable
    is_classification: bool
    generator_params: Dict[str, Any] = None
    model_params: Dict[str, Any] = None
    training_params: Dict[str, Any] = None
    n_trials: int = 1

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
    def __init__(self, model_class: Any, model_name: str, seed: int = 42):
        # Set seed
        self.seed = seed

        # Initialize model_class, tasks and results
        self.model_class = model_class
        self.model_name = model_name
        self.tasks = {}
        self.results = defaultdict(list)

        # Initialize logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        

    def add_task(self, task: Task):
        """Ajoute une tâche au benchmark."""
        self.tasks[task.name] = task
        self.logger.info(f"Added task: {task.name}")


    def run(self):
        """Évalue un modèle sur toutes les tâches enregistrées."""
        self.logger.info(f"Starting evaluation of model: {self.model_name}")
        
        # Pour chaque tâche
        for task in self.tasks.values():
            self.logger.info(f"Evaluating task: {task.name}")
            
            # Génération des données
            try:
                np.random.seed(self.seed) # Reset seed 
                X_train, Y_train, X_test, Y_test = task.generator(**(task.generator_params or {}))
            except Exception as e:
                self.logger.error(f"\033[91mError generating dataset for task {task.name}: {e}\033[0m")
                continue

            # Entraînement et évaluation du modèle pour un ensemble d'hyperparamètres aléatoires
            for i in range(task.n_trials):

                # Randomly select model hyperparameters
                model_hp = self._select_random_hyperparameters(task)
                print(model_hp)

                # Evaluate model with hyperparameters
                self._evaluate_model_with_hp(X_train, Y_train, X_test, Y_test, task, model_hp)



    def _select_random_hyperparameters(self, task: Task) -> Dict[str, Any]:
        """Sélectionne aléatoirement des hyperparamètres pour un modèle."""
        model_hp = {}
        for hp_name, hp_values in task.model_params.items():
            if type(hp_values) in [int, float]: # Unique value, no choice
                model_hp[hp_name] = hp_values
            elif type(hp_values) == list and len(hp_values) == 2: # Range of values
                model_hp[hp_name] = np.random.choice(list(range(hp_values[0], hp_values[1]+1)))
            else: # Error
                raise ValueError(f"Invalid hyperparameter values for task {task.name}: {hp_values}")
        return model_hp

    def _evaluate_model_with_hp(self, X_train, Y_train, X_test, Y_test, task, model_hp):
        """Pour chaque tâche du benchmark, évalue un modèle en effectuant une recherche d'hyperparamètres."""

        # Retrieve task & Initialisation des mesures de performance
        start_memory = psutil.Process().memory_info().rss
        start_time = time.time()
        
        # Entraînement
        try:
            model = self.model_class(**model_hp)
            model.train(X_train, Y_train)
        except Exception as e:
            self.logger.error(f"\033[91mError training model on task {task.name}: {e}\033[0m")
            return
        
        # Mesure du temps et de la mémoire pour l'entrainement
        training_time = time.time() - start_time
        memory_used = (psutil.Process().memory_info().rss - start_memory) / 1024 / 1024  # MB
        
        # Prédiction des données de test
        start_time = time.time()
        try:
            predictions = model.run(X_test)
        except Exception as e:
            self.logger.error(f"\033[91mError during inference on task {task.name}: {e}\033[0m")
            return
        inference_time = time.time() - start_time
        
        # Evaluation des prédictions
        metrics = self._evaluate_predictions(Y_test, predictions, task.is_classification)
        
        # Enregistrement des résultats
        result = TaskResult(
            task_name=task.name,
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
        
        self.results[self.model_name].append(result)
        self.logger.info(f"Completed evaluation for task: {task.name}")



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





    # def generate_report(self, output_path: str = 'benchmark_results'):
    #     """Génère un rapport détaillé des résultats"""
    #     import pandas as pd
        
    #     # Création du dossier de sortie
    #     import os
    #     os.makedirs(output_path, exist_ok=True)
        
    #     # Conversion des résultats en DataFrame
    #     results_data = []
    #     for model_name, task_results in self.results.items():
    #         for result in task_results:
    #             results_data.append({
    #                 'Model': model_name,
    #                 'Task': result.task_name,
    #                 'Accuracy': result.accuracy,
    #                 'MSE': result.mse,
    #                 'Precision': result.precision,
    #                 'Recall': result.recall,
    #                 'Training Time (s)': result.training_time,
    #                 'Inference Time (s)': result.inference_time,
    #                 'Memory Usage (MB)': result.memory_usage,
    #                 'Training Iterations': result.training_iterations,
    #                 'Sequence Length': result.sequence_length,
    #                 'Feature Dimension': result.feature_dim
    #             })
        
    #     df = pd.DataFrame(results_data)
        
    #     # Sauvegarde des résultats bruts
    #     df.to_csv(f'{output_path}/results.csv', index=False)
        
    #     # Génération des graphiques
    #     self._generate_plots(df, output_path)
        
    #     # Génération du rapport markdown
    #     with open(f'{output_path}/report.md', 'w') as f:
    #         f.write('# Benchmark Results\n\n')
            
    #         # Résumé global
    #         f.write('## Global Performance Summary\n\n')
    #         summary = df.groupby('Model').mean()
    #         f.write(summary.to_markdown() + '\n\n')
            
    #         # Détails par tâche
    #         for task_name in df['Task'].unique():
    #             f.write(f'## Task: {task_name}\n\n')
    #             task_df = df[df['Task'] == task_name]
    #             f.write(task_df.to_markdown() + '\n\n')

    # def _generate_plots(self, df: 'pd.DataFrame', output_path: str):
    #     """Génère des visualisations des résultats"""
    #     # Performance plot
    #     plt.figure(figsize=(12, 6))
    #     for metric in ['Accuracy', 'MSE']:
    #         if df[metric].sum() > 0:  # Only plot if metric is used
    #             plt.subplot(1, 2, 1 if metric == 'Accuracy' else 2)
    #             df.boxplot(column=metric, by='Model')
    #             plt.title(f'{metric} by Model')
    #             plt.xticks(rotation=45)
    #     plt.tight_layout()
    #     plt.savefig(f'{output_path}/performance.png')
        
    #     # Time and Memory plot
    #     plt.figure(figsize=(12, 6))
    #     df.boxplot(column=['Training Time (s)', 'Memory Usage (MB)'], by='Model')
    #     plt.title('Computational Resources by Model')
    #     plt.xticks(rotation=45)
    #     plt.tight_layout()
    #     plt.savefig(f'{output_path}/resources.png')

