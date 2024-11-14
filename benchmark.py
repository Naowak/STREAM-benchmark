from typing import Dict, Any, Callable, List, Tuple, Optional
from sklearn.metrics import accuracy_score, root_mean_squared_error, precision_score, recall_score
from dataclasses import dataclass
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import psutil
import logging
import os

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
    is_classification: bool
    mse: float                  # régression
    accuracy: float             # classification
    precision: Optional[float]  # classification
    recall: Optional[float]     # classification
    training_time: float
    inference_time: float
    memory_usage: float
    task_params: Dict[str, Any]
    model_params: Dict[str, Any]
    training_params: Dict[str, Any]
    # parameters: int

classification_task_template = """
#### Results
- Accuracy: {}
- Precision: {}
- Recall: {}
- Training Time: {}
- Inference Time: {}
- Memory Usage: {}

#### Task Parameters
{}

#### Model Parameters
{}

#### Training Parameters
{}

"""

regression_task_template = """
#### Results
- MSE: {}
- Training Time: {}
- Inference Time: {}
- Memory Usage: {}

#### Task Parameters
{}

#### Model Parameters
{}

#### Training Parameters
{}

"""

class BenchmarkSuite:
    def __init__(self, model_class: Any, model_name: str, seeds: list[int] = [42, 43, 44]):
        # Set seed
        self.seeds = seeds

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
            # Logs & Progress bar
            self.logger.info(f"Evaluating task: {task.name}")
            progress_bar = tqdm(total=task.n_trials * len(self.seeds), position=0, leave=True)
            progress_bar.set_description(f"Task: {task.name}, Seed: 0/{len(self.seeds)}, Trials: 0/{task.n_trials}")

            # Pour chaque seed
            for s, seed in enumerate(self.seeds):
                # Set seed 
                np.random.seed(seed) 
            
                # Génération des données
                try:
                    X_train, Y_train, X_test, Y_test = task.generator(**(task.generator_params or {}))
                except Exception as e:
                    self.logger.error(f"\033[91mError generating dataset for task {task.name}: {e}\033[0m")
                    continue

                # Pour chaque essai
                for i in range(task.n_trials):
                    # Randomly select model hyperparameters
                    model_hp = self._select_random_hyperparameters(task)

                    # Evaluate model with hyperparameters
                    self._evaluate_model_with_hp(X_train, Y_train, X_test, Y_test, task, model_hp)

                    # Update progress bar
                    progress_bar.update(1)
                    progress_bar.set_description(f"Task: {task.name}, Seed: {s+1}/{len(self.seeds)}, Trials: {i+1}/{task.n_trials}")

            # Logs
            progress_bar.close()
            self.logger.info(f"Completed evaluation for task: {task.name}")


    def _select_random_hyperparameters(self, task: Task) -> Dict[str, Any]:
        """Sélectionne aléatoirement des hyperparamètres pour un modèle."""
        model_hp = {}
        for hp_name, hp_values in task.model_params.items():
            if type(hp_values) in [int, float]: # Unique value, no choice
                model_hp[hp_name] = hp_values
            elif type(hp_values) == list and len(hp_values) == 2: # Range of values
                model_hp[hp_name] = int(np.random.choice(list(range(hp_values[0], hp_values[1]+1))))
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
            is_classification=task.is_classification,
            accuracy=metrics['accuracy'],
            mse=metrics['mse'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            training_time=training_time,
            inference_time=inference_time,
            memory_usage=memory_used,
            task_params=task.generator_params,
            model_params=model_hp,
            training_params=task.training_params,
        )
        self.results[task.name].append(result)


    def _evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, is_classification: bool) -> Dict[str, float]:
        """Évalue les prédictions selon le type de tâche"""
        if is_classification:
            # Convert to class indices if needed
            y_true = (y_true > 0.5).astype(int)
            y_pred = (y_pred > 0.5).astype(int)
                
            return {
                'accuracy': accuracy_score(y_true.flatten(), y_pred.flatten()),
                'precision': precision_score(y_true.flatten(), y_pred.flatten(), average='weighted', zero_division=0),
                'recall': recall_score(y_true.flatten(), y_pred.flatten(), average='weighted', zero_division=0),
                'mse': None  # Non applicable pour la classification
            }
        else:
            return {
                'accuracy': None,  # Non applicable pour la régression
                'precision': None,
                'recall': None,
                'mse': root_mean_squared_error(y_true.flatten(), y_pred.flatten())
            }
        

    def generate_report(self, output_path: str='benchmark_results'):
        """Génère un rapport détaillé des résultats.
        aggregate: 'mean', 'median' or 'max'"""
        
        # Création du dossier de sortie
        os.makedirs(output_path, exist_ok=True)
        
        # Conversion des résultats en DataFrame
        results_data = []
        for task_results in self.results.values():
            for result in task_results:
                results_data.append({
                    'Model': self.model_name,
                    'Task': result.task_name,
                    'Accuracy': result.accuracy,
                    'MSE': result.mse,
                    'Precision': result.precision,
                    'Recall': result.recall,
                    'Training Time (s)': result.training_time,
                    'Inference Time (s)': result.inference_time,
                    'Memory Usage (MB)': result.memory_usage,
                    'Task Params': result.task_params,
                    'Model Params': result.model_params,
                    'Training Params': result.training_params,
                })
        
        df = pd.DataFrame(results_data)
        
        # Sauvegarde des résultats bruts
        df.to_csv(f'{output_path}/results.csv', index=False)
        
        # Génération des graphiques
        #self._generate_plots(df, output_path)
        
        # Génération du rapport markdown
        with open(f'{output_path}/report.md', 'w') as f:
            f.write('# Benchmark Results\n\n')
            
            # Résumé global
            f.write('## Global Performance Summary\n\n')

            # Extrait le meilleur essai pour chaque tâches
            dfnp = df.drop(['Task Params', 'Model Params', 'Training Params'], axis=1)
            groups = dfnp.groupby(['Task', 'Model'])
            best_idx_acc = groups['Accuracy'].idxmax()
            best_idx_mse = groups['MSE'].idxmin()
            best_idx = pd.concat([best_idx_acc, best_idx_mse]).dropna()

            # Create summary dataframe
            summary = dfnp[dfnp.index.isin(best_idx)]
            f.write(summary.to_markdown() + '\n\n')
            
            # Détails par tâche
            best = df[df.index.isin(best_idx)]
            for task_name in summary['Task']:
                f.write(f'## Task: {task_name}')
                task_df = best[best['Task'] == task_name].iloc[0]

                # Classification task
                if not np.isnan(task_df['Accuracy']):
                    f.write(classification_task_template.format(
                        task_df['Accuracy'], 
                        task_df['Precision'], 
                        task_df['Recall'], 
                        task_df['Training Time (s)'], 
                        task_df['Inference Time (s)'], 
                        task_df['Memory Usage (MB)'], 
                        task_df['Task Params'], 
                        task_df['Model Params'], 
                        task_df['Training Params']
                    ))
                
                # Régression task
                elif not np.isnan(task_df['MSE']):
                    f.write(regression_task_template.format(
                        task_df['MSE'],
                        task_df['Training Time (s)'],
                        task_df['Inference Time (s)'],
                        task_df['Memory Usage (MB)'], 
                        task_df['Task Params'], 
                        task_df['Model Params'], 
                        task_df['Training Params']
                    ))
                else:
                    raise ValueError('Invalid task type')

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

