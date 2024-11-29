from typing import Dict, Any, Callable, List, Tuple, Optional
from sklearn.metrics import accuracy_score, root_mean_squared_error, precision_score, recall_score
from src.report_templates import classification_task_template, regression_task_template
from dataclasses import dataclass
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
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
    model_args: Dict[str, Any] = None
    training_args: Dict[str, Any] = None
    n_trials: int = 1

@dataclass
class TaskResult:
    task_name: str
    is_classification: bool
    mse: float                  # régression
    accuracy: float             # classification
    precision: Optional[float]  # classification
    recall: Optional[float]     # classification
    bic: float
    training_time: float
    inference_time: float
    memory_usage: float
    task_args: Dict[str, Any]
    model_args: Dict[str, Any]
    training_args: Dict[str, Any]
    number_params: int

class Benchmark:
    def __init__(self, model_class: Any, model_name: str, seeds: list[int] = [42, 43, 44, 45]):
        """Initialise un processus de benchmark pour un modèle.
        
        Paramètres :
        - model_class : Classe du modèle à évaluer.
        - model_name (str) : Nom du modèle.
        - seeds (list[int]) : Liste des graines aléatoires à utiliser.
        """
        # Set seed
        self.seeds = seeds

        # Initialize model_class, tasks and results
        self.model_class = model_class
        self.model_name = model_name
        self.tasks = {}
        self.results = defaultdict(list)
        
        # Check model class has count params method
        if 'count_params' not in dir(model_class):
            raise ValueError('Model class must have a count_params method')

        # Initialize logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized benchmark process for model: {model_name}")

    def add_task(self, task: Task):
        """Ajoute une tâche au benchmark.
        
        Paramètres :
        - task : Tâche à ajouter.
        """
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
            progress_bar.set_description(f"Task: {task.name}, Seed: 0/{len(self.seeds)}, Trial: 0/{task.n_trials}")

            # Pour chaque seed
            for s, seed in enumerate(self.seeds):
                # Set seed 
                np.random.seed(seed) 
            
                # Génération des données
                try:
                    X_train, Y_train, X_test, Y_test, prediction_start = task.generator(**(task.generator_params or {}))
                except Exception as e:
                    self.logger.error(f"\033[91mError generating dataset for task {task.name}: {e}\033[0m")
                    continue

                # Pour chaque essai
                for i in range(task.n_trials):
                    # Randomly select model hyperparameters
                    model_hp = self._select_random_hyperparameters(task)

                    # Evaluate model with hyperparameters
                    self._evaluate_model_with_hp(X_train, Y_train, X_test, Y_test, prediction_start, task, model_hp)

                    # Update progress bar
                    progress_bar.update(1)
                    progress_bar.set_description(f"Task: {task.name}, Seed: {s+1}/{len(self.seeds)}, Trial: {i+1}/{task.n_trials}")

            # Logs
            progress_bar.close()
            self.logger.info(f"Completed evaluation for task: {task.name}")

    def generate_report(self, output_path: str=''):
        """Génère un rapport détaillé des résultats.
        
        Paramètres :
        - output_path (str) : Chemin de sortie pour les résultats.
        """
        self.logger.info(f"Generating report & plots for model: {self.model_name}")

        if not output_path:
            output_path = f'./results/{self.model_name}'
        
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
                    'BIC': result.bic,
                    'Training Time (s)': result.training_time,
                    'Inference Time (s)': result.inference_time,
                    'Memory Usage (MB)': result.memory_usage,
                    'Task Args': result.task_args,
                    'Model Args': result.model_args,
                    'Training Args': result.training_args,
                    'Number Params': result.number_params
                })
    
        df = pd.DataFrame(results_data)
        
        # Sauvegarde des résultats bruts
        df.to_csv(f'{output_path}/results.csv', index=False)
        
        # Génération des graphiques
        self._generate_plots(df, output_path)
        
        # Génération du rapport markdown
        with open(f'{output_path}/report.md', 'w') as f:
            f.write(f'# Benchmark Results for model {self.model_name}\n\n')
            
            # Résumé global
            f.write('## Global Performance Summary\n\n')

            # Create summary dataframe
            best = self._extract_best_models(df)
            f.write(best.round(4).to_markdown() + '\n\n')
            
            # Détails par tâche
            for task_name in best['Task']:
                f.write(f'## Task: {task_name}')
                task_df = best[best['Task'] == task_name].iloc[0]
                is_classification = self._is_classification_task(task_name)

                # Classification task
                if is_classification:
                    f.write(classification_task_template.format(
                        task_df['Accuracy'], 
                        task_df['Precision'], 
                        task_df['Recall'], 
                        task_df['Training Time (s)'], 
                        task_df['Inference Time (s)'], 
                        task_df['Memory Usage (MB)'], 
                        task_df['Number Params'],
                        task_df['Task Args'], 
                        task_df['Model Args'], 
                        task_df['Training Args'],
                        task_name,
                    ))
                
                # Régression task
                else:
                    f.write(regression_task_template.format(
                        task_df['MSE'],
                        task_df['Training Time (s)'],
                        task_df['Inference Time (s)'],
                        task_df['Memory Usage (MB)'], 
                        task_df['Number Params'],
                        task_df['Task Args'], 
                        task_df['Model Args'], 
                        task_df['Training Args'],
                        task_name,
                    ))
        
        self.logger.info(f"Report & plots generated at {output_path}")

    def _select_random_hyperparameters(self, task: Task) -> Dict[str, Any]:
        """Sélectionne aléatoirement des hyperparamètres pour un modèle.
        
        Paramètres :
        - task : Tâche pour laquelle les hyperparamètres doivent être sélectionnés.
        
        Retourne :
        - Dict[str, Any] : Dictionnaire d'hyperparamètres sélectionnés.
        """
        model_hp = {}
        for hp_name, hp_values in task.model_args.items():
            if type(hp_values) in [int, float]: # Unique value, no choice
                model_hp[hp_name] = hp_values
            elif type(hp_values) == list and len(hp_values) == 2: # Range of values
                if type(hp_values[0]) == float:
                    model_hp[hp_name] = np.random.uniform(hp_values[0], hp_values[1])
                elif type(hp_values[0]) == int:
                    model_hp[hp_name] = int(np.random.choice(list(range(hp_values[0], hp_values[1]+1))))
                else:
                    raise ValueError(f"Invalid hyperparameter values for task {task.name}: {hp_values}")
            elif type(hp_values) == tuple: # Choice of values
                model_hp[hp_name] = np.random.choice(hp_values)
            else: # Error
                raise ValueError(f"Invalid hyperparameter values for task {task.name}: {hp_values}")
        return model_hp

    def _evaluate_model_with_hp(self, X_train, Y_train, X_test, Y_test, prediction_start, task, model_hp):
        """Pour chaque tâche du benchmark, évalue un modèle en effectuant une recherche d'hyperparamètres.
        
        Paramètres :
        - X_train : Données d'entraînement.
        - Y_train : Données de sortie d'entraînement.
        - X_test : Données de test.
        - Y_test : Données de sortie de test.
        - prediction_start : Indice de début de prédiction.
        - task : Tâche à évaluer.
        - model_hp : Hyperparamètres du modèle.
        """

        # Retrieve task & Initialisation des mesures de performance
        start_memory = psutil.Process().memory_info().rss
        start_time = time.time()
        
        # Entraînement
        try:
            model = self.model_class(**model_hp)
            model.train(X_train, Y_train, classification=task.is_classification, prediction_start=prediction_start)
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
        metrics = self._evaluate_predictions(Y_test, predictions, prediction_start, model.count_params(), task.is_classification)
        
        # Enregistrement des résultats
        result = TaskResult(
            task_name=task.name,
            is_classification=task.is_classification,
            **metrics, # accuracy, precision, recall, mse, bic
            training_time=training_time,
            inference_time=inference_time,
            memory_usage=memory_used,
            task_args=task.generator_params,
            model_args=model_hp,
            training_args=task.training_args,
            number_params=model.count_params()
        )
        self.results[task.name].append(result)


    def _evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, prediction_start: int, n_params: int, is_classification: bool) -> Dict[str, float]:
        """Évalue les prédictions selon le type de tâche.
        
        Paramètres :
        - y_true : Données de sortie réelles.
        - y_pred : Données de sortie prédites.
        - prediction_start : Indice de début de prédiction.
        - n_params : Nombre de paramètres du modèle.
        - is_classification : Indique si la tâche est une classification.
        
        Retourne :
        - Dict[str, float] : Mesures de performance.
        """
        # Flatten arrays
        y_true = np.array(y_true)[:, prediction_start:, :].flatten()
        y_pred = np.array(y_pred)[:, prediction_start:, :].flatten()

        if is_classification:
            # Convert to class indices if needed
            y_true_bool = (y_true > 0.5).astype(int)
            y_pred_bool = (y_pred > 0.5).astype(int)
                
            return {
                'accuracy': accuracy_score(y_true_bool, y_pred_bool),
                'precision': precision_score(y_true_bool, y_pred_bool, average='weighted', zero_division=0),
                'recall': recall_score(y_true_bool, y_pred_bool, average='weighted', zero_division=0),
                'mse': None,  # Non applicable pour la classification
                'bic': self._compute_bic(y_true, y_pred, n_params, is_classification)
            }
        else:
            return {
                'accuracy': None,  # Non applicable pour la régression
                'precision': None,
                'recall': None,
                'mse': root_mean_squared_error(y_true, y_pred),
                'bic': self._compute_bic(y_true, y_pred, n_params, is_classification)
            }
        

    def _compute_bic(self, y_true: np.ndarray, y_pred: np.ndarray, n_params: int, is_classification: bool, epsilon: float=1e-7) -> float:
        """Calcule le critère d'information bayésien.
        
        Paramètres :
        - y_true : Données de sortie réelles.
        - y_pred : Données de sortie prédites.
        - n_params : Nombre de paramètres du modèle.
        - is_classification : Indique si la tâche est une classification.
        - epsilon : Valeur pour éviter les erreurs de calcul.
        
        Retourne :
        - float : Critère d'information bayésien (BIC).
        """
        # Get number of samples
        n_sample = y_true.shape[0]

        # Compute log likelihood
        if is_classification:
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon) # Avoid log(0)
            loglikelihood = np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:
            # Compute BIC for regression
            rss = np.sum((y_true - y_pred) ** 2)  # Residual Sum of Squares
            loglikelihood = -n_sample / 2 * np.log(rss / n_sample)  # Gaussian likelihood

        # Compute BIC and return it
        bic = -2 * loglikelihood + n_params * np.log(n_sample)
        return bic
    
    def _extract_best_models(self, df: 'pd.DataFrame') -> 'pd.DataFrame':
        """Extrait le meilleur modèle pour chaque tâche.
        
        Paramètres :
        - df : DataFrame des résultats.
        
        Retourne :
        - pd.DataFrame : DataFrame des meilleurs modèles.
        """
        # Suppression des colonnes inutiles
        dfnp = df.drop(['Task Args', 'Model Args', 'Training Args'], axis=1)

        # Création des groupes par 'Task' et 'Model'
        groups = dfnp.groupby(['Task', 'Model'])

        # Initialisation d'une liste pour les indices des meilleurs modèles
        best_indices = []

        # Parcours de chaque groupe
        for (task, model), group in groups:
            # Tri des modèles en fonction de la tâche
            if self._is_classification_task(task):
                sorted_group = group.sort_values(by='Accuracy', ascending=False)
            else:
                sorted_group = group.sort_values(by='MSE', ascending=True)
            
            # Obtenir les 5% meilleurs modèles
            top_5_percent = sorted_group.head(max(1, int(len(sorted_group) * 0.05)))  # Au moins 1 modèle
            best_idx = top_5_percent['BIC'].idxmin() # Choix du meilleur modèle en fonction du BIC
            best_indices.append(best_idx) # Ajout de l'indice du meilleur modèle

        # Récupérer les meilleurs modèles en fonction des indices
        best_models = df.loc[best_indices]
        return best_models
        
    def _is_classification_task(self, task_name: str) -> bool:
        """Vérifie si une tâche est de classification ou de régression.
        
        Paramètres :
        - task_name : Nom de la tâche.
        
        Retourne :
        - bool : True si la tâche est de classification, False sinon.
        """
        return self.tasks[task_name].is_classification

    def _generate_plots(self, df: 'pd.DataFrame', output_path: str):
        """Génère des visualisations des résultats pour chaque tâche.
        
        Paramètres :
        - df : DataFrame des résultats.
        - output_path : Chemin de sortie pour les visualisations.
        """
        # Pour chaque tâche
        for task_name in df['Task'].unique():
            # Filter task
            task_df = df[df['Task'] == task_name]
            
            # Performance plot
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f'Performance for Task: {task_name}', fontsize=16)

            # Performance vs Number of Parameters
            metric = 'Accuracy' if self._is_classification_task(task_name) else 'MSE'
            sns.scatterplot(data=task_df, x='Number Params', y=metric, hue=metric, ax=axes[0])
            axes[0].set_title('Number Parameters vs ' + metric)

            # Performance vs Training Time
            sns.scatterplot(data=task_df, x='Training Time (s)', y=metric, hue=metric, ax=axes[1])
            axes[1].set_title('Training Time vs ' + metric)

            # Performance vs Memory Usage
            sns.scatterplot(data=task_df, x='Memory Usage (MB)', y=metric, hue=metric, ax=axes[2])
            axes[2].set_title('Memory Usage vs ' + metric)
            
            # Save plot
            plt.xticks(rotation=45)
            plt.tight_layout(rect=(0, 0, 1, 1))
            plt.savefig(f'{output_path}/{task_name}_performance.png')
            plt.close(fig)  # Close the figure to prevent display

