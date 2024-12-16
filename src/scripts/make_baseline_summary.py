import os
import pandas as pd 
import numpy as np

# Récupération des résultats
folder = "baseline"
models = os.listdir(folder)
models.sort()

# Création d'une liste pour stocker les DataFrames
dfs = []
for model in models:
    path = os.path.join(folder, model)
    csvs = [elem for elem in os.listdir(path) if elem.split('.')[-1] == 'csv']
    for csv in csvs:
        df = pd.read_csv(os.path.join(path, csv))
        df['Model'] = model
        dfs.append(df)

# Concaténation des DataFrames
df = pd.concat(dfs).reset_index(drop=True)

# Ajout de la colonne 'Error'
df['Error'] = df.apply(lambda x: x['MSE'] if not np.isnan(x['MSE']) else 1-x['Accuracy'], axis=1)

# Suppression des colonnes inutiles
usefull_columns = ['Model', 'Task', 'Error', 'Number Params', 'Model Args', 'Task Args', 'Training Args']
df = df[usefull_columns]

# Define the ranges of parameters to consider
max_params = [1e3, 1e4, 1e5, 1e6, np.inf]
max_names = ['1k', '10k', '100k', '1M', 'inf']

# Init markdown with header
tab_markdown = f"| Task | {' |||||| '.join(models)} ||||||\n"
tab_markdown += "|-|" + "-|-|-|-|-|-|" * len(models) + "\n"
param_ranges = [f"<{name}" for name in max_names]
header_row = "|| " + " || ".join([" | ".join(param_ranges) for _ in models]) + " |\n"
tab_markdown += header_row

# For each tasks
tasks = df['Task'].unique()
for task in tasks:
    # Select results for the task
    df_task = df[df['Task'] == task]

    # Init row markdown
    row = f"| {task} |"

    # Extract best errors
    errors = {}
    for model in models:
        # Select results for the model
        df_model = df_task[df_task['Model'] == model]
        errors[model] = []

        # Extract best result for each max number of parameters
        min_param = 0
        for i, max_param in enumerate(max_params):
            df_max_param = df_model[(df_model['Number Params'] <= max_param) & (df_model['Number Params'] > min_param)]
            min_param = max_param
            errors[model] += [df_max_param['Error'].min() if len(df_max_param) > 0 else None]

    # Display errors
    for model in models:
        min_error = min([error for error in errors[model] if error is not None])
        # Add errors to the row
        for error in errors[model]:
            if error is not None:
                row += f" **{error:.3f}** |" if error == min_error else f" {error:.3f} |"
            else:
                row += " - |"
        row += "|"

    row += "\n"  # End the row
    tab_markdown += row

# Output the markdown table
print(tab_markdown)