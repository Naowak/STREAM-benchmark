report_template = """
- Training Time: {:.4f} seconds
- Inference Time: {:.4f} seconds
- Memory Usage: {:.4f} MB
- Number Params: {}

#### Task Parameters
{}

#### Model Parameters
{}

#### Training Parameters
{}

#### Performance Plot
![Performance Plot](./{}_performance.png)

"""

classification_task_template = """
#### Results
- Accuracy: {:.4f}
- Precision: {:.4f}
- Recall: {:.4f}""" + report_template

regression_task_template = """
#### Results
- MSE: {:.4f}""" + report_template