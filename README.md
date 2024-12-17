# STREAM: Sequential Task Review to Evaluate Artificial Memory

STREAM is a benchmark designed to evaluate the memory and processing capabilities of sequential models (RNNs, Transformers, etc.) through 12 diverse tasks. These tasks are specifically crafted to test different aspects of artificial memory and information processing.

## 📊 Baseline Results

| Task | ESN |||||| LSTM |||||| TransformerDecoderOnly |||||| Transformers ||||||
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|| <1k | <10k | <100k | <1M | <inf || <1k | <10k | <100k | <1M | <inf || <1k | <10k | <100k | <1M | <inf || <1k | <10k | <100k | <1M | <inf |
| copy_task | 1.000 | **0.984** | 0.997 | - | - || 0.993 | 0.892 | **0.858** | 0.864 | 0.890 || - | 0.893 | **0.891** | 0.899 | - || - | - | **0.894** | 0.897 | - ||
| discrete_postcasting | 1.000 | 0.958 | **0.916** | - | - || - | **0.963** | 0.968 | 0.984 | 0.995 || - | 0.968 | **0.942** | 0.968 | - || - | 1.000 | 0.958 | **0.953** | - ||
| continue_pattern_completion | 0.015 | 0.014 | **0.012** | - | - || 0.014 | 0.008 | 0.003 | 0.002 | **0.002** || - | 0.015 | 0.015 | **0.015** | - || - | 0.281 | 0.080 | **0.079** | - ||
| selective_copy_task | 1.000 | **0.980** | 0.995 | - | - || 0.946 | 0.895 | 0.892 | **0.890** | 0.891 || - | 0.891 | **0.890** | 0.892 | - || - | 0.907 | 0.902 | **0.898** | - ||
| sorting_problem | 1.000 | 0.987 | **0.984** | - | - || - | 0.903 | 0.896 | 0.893 | **0.892** || - | **0.892** | **0.892** | **0.892** | - || - | 0.999 | **0.893** | 0.896 | - ||
| chaotic_forecasting | 0.000 | 0.000 | **0.000** | - | - || 0.085 | 0.002 | 0.000 | **0.000** | 0.000 || - | 0.045 | **0.024** | 0.029 | - || - | 0.146 | 0.124 | **0.123** | - ||
| adding_problem | 1.000 | 0.990 | **0.960** | - | - || - | **1.000** | **1.000** | **1.000** | **1.000** || - | 1.000 | 0.990 | **0.985** | - || - | 1.000 | **0.970** | 0.995 | - ||
| sin_forecasting | 0.005 | **0.003** | 0.004 | - | - || 0.302 | 0.046 | 0.027 | 0.019 | **0.018** || - | 0.030 | **0.027** | 0.027 | - || - | 0.513 | **0.483** | 0.489 | - ||
| bracket_matching | 0.125 | 0.120 | **0.110** | - | - || **0.430** | **0.430** | **0.430** | **0.430** | **0.430** || - | 0.430 | **0.320** | 0.350 | - || - | 0.500 | 0.430 | **0.420** | - ||
| continue_postcasting | 0.010 | 0.001 | **0.001** | - | - || 0.201 | 0.160 | 0.130 | 0.102 | **0.090** || - | 0.198 | **0.197** | 0.197 | - || - | 0.211 | **0.197** | 0.197 | - ||
| discrete_pattern_completion | **0.200** | **0.200** | **0.200** | - | - || - | 0.363 | 0.220 | 0.203 | **0.199** || - | **0.200** | **0.200** | **0.200** | - || - | 1.000 | **0.928** | 0.933 | - ||
| mnist_classification | 0.505 | **0.220** | **0.220** | - | - || - | **0.995** | 1.000 | 1.000 | 1.000 || - | 0.855 | **0.835** | 0.840 | - || - | 0.870 | **0.840** | **0.840** | - ||

## 📚 Tasks Overview

The benchmark includes the following tasks:

### 1. Simple Memory Tests
- **Discrete Postcasting**: Copy a discrete sequence after a delay
![Discrete Postcasting](./images/discrete_postcasting.png)
- **Continue Postcasting**: Copy a continue sequence after a delay
![Continue Postcasting](./images/continue_postcasting.png)

### 2. Signal Processing Tests
- **Sin Forecasting**: Predict frequency-modulated sinusoidal signals
![Sin Forecasting](./images/sin_forecasting.png)
- **Chaotic Forecasting**: Predict states in a chaotic system (Lorenz attractor)
![Chaotic Forecasting](./images/chaotic_forecasting.png)

### 3. Long-Term Dependency Tests
- **Discrete Pattern Completion**: Identify and complete repetitive discrete patterns
![Discrete Pattern Completion](./images/discrete_pattern_completion.png)
- **Continue Pattern Completion**: Identify and complete repetitive continue patterns
![Continue Pattern Completion](./images/continue_pattern_completion.png)
- **Copy Task**: Memorize and reproduce a sequence when triggered
![Copy Task](./images/copy_task.png)
- **Selective Copy**: Memorize and reproduce only marked elements
![Selective Copy](./images/selective_copy.png)

### 4. Information Manipulation Tests
- **Adding Problem**: Add numbers at marked positions in a sequence
![Adding Problem](./images/adding_problem.png)
- **Sorting Problem**: Sort a sequence based on given positions
![Sorting Problem](./images/sorting_problem.png)
- **MNIST Classification**: Process MNIST images sequentially and classify them
![MNIST Classification](./images/mnist_classification.png)
- **Bracket Matching**: Validate nested bracket sequences
![Bracket Matching](./images/bracket_matching.png)



## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/naowak/stream-benchmark
cd stream-benchmark

# Create a virtual environment & activate it (optional)
python -m venv stream_venv
source stream_venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### How it works

The benchmark is designed to find by itself the best Hyper-Parameters (HP) for each model on each task. It will automatically run a random search for each task on the HP space defined in the `run_evaluation.py` file and create a report. The number of trials and seeds can be defined in the command line arguments, as well as the model to evaluate and the tasks to evaluate on.

LSTM, Transformers, Transformer-Decoder and Echo State Networks (ESN) are implemented as baseline models. 

### Basic Usage
To evaluate LSTM on all tasks:
```bash
python run_evaluation.py --model LSTM --n_trials 5 --nb_seeds 3
```

To evaluate on a specific task:
```bash
python run_evaluation.py --model ESN --task discrete_postcasting --n_trials 5 --nb_seeds 3
```

A report will automatically be generated in the `results/` directory. 
You can also find the `.csv` files containing the results for each task in the `results/` directory.

### Command Line Arguments

- `--model` (default: `ESN`) : Model architecture to evaluate
- `--n_trials` (default: `1`) : Number of optimization trials for each task per seed
- `--nb_seeds` (default: `1`): Number of random seeds for statistical significance
- `--report` (default: `True`): Whether to generate a detailed report (default: True)
- `--tasks` (default: `all`): Specific task to evaluate ('all' for all tasks, 'none' for no tasks), you can specify multiple tasks by separating them with commas spaces ' ' (e.g. `--tasks discrete_postcasting continue_postcasting`)
- `--device` (default: `none`): If the model is GPU compatible, specify the device to use (e.g. `--device cuda`). This value will be add to the args of the model as `device`.

#### List of available models:
- `ESN`
- `LSTM`
- `Transformers`
- `TransformerDecoderOnly`
+ Any custom model you want to evaluate -> see section below

#### List of available tasks:
- `discrete_postcasting`
- `continue_postcasting`
- `sin_forecasting`
- `chaotic_forecasting`
- `discrete_pattern_completion`
- `continue_pattern_completion`
- `copy_task`
- `selective_copy`
- `adding_problem`
- `sorting_problem`
- `mnist_classification`
- `bracket_matching`

## 📊 Evaluate Your Own Model

To evaluate your own model on STREAM, follow these steps:

1. Implement your model in `src/models/your_model.py`. It must have the followings methods implemented:  

```python
class YourModel():
    def __init__(self, **kwargs):
        # Initialize your model
        
    def train(self, X, Y, classification=False, prediction_timesteps=[], **kwargs):
        # Train your model
        #
        # Parameters :
        # X (np.ndarray) : Input data. (sample, time, input_dim) 
        # Y (np.ndarray) : Output data. (sample, time, output_dim) 
        # classification (bool) : Whether the task is a classification task
        # prediction_timesteps (list of int) : Timesteps at which the model should predict -> timesteps at which you should compute the error between predictions and ground truth (Y)
        
    def run(self, X, **kwargs):
        # Run your model sequentially
        #
        # Parameters :
        # X (np.ndarray) : Input data. (sample, time, input_dim) 
        #
        # Returns :
        # (np.ndarray) : Predictions. (sample, time, output_dim) 
    
    def count_params(self, **kwargs):
        # Count the number of parameters in your model
        #
        # Returns :
        # (int) : Number of parameters
```

2. Add your model to the MODELS dictionary in `run_evaluation.py`. You can specify the args your model needs to be trained with the `training_args` key. And you can use the `args` key to define the HP space with the following rules:
- Use `[int, int]` or `[float, float]` to sample uniformly in the range
- Use `(value_1, value_2, value_3, ...)` to make a random choice between the values
- Use a single value to fix the HP

For example:
```python
MODELS = {
    'YourModel': {
        'model': YourModel,
        'args': {
            # Your model's hyperparameters
            "layers": [1, 20], # Randomly choose an integer between 1 and 20
            "learning_rate": [0.0001, 0.1], # Randomly choose a float between 0.0001 and 0.1
            "hidden_size": (64, 128, 256), # Randomly choose a value between 64, 128 and 256
            "dropout": 0.2, # Fix the dropout to 0.2
            "activation": ("relu", "tanh"), # Randomly choose between "relu" and "tanh"
        },
        'training_args': {
            # Your model's training args
            "epochs": 10,
            "batch_size": 32,
        },
    },
    # ... other models
}
```

3. Run the evaluation:
```bash
python run_evaluation.py --model YourModel
```

## 📈 Understanding the Results

The benchmark generates a report that automatically picks the best hyperparameters for each task and model. To do so, it selects the best BIC score (Bayesian Information Criterion) in the top 5% of the trials for each task and model. It then show some plots displaying the performance of the model on each task in function of its number of parameters.

Example report structure:
```bash
results/
├── YourModel/
│   ├── adding_problem.csv # results for the adding problem task
│   ├── adding_problem_performance.png # performance plot for the adding problem task
│   ├── bracket_matching.csv # results for the bracket matching task
│   ├── bracket_matching_performance.png # performance plot for the bracket matching task
│   ├── ...
│   └── report.md # report generated for YourModel
```

## 🔍 Task Parameters

Each task can be customized through various parameters. You can find and modify the definition of these parameters in the `./src/evaluation.py` file. To find more information for each parameter, you can have a look at the file where all tasks are defined : `./src/tasks.py`.  

Here is an example of the parameters for the Discrete Postcasting task:

```python
# Example: Discrete Postcasting
params = {
    "sequence_length": 1000,  # Length of the sequence
    "delay": 10,             # Delay before reproduction
    "n_symbols": 8,          # Number of possible symbols
    "training_ratio": 0.8    # Train/test split ratio
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 Citation

If you use STREAM in your research, please cite:
```bibtex
@misc{stream2024,
  title={STREAM: Sequential Task Review for Evaluating Artificial Memory},
  author={Yannis Bendi-Ouis, Xavier Hinaut},
  year={2024},
  publisher={GitHub},
  howpublished={\url{https://github.com/naowak/stream-benchmark}}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.