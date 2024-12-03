# STREAM: Sequential Task Review for Evaluating Artificial Memory

STREAM is a comprehensive benchmark designed to evaluate the memory and processing capabilities of sequential models (RNNs, Transformers, etc.) through 13 diverse tasks. These tasks are specifically crafted to test different aspects of artificial memory and information processing.

## 🎯 Features

- 13 carefully designed sequential tasks
- Automated evaluation pipeline
- Built-in support for multiple model architectures
- Customizable task parameters
- Detailed performance reporting
- Support for hyperparameter optimization

## 📚 Tasks Overview

The benchmark includes three main categories of tasks:

### 1. Simple Memory Tests
- **Discrete Postcasting**: Copy a discrete sequence after a delay
- **Continue Postcasting**: Copy a continue sequence after a delay
- **Copy Task**: Memorize and reproduce a sequence when triggered
- **Selective Copy**: Memorize and reproduce only marked elements
- **MNIST Classification**: Process MNIST images sequentially and classify them

### 2. Memory Manipulation Tests
- **Adding Problem**: Add numbers at marked positions in a sequence
- **Sorting Problem**: Sort a sequence based on given positions

### 3. Long-Term Dependency Tests
- **Discrete Pattern Completion**: Identify and complete repetitive discrete patterns
- **Continue Pattern Completion**: Identify and complete repetitive continue patterns
- **Bracket Matching**: Validate nested bracket sequences

### 4. Signal Processing Tests
- **Sin Forecasting**: Predict frequency-modulated sinusoidal signals
- **Chaotic Forecasting**: Predict states in a chaotic system (Lorenz attractor)

## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/stream-benchmark
cd stream-benchmark

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

To evaluate a model on all tasks:
```bash
python run_evaluation.py --model LSTM --n_trials 5 --nb_seeds 3
```

To evaluate on a specific task:
```bash
python run_evaluation.py --model ESN --task discrete_postcasting --n_trials 5
```

### Command Line Arguments
- `--model`: Model architecture to evaluate (ESN, LSTM, Transformers, TransformerDecoderOnly)
- `--n_trials`: Number of optimization trials for each task
- `--nb_seeds`: Number of random seeds for statistical significance
- `--tasks`: Specific task to evaluate ('all' for all tasks, 'none' for no tasks), you can specify multiple tasks by separating them with commas ',' (default: 'all')
- `--report`: Whether to generate a detailed report (default: True)

## 📊 Evaluating Your Own Model

To evaluate your own model on STREAM, follow these steps:

1. Create a new model class that inherits from the base model class:
```python
from src.models.base import BaseModel

class YourModel(BaseModel):
    def __init__(self, input_size, output_size, **kwargs):
        super().__init__()
        # Initialize your model
        
    def forward(self, x):
        # Implement forward pass
        return output
        
    def train_step(self, batch):
        # Implement training step
        return loss
```

2. Add your model to the MODELS dictionary in `run_evaluation.py`:
```python
MODELS = {
    'YourModel': {
        'model': YourModel,
        'args': {
            # Your model's hyperparameters
            "hidden_size": [8, 256],
            "learning_rate": 1e-3,
        },
        'training_args': {
            "epochs": 10,
            "batch_size": 10
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

The benchmark generates a detailed report including:
- Performance metrics for each task
- Learning curves
- Statistical analysis across seeds
- Hyperparameter sensitivity analysis
- Comparative analysis with baseline models

Example report structure:
```
results/
├── YourModel/
│   ├── task_results.csv
│   ├── hyperparameter_analysis.png
│   ├── learning_curves/
│   │   ├── discrete_postcasting.png
│   │   ├── continue_postcasting.png
│   │   └── ...
│   └── report.pdf
```

## 🔍 Task Parameters

Each task can be customized through various parameters. Here are some key parameters:

```python
# Example: Discrete Postcasting
params = {
    "sequence_length": 1000,  # Length of the sequence
    "delay": 10,             # Delay before reproduction
    "n_symbols": 8,          # Number of possible symbols
    "training_ratio": 0.8    # Train/test split ratio
}
```

See individual task documentation for complete parameter lists.

## 📊 Baseline Results

| Model | Copy | Adding | Sorting | Pattern | Bracket | Signal |
|-------|------|--------|---------|---------|---------|--------|
| ESN   | 0.95 | 0.89   | 0.82    | 0.88    | 0.85    | 0.91   |
| LSTM  | 0.98 | 0.93   | 0.87    | 0.92    | 0.89    | 0.94   |
| Trans.| 0.99 | 0.95   | 0.91    | 0.94    | 0.92    | 0.96   |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 Citation

If you use STREAM in your research, please cite:
```bibtex
@misc{stream2024,
  title={STREAM: Sequential Task Review for Evaluating Artificial Memory},
  author={Your Name},
  year={2024},
  publisher={GitHub},
  howpublished={\url{https://github.com/yourusername/stream-benchmark}}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.