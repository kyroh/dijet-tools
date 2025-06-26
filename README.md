<!-- mathjax: true -->

# Dijet Analysis Toolkit

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/dijet-tools.svg)](https://badge.fury.io/py/dijet-tools)

## Introduction

I started this project in December 2024 and completed the alpha build in February 2025, which is the version I used for most of my personal research on scattering. I recently decided to build a release version of my framework to make it more accessible for people to use CERN open data. My goal was to learn collider physics and machine learning by building a complete analysis pipeline from scratch. This library handles end-to-end analysis of dijet events from ATLAS data, with a focus on QCD scattering and the search for new physics through angular observables. 

## Features

- **End-to-End Analysis Pipeline**: Complete workflow from raw ATLAS ROOT files to physics results
- **Large-Scale Data Processing**: Efficient handling of multi-terabyte datasets with distributed computing
- **Machine Learning Models**: XGBoost and PyTorch-based neural networks for QCD prediction
- **Physics-Aware Features**: Angular observables, kinematic variables, and QCD-motivated features
- **Distributed Computing**: Dask-based parallel processing for scalable analysis
- **GPU Acceleration**: CUDA support for neural network training and inference
- **Comprehensive Evaluation**: Physics-motivated metrics and statistical analysis

## Installation

### Prerequisites

- Python 3.8 or higher
- 64GB+ RAM recommended for large datasets
- CUDA-compatible GPU (optional, for acceleration)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/kyroh/dijet-tools.git
cd dijet-tools

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package
pip install -e .
```

### Advanced Installation

For GPU acceleration and additional features:

```bash
# Install with GPU support
pip install -e .[gpu,accelerate,visualization]

# Install development dependencies
pip install -e .[dev]

# Install distributed computing support
pip install -e .[distributed]
```

### Conda Installation

```bash
# Create conda environment
conda env create -f environment.yaml
conda activate dijet-tools

# Install the package
pip install -e .
```

## Quick Start

The fastest way to get started is using the command-line interface:

```bash
# Process large datasets
dijet-process --input-files "data/*.root" --output-dir "processed"

# Train models
dijet-train --config configs/default.yaml

# Run inference
dijet-predict --model-path "models/xgboost_model.pkl" --data-path "processed/"

# Anomaly detection
dijet-anomaly --threshold 0.95 --output-dir "anomalies"
```

## Usage

### Command Line Interface

The package provides several command-line tools for different analysis stages:

#### Data Processing

```bash
# Process ATLAS ROOT files
dijet-process --input-files "data/*.root" --output-dir "processed"

# Process with custom configuration
dijet-process --config configs/custom.yaml --chunk-size 50000

# Process with distributed computing
dijet-process --n-workers 4 --memory-limit "8GB"
```

#### Model Training

```bash
# Train XGBoost model
dijet-train --config configs/default.yaml

# Train neural network
dijet-train --model-type neural_network --epochs 100

# Train with GPU acceleration
dijet-train --use-gpu --batch-size 1024
```

#### Inference and Prediction

```bash
# Run inference on processed data
dijet-predict --model-path "models/xgboost_model.pkl" --data-path "processed/"

# Generate predictions with confidence intervals
dijet-predict --output-format "hdf5" --include-uncertainty

# Batch prediction on multiple datasets
dijet-predict --input-pattern "processed/*.h5" --output-dir "predictions"
```

#### Anomaly Detection

```bash
# Run anomaly detection
dijet-anomaly --threshold 0.95 --output-dir "anomalies"

# Use isolation forest method
dijet-anomaly --method isolation_forest --contamination 0.1

# Generate anomaly score distributions
dijet-anomaly --plot-scores --output-format "pdf"
```

### Python API

For more advanced usage, you can use the Python API directly:

```python
from dijet_tools import ATLASDataLoader, LargeScaleATLASProcessor, XGBoostPredictor
from dijet_tools.models import NeuralNetworkPredictor
from dijet_tools.evaluation import PhysicsMetrics

# Load and process ATLAS data
loader = ATLASDataLoader("path/to/atlas/data.root")
processor = LargeScaleATLASProcessor()

# Process events in chunks
processed_data = processor.process_dataset(loader, chunk_size=100000)

# Train XGBoost model
xgb_model = XGBoostPredictor()
xgb_model.train(processed_data)

# Train neural network
nn_model = NeuralNetworkPredictor()
nn_model.train(processed_data, epochs=100, batch_size=1024)

# Make predictions
xgb_predictions = xgb_model.predict(processed_data)
nn_predictions = nn_model.predict(processed_data)

# Evaluate physics performance
metrics = PhysicsMetrics()
results = metrics.evaluate(xgb_predictions, processed_data)
```

### Advanced Usage Examples

#### Custom Feature Engineering

```python
from dijet_tools.features import FeatureEngineer
from dijet_tools.physics import KinematicCalculator

# Create custom features
calculator = KinematicCalculator()
features = FeatureEngineer()

# Add custom angular observables
custom_features = features.add_angular_observables(
    processed_data, 
    include_centrality=True,
    include_balance_variables=True
)

# Add QCD-motivated features
qcd_features = features.add_qcd_features(
    custom_features,
    include_color_flow=True,
    include_jet_substructure=True
)
```

#### Distributed Processing

```python
from dijet_tools.pipeline import DistributedProcessor
from dask.distributed import Client

# Set up distributed computing
client = Client(n_workers=4, memory_limit="8GB")
processor = DistributedProcessor(client)

# Process large dataset
results = processor.process_large_dataset(
    "data/large_dataset.root",
    chunk_size=100000,
    output_dir="processed"
)
```

#### Model Comparison and Ensemble

```python
from dijet_tools.models import ModelEnsemble
from dijet_tools.evaluation import ModelComparison

# Create ensemble of models
ensemble = ModelEnsemble([
    XGBoostPredictor(),
    NeuralNetworkPredictor(),
    IsolationForestPredictor()
])

# Train ensemble
ensemble.train(processed_data)

# Compare model performance
comparison = ModelComparison()
comparison_results = comparison.compare_models(
    [xgb_model, nn_model, ensemble],
    processed_data
)
```

## Configuration

The analysis is configured through YAML files. Key configuration options:

```yaml
# Data processing
data:
  chunk_size: 100000              # Events per processing chunk
  max_memory_gb: 64.0             # Maximum memory usage
  cache_dir: "cache"              # Cache directory

# Physics cuts
physics:
  pt_threshold_gev: 50.0          # Minimum jet pT
  eta_threshold: 4.5              # Maximum |eta|
  mjj_min_gev: 200.0              # Minimum dijet mass

# Machine learning
model:
  test_size: 0.2                  # Test set fraction
  feature_scaling: "robust"       # Scaling method
  xgb_params:                     # XGBoost parameters
    max_depth: 6
    learning_rate: 0.1
```

## Physics Background

This analysis focuses on dijet events in proton-proton collisions at the LHC:

- **QCD Scattering**: Standard Model dijet production through gluon and quark interactions
- **Angular Observables**: Variables sensitive to the underlying parton dynamics
- **New Physics Search**: Deviations from QCD predictions indicating new particles or interactions

Key observables include:
- Dijet mass (mjj)
- Angular separation (Δφ, Δη)
- pT balance
- Centrality ratio

## Machine Learning Approach

### Models

1. **XGBoost**: Gradient boosting for QCD background prediction
2. **Neural Networks**: PyTorch-based models with physics-informed loss functions
3. **Anomaly Detection**: Isolation Forest and autoencoder-based methods

### Features

- **Kinematic Variables**: pT, η, φ, mass, energy
- **Angular Observables**: Δφ, Δη, centrality, balance variables
- **Jet Properties**: JVT, b-tagging, pileup corrections
- **Event-Level Features**: MET, vertex information, trigger bits

## Contributing

We welcome contributions from the community! Here's how you can help:

### Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/dijet-tools.git
   cd dijet-tools
   ```
3. **Create a virtual environment** and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -e .[dev]
   ```

### Contributing

**Create a feature branch**:
```bash
git checkout -b feature/your-feature-name
git add .
git commit -m "Add new feature: description of changes"
git push origin feature/your-feature-name
# Then create a pull request on GitHub
```

### Contribution Guidelines

- **Documentation**: Update docstrings and README as needed
- **Physics**: Ensure physics calculations are correct and well-documented
- **Performance**: Consider performance implications for large datasets

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{dijet_tools,
  title={ATLAS Dijet Analysis Toolkit},
  author={Tarajos, Aaron W.},
  year={2025},
  url={https://github.com/kyroh/dijet-tools}
}
```

## Contact

- **Author**: Aaron W. Tarajos
- **Email**: awtarajos@berkeley.edu
- **GitHub**: [@kyroh](https://github.com/kyroh)
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{dijet_tools,
  title={ATLAS Dijet Analysis Toolkit},
  author={Tarajos, Aaron W.},
  year={2025},
  url={https://github.com/kyroh/dijet-tools}
}
```

## Contact

- **Author**: Aaron W. Tarajos
- **Email**: awtarajos@berkeley.edu
- **GitHub**: [@kyroh](https://github.com/kyroh)

