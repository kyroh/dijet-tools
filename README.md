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

### Basic Usage

```python
from dijet_tools import ATLASDataLoader, LargeScaleATLASProcessor, XGBoostPredictor

# Load and process ATLAS data
loader = ATLASDataLoader("path/to/atlas/data.root")
processor = LargeScaleATLASProcessor()

# Process events in chunks
processed_data = processor.process_dataset(loader, chunk_size=100000)

# Train XGBoost model
model = XGBoostPredictor()
model.train(processed_data)

# Make predictions
predictions = model.predict(processed_data)
```

### Command Line Interface

The package provides several command-line tools:

```bash
# Process large datasets
dijet-process --input-files "/path/to/data/*.root.1" --output-dir "processed"

# Train models
dijet-train --config configs/default.yaml #make sure to edit the yaml to include your processed data files as an input

# Run inference
dijet-predict --model-path "models/xgboost_model.pkl" --data-path "processed/"

# Anomaly detection
dijet-anomaly --threshold 0.95 --output-dir "anomalies"
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

