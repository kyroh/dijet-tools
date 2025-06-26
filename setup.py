from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "Dijet analysis toolkit for ATLAS Data"

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, "r", encoding="utf-8") as fh:
        requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
else:
    requirements = [
        "numpy>=1.19.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0", 
        "xgboost>=1.5.0",
        "torch==2.3.0",
        "torchvision==0.18.0",
        "torchaudio==2.3.0",
        "uproot>=4.0.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.62.0",
        "seaborn>=0.11.0",
        "scipy>=1.7.0"
    ]

setup(
    name="dijet-tools",
    version="1.0.0",  
    author="Aaron W. Tarajos",
    author_email="awtarajos@berkeley.edu",
    description="Machine Learning Analysis of Dijet Angular Scattering in ATLAS Run 2 Data ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/dijet-tools",
    project_urls={
        "Bug Tracker": "https://github.com/kyroh/dijet-tools/issues",
        "Documentation": "https://github.com/kyroh/dijet-tools/main/README.md",
        "Source Code": "https://github.com/kyroh/dijet-tools",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    license="MIT",
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "isort>=5.0",
            "flake8>=3.8",
            "mypy>=0.950",
        ],
        "distributed": [
            "dask[distributed]>=2021.0",
            "dask-jobqueue>=0.7",
        ],
        "notebooks": [
            "jupyter>=1.0",
            "ipywidgets>=7.0",
            "plotly>=5.0",
        ],
        # Remove the CUDA-specific versions - they need to be installed manually
        # or through conda/pip with --index-url
        "gpu": [
            "torch==2.3.0",
            "torchvision==0.18.0",
        ],
        "accelerate": [
            "accelerate>=0.15.0",  # For distributed training
            "torchaudio==2.3.0",  # Fixed typo: torch-audio -> torchaudio
        ],
        "visualization": [
            "tensorboard>=2.8.0",
            "wandb>=0.12.0",  # Weights & Biases for experiment tracking
            "plotly>=5.0",
            "bokeh>=2.4.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "dijet-process=dijet_tools.scripts.process_large_dataset:main",
            "dijet-train=dijet_tools.scripts.train_models:main",
            "dijet-predict=dijet_tools.scripts.run_inference:main",
            "dijet-anomaly=dijet_tools.scripts.anomaly_detection:main",
            "dijet-physics-nn=dijet_tools.scripts.physics_nn:main",
        ],
    },
    include_package_data=True,
    package_data={
        "dijet_tools": [
            "configs/*.yaml",
            "data/*.json",
        ],
    },
    zip_safe=False,
    keywords="particle physics, machine learning, ATLAS, LHC, QCD, dijet analysis, pytorch, deep learning",
)
