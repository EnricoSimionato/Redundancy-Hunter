# Redundancy-Hunter

The Redundancy-Hunter repository contains the code to perform redundancy analysis and evaluate the information content of layers in large language models. It provides tools to identify, evaluate, and potentially replace redundant components in transformer-based architectures, facilitating model compression and efficiency improvements. It builds upon the experimental framework established in Experiment-Orchestrator, adapting it to the specific requirements of these analyses. In addition to the investigations presented in the thesis "Exploring Redundancy and Shared Representations for Transformer Models Optimization", it includes further studies that are not explicitly covered in the main text.

## Features
-	Comprehensive Analysis Modules: Includes various analysis scripts to assess different aspects of model redundancy, such as matrix ranks, functional redundancy, repeated modules.
-	Layer Replacement Utilities: Tools to experiment with replacing layers with other from the same model or from another source.
-	Configurable Experiment Launcher: Easily run experiments with customizable configurations.
-	Results Logging and Merging: Systematic logging of experiment results and utilities to merge multiple analysis outputs.
-	Docker Support: Containerized environment for consistent and reproducible experiments.
 
## Directory Structure

Redundancy-Hunter/
├── Dockerfile                  # Docker setup for containerized environment
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Project metadata
├── docker_container.sh         # Shell script to build and run Docker container
├── src/
│   ├── redhunter/
│   │   ├── __init__.py
│   │   ├── analysis/           # Core analysis scripts
│   │   │   ├── activations_analysis.py
│   │   │   ├── analysis_utils.py
│   │   │   ├── basic_model_analysis.py
│   │   │   ├── head_analysis.py
│   │   │   ├── layer_replacement_analysis.py
│   │   │   ├── matrix_initialization_analysis.py
│   │   │   ├── rank_analysis.py
│   │   │   ├── sorted_layers_compression_analysis.py
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   ├── list_utils/
│   │   │   │   ├── list_utils.py
│   │   │   ├── layer_replacement_wrapper/
│   │   │   │   ├── layer_replacement_wrapper.py
│   ├── analysis_experiment.py  # Refactored analysis runner
│   ├── analysis_launcher.py    # Config loader and launcher
│   ├── analysis_merger.py      # Merge multiple analysis results
├── experiments/
│   ├── configurations/         # YAML/JSON configs for experiments
│   ├── results/                # Output results
├── test/
│   ├── analysis/               # Unit tests (if any)

## Installation
1.	Clone the repository
  ```bash
  git clone https://github.com/EnricoSimionato/Redundancy-Hunter.git
  cd Redundancy-Hunter
  ```

2.	Create a virtual environment (optional but recommended)

3.	Install dependencies
  ```bash
  pip install -r requirements.txt
  ```

## Usage
1. You can launch experiments or analysis types by using the analysis_launcher.py script. It reads configuration files and triggers the corresponding analyses.
  ```bash
  python src/redhunter/analysis_launcher.py --config path/to/config.yaml
  ```
2. Alternatively, for direct execution:
  ```bash
  python src/redhunter/analysis_experiment.py
  ```
To customize the experiment, edit the YAML/JSON files under experiments/configurations/.

3. Optionally, you can use Docker and run using:

  ```bash
  # Build and run in container
  bash docker_container.sh
  ```

## Notes
- Ensure the model used is supported by HuggingFace Transformers.
-	Analysis results are saved under experiments/results/.
-	The project is designed to explore redundancy in Transformer layers, with potential applications to model compression.

## Contributions

## References 


