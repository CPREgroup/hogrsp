# HOGSP Hyperspectral Band Selection

This MATLAB project implements a high-order graph-based method for hyperspectral band selection. The selected bands are evaluated using an SVM classifier.

## Requirements

- MATLAB
- Statistics and Machine Learning Toolbox
- Optimization Toolbox

## Usage

1. Place the datasets in the `Dataset` folder.
2. Select the datasets and configure the parameters in `config/get_config.m`.
3. Change `project_root` in `run_fixed.m` to the actual project path on your computer.
4. Open `run_fixed.m` in MATLAB and run the script.

Experimental results and loss curves will be saved in the `results` folder.

## Project Structure

- `config`: Experiment configuration
- `data`: Dataset loading and preprocessing
- `graph`: Graph construction
- `optimization`: Model optimization
- `selection`: Band ranking and selection
- `evaluation`: Classification evaluation
- `utils`: Utility functions
- `results`: Experimental results
- `run_fixed.m`: Main experiment script
