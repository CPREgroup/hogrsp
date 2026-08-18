# HOGSP Hyperspectral Band Selection

High-order graph-based hyperspectral band selection evaluated with an RBF-SVM classifier.

## Requirements

- MATLAB
- Statistics and Machine Learning Toolbox
- Optimization Toolbox

## Configuration

Edit `config/get_config.m`:

```matlab
cfg.dataset_name = "IndianPines";       % Run one dataset at a time
cfg.search.alpha = [0.001, 0.002];      % Candidate values
cfg.search.topk_list = [5 10 15 20 25 30];
cfg.eval.seeds = 42:44;                 % Repeated classification seeds
cfg.eval.train_ratio = 0.05;
cfg.eval.svm_C = 1000;
cfg.opt.print_interval = 10;            % Optimization log interval
```

Values under `cfg.search` are evaluated as a Cartesian product. Place dataset files in the `Dataset` directory.

## Run

Run `run.m` in MATLAB.

For each band count (5, 10, 15, 20, 25, and 30 by default), the program selects the parameter combination with the highest mean OA. Ties are resolved by lower OA standard deviation and then earlier search order.

## Output

Results are saved under `results/<timestamp>/`:

- `best_results_by_band_count.csv`: best result for each band count
- `parameter_search_results.mat`: configuration and best results
- `parameter_search.log`: execution log
- `errors.csv`: failed parameter combinations
