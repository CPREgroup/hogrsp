clc; clear; close all;

%% Paths and output directory
project_root = fileparts(mfilename('fullpath'));
dataset_root = fullfile(project_root, 'Dataset');
run_tag = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss_SSS'));
output_dir = fullfile(project_root, 'results', run_tag);

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

restoredefaultpath;
rehash toolboxcache;
clear pca;

addpath(project_root);
addpath(fullfile(project_root, 'config'));
addpath(fullfile(project_root, 'data'));
addpath(fullfile(project_root, 'graph'));
addpath(fullfile(project_root, 'optimization'));
addpath(fullfile(project_root, 'selection'));
addpath(fullfile(project_root, 'evaluation'));
addpath(fullfile(project_root, 'utils'));

log_file = fullfile(output_dir, 'parameter_search.log');
diary(log_file);
diary_cleanup = onCleanup(@() diary('off')); 

%% Configuration: one dataset and one common parameter search space
base_cfg = get_config();
base_cfg.path.dataset_dir = dataset_root;

[dataset_key, dataset_cfg] = resolve_dataset( ...
    base_cfg.dataset_name, base_cfg.datasets);
base_cfg.dataset_name = dataset_cfg.name;
base_cfg.path.hsi_file = fullfile(dataset_root, dataset_cfg.hsi_file);
base_cfg.path.gt_file = fullfile(dataset_root, dataset_cfg.gt_file);
base_cfg.path.ers_file = fullfile(dataset_root, dataset_cfg.ers_file);

search_grid = build_search_grid(base_cfg.search);
topk_list = validate_positive_integers( ...
    base_cfg.search.topk_list, 'search.topk_list');
classification_seeds = validate_integer_values( ...
    base_cfg.eval.seeds, 'eval.seeds');
if any(classification_seeds < 0)
    error('cfg.eval.seeds must contain nonnegative integers.');
end

fprintf('========== HOGSP Single-Dataset Parameter Search ==========\n');
fprintf('project_root = %s\n', project_root);
fprintf('output_dir   = %s\n', output_dir);
fprintf('dataset      = %s (key: %s)\n', base_cfg.dataset_name, dataset_key);
fprintf('trials       = %d\n', height(search_grid));
fprintf('topk_list    = %s\n', strjoin(string(topk_list), ' '));
fprintf('classification seeds = %s\n', ...
    strjoin(string(classification_seeds), ' '));

results_table = table('Size', [0, 21], ...
    'VariableTypes', { ...
        'double','string', ...
        'double','double','double','double','double','double','double', ...
        'double','double','string', ...
        'double','double','double','double','double','double', ...
        'string','string','double'}, ...
    'VariableNames', { ...
        'Trial','Dataset', ...
        'Alpha','Lambda','Eta','Gamma','KnnK','SubspaceDim','HighOrderT', ...
        'NumBands','NumClassificationRuns','ClassificationSeeds', ...
        'OA_mean_percent','OA_std_percent','AA_mean_percent','AA_std_percent', ...
        'Kappa_mean','Kappa_std', ...
        'SelectedBands1Based','SelectedBands0Based','RuntimeSeconds'});

error_table = table('Size', [0, 3], ...
    'VariableTypes', {'double','string','string'}, ...
    'VariableNames', {'Trial','Dataset','Message'});

%% Load and preprocess the selected dataset once
[cube, gt, ers_labels] = load_hsi_dataset(base_cfg);
cube = normalize_cube(cube);
[X, ~] = build_region_samples(cube, ers_labels, gt);

first_order_cache = containers.Map('KeyType', 'char', 'ValueType', 'any');
high_order_cache = containers.Map('KeyType', 'char', 'ValueType', 'any');

%% Cartesian grid search
for itrial = 1:height(search_grid)
    trial_timer = tic;
    row = search_grid(itrial, :);
    cfg = apply_trial(base_cfg, row);
    first_result_row = height(results_table) + 1;

    fprintf('\n%s\n', repmat('#', 1, 90));
    fprintf('[TRIAL] %d / %d\n', itrial, height(search_grid));
    fprintf(['alpha=%.6g, lambda=%.6g, eta=%.6g, gamma=%.6g, ', ...
             'knn=%d, dim=%d, T=%d\n'], ...
        cfg.opt.alpha, cfg.opt.lambda, cfg.opt.eta, cfg.opt.gamma, ...
        cfg.graph.knn_k, cfg.opt.subspace_dim, cfg.graph.high_order_T);
    fprintf('%s\n', repmat('#', 1, 90));

    try
        c_key = sprintf('knn=%d|sigma=%.17g', ...
            cfg.graph.knn_k, cfg.graph.sigma);
        if isKey(first_order_cache, c_key)
            C = first_order_cache(c_key);
        else
            C = build_knn_graph(X, cfg.graph.knn_k, cfg.graph.sigma);
            first_order_cache(c_key) = C;
        end

        f_key = sprintf('%s|T=%d', c_key, cfg.graph.high_order_T);
        if isKey(high_order_cache, f_key)
            graph_data = high_order_cache(f_key);
            F = graph_data.F;
            Lf = graph_data.Lf;
        else
            [F, Lf] = build_high_order_graph(C, cfg.graph.high_order_T);
            high_order_cache(f_key) = struct('F', F, 'Lf', Lf);
        end

        [H, S, G] = initialize_variables(X, F, cfg);
        [H, S, ~] = optimize_HS(X, F, Lf, H, S, G, cfg); %#ok<ASGLU>
        [band_ranking, ~] = rank_bands_by_H(H);

        for ik = 1:numel(topk_list)
            num_bands = topk_list(ik);
            if num_bands > numel(band_ranking)
                fprintf('[SKIP] top-%d exceeds available bands %d.\n', ...
                    num_bands, numel(band_ranking));
                continue;
            end

            selected_1based = band_ranking(1:num_bands);
            selected_0based = selected_1based - 1;
            metric = evaluate_repeated( ...
                cube, gt, selected_1based, cfg.eval, classification_seeds);

            results_table(end + 1, :) = { ...
                itrial, string(cfg.dataset_name), ...
                cfg.opt.alpha, cfg.opt.lambda, cfg.opt.eta, cfg.opt.gamma, ...
                cfg.graph.knn_k, cfg.opt.subspace_dim, cfg.graph.high_order_T, ...
                num_bands, numel(classification_seeds), ...
                strjoin(string(classification_seeds), ' '), ...
                100 * metric.OA_mean, 100 * metric.OA_std, ...
                100 * metric.AA_mean, 100 * metric.AA_std, ...
                metric.Kappa_mean, metric.Kappa_std, ...
                strjoin(string(selected_1based(:)'), ' '), ...
                strjoin(string(selected_0based(:)'), ' '), NaN};

            fprintf(['[TOP-%02d, MEAN +/- STD] OA=%.2f +/- %.2f%%, ', ...
                     'AA=%.2f +/- %.2f%%, Kappa=%.6f +/- %.6f\n'], ...
                num_bands, 100 * metric.OA_mean, 100 * metric.OA_std, ...
                100 * metric.AA_mean, 100 * metric.AA_std, ...
                metric.Kappa_mean, metric.Kappa_std);
        end

        trial_runtime = toc(trial_timer);
        if height(results_table) >= first_result_row
            results_table.RuntimeSeconds(first_result_row:end) = trial_runtime;
        end
    catch ME
        fprintf('[ERROR] Trial %d failed: %s\n', itrial, ME.message);
        for si = 1:numel(ME.stack)
            fprintf('  at %s, line %d\n', ME.stack(si).name, ME.stack(si).line);
        end
        error_table(end + 1, :) = { ...
            itrial, string(base_cfg.dataset_name), ...
            string(getReport(ME, 'extended', 'hyperlinks', 'off'))};
    end

    writetable(error_table, fullfile(output_dir, 'errors.csv'));
end

%% Select the highest-mean-OA result separately for every band count
if isempty(results_table)
    save(fullfile(output_dir, 'parameter_search_results.mat'), ...
        'base_cfg', 'search_grid', 'error_table', '-v7.3');
    error('No parameter combination completed successfully. See %s.', log_file);
end

best_indices = choose_best_results_by_band_count(results_table, topk_list);
best_results = results_table(best_indices, :);
writetable(best_results, fullfile(output_dir, 'best_results_by_band_count.csv'));
save(fullfile(output_dir, 'parameter_search_results.mat'), ...
    'base_cfg', 'search_grid', 'error_table', ...
    'best_results', '-v7.3');

fprintf('\n%s\n', repmat('=', 1, 90));
fprintf('BEST RESULT BY MEAN OA FOR EACH BAND COUNT: %s\n', ...
    base_cfg.dataset_name);
fprintf('%s\n', repmat('=', 1, 90));

for ibest = 1:height(best_results)
    best = best_results(ibest, :);
    fprintf('\n[TOP-%02d BEST]\n', best.NumBands);
    fprintf('Mean OA = %.4f%%, Std OA = %.4f%%\n', ...
        best.OA_mean_percent, best.OA_std_percent);
    fprintf('Mean AA = %.4f%%, Std AA = %.4f%%\n', ...
        best.AA_mean_percent, best.AA_std_percent);
    fprintf('Mean Kappa = %.6f, Std Kappa = %.6f\n', ...
        best.Kappa_mean, best.Kappa_std);
    fprintf('Selected bands (1-based): %s\n', best.SelectedBands1Based);
end

fprintf('Best results: %s\n', ...
    fullfile(output_dir, 'best_results_by_band_count.csv'));
fprintf('MAT details: %s\n', fullfile(output_dir, 'parameter_search_results.mat'));
fprintf('Log:         %s\n', log_file);
fprintf('========== Done ==========\n');

%% Local helper functions
function [dataset_key, dataset_cfg] = resolve_dataset(requested_name, datasets)
keys = string(fieldnames(datasets));
match = find(strcmpi(keys, string(requested_name)), 1);

if isempty(match)
    error('Unknown dataset "%s". Available keys: %s.', ...
        requested_name, strjoin(keys, ', '));
end

dataset_key = keys(match);
dataset_cfg = datasets.(char(dataset_key));
end

function search_grid = build_search_grid(search)
names = {'alpha', 'lambda', 'eta', 'gamma', ...
         'knn_k', 'subspace_dim', 'high_order_T'};
values = cell(size(names));

for i = 1:numel(names)
    name = names{i};
    if ~isfield(search, name)
        error('Missing search range: cfg.search.%s.', name);
    end
    value = search.(name);
    if ~isnumeric(value) || isempty(value) || any(~isfinite(value(:)))
        error('cfg.search.%s must contain finite numeric values.', name);
    end
    values{i} = unique(value(:)', 'stable');
end

values{5} = validate_positive_integers(values{5}, 'search.knn_k');
values{6} = validate_positive_integers(values{6}, 'search.subspace_dim');
values{7} = validate_positive_integers(values{7}, 'search.high_order_T');

if any(values{1} < 0) || any(values{2} < 0) || ...
        any(values{3} < 0) || any(values{4} < 0)
    error('alpha, lambda, eta, and gamma search values must be nonnegative.');
end

grids = cell(size(values));
[grids{:}] = ndgrid(values{:});
search_grid = table();

for i = 1:numel(names)
    search_grid.(names{i}) = grids{i}(:);
end
end

function cfg = apply_trial(base_cfg, row)
cfg = base_cfg;
cfg.opt.alpha = row.alpha;
cfg.opt.lambda = row.lambda;
cfg.opt.eta = row.eta;
cfg.opt.gamma = row.gamma;
cfg.graph.knn_k = row.knn_k;
cfg.opt.subspace_dim = row.subspace_dim;
cfg.graph.high_order_T = row.high_order_T;
end

function metric = evaluate_repeated( ...
    cube, gt, selected_bands, base_eval_cfg, classification_seeds)

num_runs = numel(classification_seeds);
OA_values = zeros(num_runs, 1);
AA_values = zeros(num_runs, 1);
Kappa_values = zeros(num_runs, 1);

for irun = 1:num_runs
    eval_cfg = base_eval_cfg;
    eval_cfg.seed = classification_seeds(irun);
    eval_cfg.num_bands = numel(selected_bands);
    fprintf('[CLASSIFY %d/%d] top-%d, seed=%d\n', ...
        irun, num_runs, numel(selected_bands), eval_cfg.seed);

    eval_result = evaluate_svm_once(cube, gt, selected_bands, eval_cfg);
    OA_values(irun) = eval_result.OA;
    AA_values(irun) = eval_result.AA;
    Kappa_values(irun) = eval_result.Kappa;
end

metric.OA_mean = mean(OA_values);
metric.OA_std = std(OA_values);
metric.AA_mean = mean(AA_values);
metric.AA_std = std(AA_values);
metric.Kappa_mean = mean(Kappa_values);
metric.Kappa_std = std(Kappa_values);
end

function best_indices = choose_best_results_by_band_count( ...
    results_table, topk_list)

best_indices = zeros(numel(topk_list), 1);

for ik = 1:numel(topk_list)
    num_bands = topk_list(ik);
    candidates = find(results_table.NumBands == num_bands & ...
        isfinite(results_table.OA_mean_percent));

    if isempty(candidates)
        error(['No successful result is available for top-%d; cannot ', ...
               'produce the requested complete best-results table.'], num_bands);
    end

    % Within this band count: highest mean OA, then lowest OA std, then
    % the earlier search trial for deterministic tie-breaking.
    keys = [-results_table.OA_mean_percent(candidates), ...
             results_table.OA_std_percent(candidates), ...
             results_table.Trial(candidates)];
    [~, order] = sortrows(keys, [1, 2, 3]);
    best_indices(ik) = candidates(order(1));
end
end

function values = validate_positive_integers(values, name)
values = validate_integer_values(values, name);
if any(values <= 0)
    error('cfg.%s must contain positive integers.', name);
end
end

function values = validate_integer_values(values, name)
if ~isnumeric(values) || isempty(values) || ...
        any(~isfinite(values(:))) || any(values(:) ~= round(values(:)))
    error('cfg.%s must contain finite integer values.', name);
end
values = unique(values(:)', 'stable');
end
