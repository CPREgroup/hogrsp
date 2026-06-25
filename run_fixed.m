clc; clear; close all;

%% Paths
project_root = 'E:\HOGSP_BS';
dataset_root = fullfile(project_root, 'Dataset');

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

pca_path = which('pca');
fprintf('[INFO] pca function: %s\n', pca_path);

%% Configuration
base_cfg = get_config();
base_cfg.path.dataset_dir = dataset_root;

topk_list = [5, 10, 15, 20, 25, 30];
dataset_keys = string(base_cfg.datasets.enabled);

fprintf('========== HOGSP Fixed-Parameter Run ==========\n');
fprintf('project_root = %s\n', project_root);
fprintf('datasets     = %s\n', strjoin(dataset_keys, ', '));
fprintf('topk_list    = %s\n', strjoin(string(topk_list), ' '));

%% Main loop
for idata = 1:numel(dataset_keys)
    dataset_key = dataset_keys(idata);
    dataset_key_char = char(dataset_key);

    if ~isfield(base_cfg.datasets, dataset_key_char)
        error('Unknown dataset key: %s', dataset_key_char);
    end

    if ~isfield(base_cfg.fixed_params, dataset_key_char)
        error('Missing fixed parameters for dataset: %s', dataset_key_char);
    end

    dataset_cfg = base_cfg.datasets.(dataset_key_char);
    fixed = base_cfg.fixed_params.(dataset_key_char);

    cfg = base_cfg;
    cfg.dataset_name = dataset_cfg.name;
    cfg.path.hsi_file = fullfile(dataset_root, dataset_cfg.hsi_file);
    cfg.path.gt_file  = fullfile(dataset_root, dataset_cfg.gt_file);
    cfg.path.ers_file = fullfile(dataset_root, dataset_cfg.ers_file);

    cfg.opt.alpha = fixed.alpha;
    cfg.opt.lambda = fixed.lambda;
    cfg.opt.eta = fixed.eta;
    cfg.opt.gamma = fixed.gamma;
    cfg.graph.knn_k = fixed.knn_k;
    cfg.graph.high_order_T = fixed.high_order_T;
    cfg.opt.subspace_dim = fixed.subspace_dim;

    cfg.exp_name = sprintf('%s_a%.4g_lam%.4g_eta%.4g_g%.4g_knn%d_k%d_T%d', ...
        cfg.dataset_name, ...
        cfg.opt.alpha, ...
        cfg.opt.lambda, ...
        cfg.opt.eta, ...
        cfg.opt.gamma, ...
        cfg.graph.knn_k, ...
        cfg.opt.subspace_dim, ...
        cfg.graph.high_order_T);

    fprintf('\n%s\n', repmat('#', 1, 90));
    fprintf('[DATASET] %d / %d\n', idata, numel(dataset_keys));
    fprintf('exp_name = %s\n', cfg.exp_name);
    fprintf('%s\n', repmat('#', 1, 90));

    try
        [cube, gt, ers_labels] = load_hsi_dataset(cfg);
        cube = normalize_cube(cube);

        [X, ~] = build_region_samples(cube, ers_labels, gt);

        C = build_knn_graph(X, cfg.graph.knn_k, cfg.graph.sigma);
        [F, Lf] = build_high_order_graph(C, cfg.graph.high_order_T);

        [H, S, G] = initialize_variables(X, F, cfg);
        [H, S, ~] = optimize_HS(X, F, Lf, H, S, G, cfg);

        [band_ranking, ~] = rank_bands_by_H(H);

        OA_sum = 0;
        results = struct( ...
            'num_bands', {}, ...
            'OA', {}, ...
            'AA', {}, ...
            'Kappa', {}, ...
            'selected_bands_1based', {}, ...
            'selected_bands_0based', {});

        for ik = 1:numel(topk_list)
            num_bands = topk_list(ik);

            if numel(band_ranking) < num_bands
                fprintf('[SKIP] top-%d exceeds available bands %d.\n', ...
                    num_bands, numel(band_ranking));
                continue;
            end

            selected_bands_1based = band_ranking(1:num_bands);
            selected_bands_0based = selected_bands_1based - 1;

            eval_cfg = cfg.eval;
            eval_cfg.num_bands = num_bands;

            eval_result = evaluate_svm_once( ...
                cube, gt, selected_bands_1based, eval_cfg);

            OA_sum = OA_sum + eval_result.OA;
            results(end + 1) = struct( ...
                'num_bands', num_bands, ...
                'OA', eval_result.OA, ...
                'AA', eval_result.AA, ...
                'Kappa', eval_result.Kappa, ...
                'selected_bands_1based', selected_bands_1based(:)', ...
                'selected_bands_0based', selected_bands_0based(:)');
        end

        fprintf('\n%s\n', repmat('=', 1, 90));
        fprintf('Fixed-parameter result: %s\n', cfg.dataset_name);
        fprintf('%s\n', repmat('=', 1, 90));
        fprintf('OA_sum = %.6f\n', OA_sum);
        fprintf('alpha  = %.6g\n', cfg.opt.alpha);
        fprintf('lambda = %.6g\n', cfg.opt.lambda);
        fprintf('eta    = %.6g\n', cfg.opt.eta);
        fprintf('gamma  = %.6g\n', cfg.opt.gamma);
        fprintf('knn_k  = %d\n', cfg.graph.knn_k);
        fprintf('dim    = %d\n', cfg.opt.subspace_dim);
        fprintf('T      = %d\n', cfg.graph.high_order_T);

        for ir = 1:numel(results)
            row = results(ir);
            fprintf('[TOP-%02d] OA=%.6f AA=%.6f Kappa=%.6f bands1=%s bands0=%s\n', ...
                row.num_bands, ...
                row.OA, ...
                row.AA, ...
                row.Kappa, ...
                strjoin(string(row.selected_bands_1based), ' '), ...
                strjoin(string(row.selected_bands_0based), ' '));
        end

    catch ME
        fprintf('[ERROR] Dataset failed: %s\n', cfg.dataset_name);
        fprintf('%s\n', ME.message);

        for si = 1:numel(ME.stack)
            fprintf('  at %s, line %d\n', ME.stack(si).name, ME.stack(si).line);
        end
    end
end

fprintf('\n========== Done ==========\n');
