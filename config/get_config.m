function cfg = get_config()

%% Dataset selection
% A normal run processes exactly one dataset. The key is case-insensitive.
% Available keys: "IndianPines", "PaviaU", "Salinas", "KSC", "Houston", "Botswana".
cfg.dataset_name = "IndianPines";

project_root = fileparts(fileparts(mfilename('fullpath')));
cfg.path.dataset_dir = fullfile(project_root, 'Dataset');

%% Dataset file catalogue
cfg.datasets.IndianPines.name = "IndianPines";
cfg.datasets.IndianPines.hsi_file = 'Indian_pines_corrected.mat';
cfg.datasets.IndianPines.gt_file  = 'Indian_pines_gt.mat';
cfg.datasets.IndianPines.ers_file = 'Indian_Pines_ers_labels.mat';

cfg.datasets.PaviaU.name = "PaviaU";
cfg.datasets.PaviaU.hsi_file = 'paviaU.mat';
cfg.datasets.PaviaU.gt_file  = 'paviaU_gt.mat';
cfg.datasets.PaviaU.ers_file = 'Pavia_University_ers_labels.mat';

cfg.datasets.Salinas.name = "Salinas";
cfg.datasets.Salinas.hsi_file = 'salinas_corrected.mat';
cfg.datasets.Salinas.gt_file  = 'salinas_gt.mat';
cfg.datasets.Salinas.ers_file = 'Salinas_ers_labels.mat';

cfg.datasets.KSC.name = "KSC";
cfg.datasets.KSC.hsi_file = 'KSC.mat';
cfg.datasets.KSC.gt_file  = 'KSC_gt.mat';
cfg.datasets.KSC.ers_file = 'KSC_ers_labels.mat';

cfg.datasets.Houston.name = "Houston";
cfg.datasets.Houston.hsi_file = 'houston.mat';
cfg.datasets.Houston.gt_file  = 'houston_gt.mat';
cfg.datasets.Houston.ers_file = 'houston_ers_labels.mat';

cfg.datasets.Botswana.name = "Botswana";
cfg.datasets.Botswana.hsi_file = 'Botswana.mat';
cfg.datasets.Botswana.gt_file  = 'Botswana_gt.mat';
cfg.datasets.Botswana.ers_file = 'Botswana_ers_labels.mat';

%% Hyperparameter search space
% Every combination of these values is evaluated (Cartesian grid search).
% Add/remove values to expand/reduce the search. The defaults produce
% 2 x 2 x 2 x 2 = 16 parameter combinations for Indian Pines.
cfg.search.alpha = [0.002];
cfg.search.lambda = [50];
cfg.search.eta = [0.01];
cfg.search.gamma = [0.005];
cfg.search.knn_k = 3;
cfg.search.subspace_dim = 20;
cfg.search.high_order_T = 3;
cfg.search.topk_list = [5, 10, 15, 20, 25, 30];

%% Graph settings shared by all search trials
cfg.graph.sigma = 1.0;

%% Optimization settings shared by all search trials
cfg.opt.max_iter = 50;
cfg.opt.tol = 1e-5;
cfg.opt.print_interval = 10;
cfg.opt.eps_G = 1e-6;
cfg.opt.eps_DH = 1e-8;
cfg.opt.use_scale_floor = true;
cfg.opt.scale_floor = 1e-3;
cfg.opt.use_quadprog = true;
cfg.opt.qp_display = 'off';

% These scalar defaults keep the auxiliary ablation scripts usable. The
% main run.m overwrites them with every combination in cfg.search.
cfg.opt.alpha = cfg.search.alpha(1);
cfg.opt.lambda = cfg.search.lambda(1);
cfg.opt.eta = cfg.search.eta(1);
cfg.opt.gamma = cfg.search.gamma(1);
cfg.opt.subspace_dim = cfg.search.subspace_dim(1);
cfg.graph.knn_k = cfg.search.knn_k(1);
cfg.graph.high_order_T = cfg.search.high_order_T(1);

%% Classification evaluation
cfg.eval.train_ratio = 0.05;
cfg.eval.seeds = 42:44;
cfg.eval.seed = cfg.eval.seeds(1);
cfg.eval.svm_C = 100;
cfg.eval.svm_kernel = "RBF";
cfg.eval.svm_gamma = "scale";

end
