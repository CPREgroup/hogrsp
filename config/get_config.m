function cfg = get_config()

%% Path settings
cfg.path.dataset_dir = 'E:\HOGSP_BS\Dataset';

%% Dataset settings
% Enable one or more datasets by key.
% Available keys: "IndianPines", "PaviaU", "Salinas", "KSC", "Houston", "Botswana".
cfg.datasets.enabled = ["IndianPines", "PaviaU", "Salinas"];

%% Indian Pines
cfg.datasets.IndianPines.name = "IndianPines";
cfg.datasets.IndianPines.hsi_file = 'Indian_pines_corrected.mat';
cfg.datasets.IndianPines.gt_file  = 'Indian_pines_gt.mat';
cfg.datasets.IndianPines.ers_file = 'Indian_Pines_ers_labels.mat';

%% Pavia University
cfg.datasets.PaviaU.name = "PaviaU";
cfg.datasets.PaviaU.hsi_file = 'paviaU.mat';
cfg.datasets.PaviaU.gt_file  = 'paviaU_gt.mat';
cfg.datasets.PaviaU.ers_file = 'Pavia_University_ers_labels.mat';

%% Salinas
cfg.datasets.Salinas.name = "Salinas";
cfg.datasets.Salinas.hsi_file = 'salinas_corrected.mat';
cfg.datasets.Salinas.gt_file  = 'salinas_gt.mat';
cfg.datasets.Salinas.ers_file = 'Salinas_ers_labels.mat';

%% KSC
cfg.datasets.KSC.name = "KSC";
cfg.datasets.KSC.hsi_file = 'KSC.mat';
cfg.datasets.KSC.gt_file  = 'KSC_gt.mat';
cfg.datasets.KSC.ers_file = 'KSC_ers_labels.mat';

%% Houston
cfg.datasets.Houston.name = "Houston";
cfg.datasets.Houston.hsi_file = 'houston.mat';
cfg.datasets.Houston.gt_file  = 'houston_gt.mat';
cfg.datasets.Houston.ers_file = 'houston_ers_labels.mat';

%% Botswana
cfg.datasets.Botswana.name = "Botswana";
cfg.datasets.Botswana.hsi_file = 'Botswana.mat';
cfg.datasets.Botswana.gt_file  = 'Botswana_gt.mat';
cfg.datasets.Botswana.ers_file = 'Botswana_ers_labels.mat';

%% Graph defaults
cfg.graph.knn_k = 3;
cfg.graph.sigma = 1.0;
cfg.graph.high_order_T = 7;

%% Fixed dataset parameters
cfg.fixed_params.IndianPines.alpha = 0.002;
cfg.fixed_params.IndianPines.lambda = 50;
cfg.fixed_params.IndianPines.eta = 0.01;
cfg.fixed_params.IndianPines.gamma = 0.005;
cfg.fixed_params.IndianPines.knn_k = 3;
cfg.fixed_params.IndianPines.subspace_dim = 20;
cfg.fixed_params.IndianPines.high_order_T = 3;

cfg.fixed_params.PaviaU.alpha = 0.1;
cfg.fixed_params.PaviaU.lambda = 50;
cfg.fixed_params.PaviaU.eta = 0.1;
cfg.fixed_params.PaviaU.gamma = 3;
cfg.fixed_params.PaviaU.knn_k = 3;
cfg.fixed_params.PaviaU.subspace_dim = 20;
cfg.fixed_params.PaviaU.high_order_T = 3;

cfg.fixed_params.Salinas.alpha = 0.1;
cfg.fixed_params.Salinas.lambda = 50;
cfg.fixed_params.Salinas.eta = 0.5;
cfg.fixed_params.Salinas.gamma = 3;
cfg.fixed_params.Salinas.knn_k = 3;
cfg.fixed_params.Salinas.subspace_dim = 15;
cfg.fixed_params.Salinas.high_order_T = 5;

%% Optimization defaults
cfg.opt.subspace_dim = 10;

cfg.opt.alpha = 0.01;
cfg.opt.gamma = 1.0;
cfg.opt.eta = 0.1;
cfg.opt.lambda = 100;

cfg.opt.max_iter = 50;
cfg.opt.tol = 1e-5;

cfg.opt.eps_G = 1e-6;
cfg.opt.eps_DH = 1e-8;

cfg.opt.use_scale_floor = true;
cfg.opt.scale_floor = 1e-3;

cfg.opt.use_quadprog = true;
cfg.opt.qp_display = 'off';

%% Evaluation defaults
cfg.eval.train_ratio = 0.05;
cfg.eval.seed = 42;

cfg.eval.svm_C = 1000;
cfg.eval.svm_kernel = "RBF";
cfg.eval.svm_gamma = "scale";

end
