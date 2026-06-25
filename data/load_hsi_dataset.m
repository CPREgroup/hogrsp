function [cube, gt, ers_labels] = load_hsi_dataset(cfg)
%LOAD_HSI_DATASET Load HSI cube, ground truth, and ERS labels.

%% Load HSI cube
if exist(cfg.path.hsi_file, 'file') ~= 2
    error('HSI file does not exist: %s', cfg.path.hsi_file);
end

hsi_data = load(cfg.path.hsi_file);
hsi_names = fieldnames(hsi_data);

cube = [];
cube_name = "";

for i = 1:numel(hsi_names)
    value = hsi_data.(hsi_names{i});

    if isnumeric(value) && ndims(value) == 3
        cube = value;
        cube_name = string(hsi_names{i});
        break;
    end
end

if isempty(cube)
    error('No 3-D HSI variable found in %s.', cfg.path.hsi_file);
end

cube = double(cube);

%% Load ground truth
if exist(cfg.path.gt_file, 'file') ~= 2
    error('GT file does not exist: %s', cfg.path.gt_file);
end

gt_data = load(cfg.path.gt_file);
gt_names = fieldnames(gt_data);

gt = [];
gt_name = "";

for i = 1:numel(gt_names)
    value = gt_data.(gt_names{i});

    if isnumeric(value) && ismatrix(value)
        gt = value;
        gt_name = string(gt_names{i});
        break;
    end
end

if isempty(gt)
    error('No 2-D GT variable found in %s.', cfg.path.gt_file);
end

gt = double(gt);

%% Load ERS labels
if exist(cfg.path.ers_file, 'file') ~= 2
    error('ERS label file does not exist: %s', cfg.path.ers_file);
end

ers_data = load(cfg.path.ers_file);

if isfield(ers_data, 'labels')
    ers_labels = ers_data.labels;
    ers_name = "labels";
elseif isfield(ers_data, 'ers_labels')
    ers_labels = ers_data.ers_labels;
    ers_name = "ers_labels";
elseif isfield(ers_data, 'label')
    ers_labels = ers_data.label;
    ers_name = "label";
else
    error('No labels / ers_labels / label variable found in %s.', cfg.path.ers_file);
end

ers_labels = double(ers_labels);

%% Size checks
if size(cube,1) ~= size(gt,1) || size(cube,2) ~= size(gt,2)
    error('cube and gt spatial sizes do not match: cube=%d x %d, gt=%d x %d', ...
        size(cube,1), size(cube,2), size(gt,1), size(gt,2));
end

if size(cube,1) ~= size(ers_labels,1) || size(cube,2) ~= size(ers_labels,2)
    error('cube and ers_labels spatial sizes do not match: cube=%d x %d, ers=%d x %d', ...
        size(cube,1), size(cube,2), size(ers_labels,1), size(ers_labels,2));
end

fprintf('[INFO] dataset_name: %s\n', cfg.dataset_name);
fprintf('[INFO] HSI variable name: %s\n', cube_name);
fprintf('[INFO] GT variable name: %s\n', gt_name);
fprintf('[INFO] ERS variable name: %s\n', ers_name);

end
