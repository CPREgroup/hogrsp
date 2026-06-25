function result = evaluate_svm_once(cube, gt, selected_bands_1based, eval_cfg)
%EVALUATE_SVM_ONCE Evaluate selected bands with one SVM train/test split.

rng(eval_cfg.seed);

[H, W, ~] = size(cube); %#ok<ASGLU>

%% Extract selected bands
data_sel = cube(:, :, selected_bands_1based);
num_bands = numel(selected_bands_1based);

X_all = reshape(data_sel, [], num_bands);
y_all = gt(:);

%% Remove background
valid_idx = y_all > 0;

X_valid = X_all(valid_idx, :);
y_valid = y_all(valid_idx);

y_valid = double(y_valid);

classes = unique(y_valid);
num_classes = numel(classes);

fprintf('[SVM] valid pixels: %d, classes: %d\n', numel(y_valid), num_classes);

%% Stratified train/test split
train_mask = false(size(y_valid));
test_mask  = false(size(y_valid));

for ci = 1:num_classes
    c = classes(ci);
    idx_c = find(y_valid == c);
    n_c = numel(idx_c);

    n_train_c = max(1, round(eval_cfg.train_ratio * n_c));
    n_train_c = min(n_train_c, n_c - 1);

    perm = idx_c(randperm(n_c));

    train_idx_c = perm(1:n_train_c);
    test_idx_c  = perm(n_train_c+1:end);

    train_mask(train_idx_c) = true;
    test_mask(test_idx_c) = true;
end

X_train = X_valid(train_mask, :);
y_train = y_valid(train_mask);

X_test = X_valid(test_mask, :);
y_test = y_valid(test_mask);

fprintf('[SVM] train pixels: %d, test pixels: %d\n', numel(y_train), numel(y_test));

%% Standardize using training statistics
mu = mean(X_train, 1);
sigma = std(X_train, 0, 1);
sigma(sigma == 0) = 1;

X_train_std = (X_train - mu) ./ sigma;
X_test_std  = (X_test  - mu) ./ sigma;

%% Match sklearn gamma='scale' for RBF
global_var = var(X_train_std(:), 1);

if global_var <= 0 || isnan(global_var)
    global_var = 1;
end

gamma_scale = 1 / (num_bands * global_var);
kernel_scale = sqrt(1 / (2 * gamma_scale));

fprintf('[SVM] C=%.4g, kernel=RBF, gamma=scale=%.4e, KernelScale=%.4e\n', ...
    eval_cfg.svm_C, gamma_scale, kernel_scale);

%% Train multiclass SVM
template = templateSVM( ...
    'KernelFunction', 'rbf', ...
    'KernelScale', kernel_scale, ...
    'BoxConstraint', eval_cfg.svm_C, ...
    'Standardize', false);

svm_model = fitcecoc( ...
    X_train_std, y_train, ...
    'Learners', template, ...
    'Coding', 'onevsone', ...
    'ClassNames', classes);

%% Predict and evaluate
y_pred = predict(svm_model, X_test_std);

confusion_mat = confusionmat(y_test, y_pred, 'Order', classes);

OA = sum(diag(confusion_mat)) / sum(confusion_mat(:));

class_acc = diag(confusion_mat) ./ max(sum(confusion_mat, 2), 1);
AA = mean(class_acc);

N = sum(confusion_mat(:));
row_sum = sum(confusion_mat, 2);
col_sum = sum(confusion_mat, 1)';

pe = sum(row_sum .* col_sum) / (N^2);
Kappa = (OA - pe) / (1 - pe + eps);

%% Pack result
result = struct();

result.OA = OA;
result.AA = AA;
result.Kappa = Kappa;

result.class_acc = class_acc;
result.confusion_mat = confusion_mat;
result.classes = classes;

result.selected_bands_1based = selected_bands_1based(:)';
result.selected_bands_0based = selected_bands_1based(:)' - 1;

result.train_ratio = eval_cfg.train_ratio;
result.num_train = numel(y_train);
result.num_test = numel(y_test);
result.svm_C = eval_cfg.svm_C;
result.svm_kernel = eval_cfg.svm_kernel;
result.svm_gamma = eval_cfg.svm_gamma;
result.gamma_scale = gamma_scale;
result.kernel_scale = kernel_scale;

end
