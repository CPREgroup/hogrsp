function [X, region_ids] = build_region_samples(cube, ers_labels, gt)
%BUILD_REGION_SAMPLES Build region-level spectral samples from ERS labels.
%
% X is d x n, where each column is the mean spectrum of one valid region.

[H, W, d] = size(cube);

if size(ers_labels,1) ~= H || size(ers_labels,2) ~= W
    error('ERS label size does not match cube spatial size.');
end

use_gt_mask = nargin >= 3 && ~isempty(gt);

if use_gt_mask
    if size(gt,1) ~= H || size(gt,2) ~= W
        error('GT size does not match cube spatial size.');
    end

    valid_mask = gt > 0;
else
    valid_mask = true(H, W);
end

region_ids_all = unique(ers_labels(:));
region_ids_all = region_ids_all(:);
region_ids_all(region_ids_all < 0) = [];

cube_2d = reshape(cube, [], d);       % (H*W) x d
label_vec = ers_labels(:);            % (H*W) x 1
valid_vec = valid_mask(:);            % (H*W) x 1

X_list = {};
region_id_list = [];

for idx = 1:numel(region_ids_all)
    rid = region_ids_all(idx);

    mask = (label_vec == rid) & valid_vec;

    if ~any(mask)
        continue;
    end

    region_pixels = cube_2d(mask, :);       % num_valid_pixels x d
    region_mean = mean(region_pixels, 1)';  % d x 1

    if norm(region_mean) > 0
        X_list{end+1} = region_mean;
        region_id_list(end+1, 1) = rid;
    end
end

if isempty(X_list)
    error('No valid region samples were built. Check GT and ERS labels.');
end

X = cat(2, X_list{:});
region_ids = region_id_list;

fprintf('[INFO] Background removed: kept %d valid regions out of %d total regions.\n', ...
    numel(region_ids), numel(region_ids_all));

end
