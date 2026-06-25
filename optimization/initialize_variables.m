function [H, S, G] = initialize_variables(X, F, cfg)
%INITIALIZE_VARIABLES Initialize H, S, and G.
%
% X: d x n
% H: d x k
% S: n x n
% G: d x d

[d, n] = size(X);
k = cfg.opt.subspace_dim;

if k > d
    error('subspace_dim k=%d cannot be larger than band count d=%d.', k, d);
end

%% Build G
G = (X * X') / n + cfg.opt.eps_G * eye(d);

%% Initialize H with PCA
[coeff, ~, ~] = pca(X');

if size(coeff, 1) ~= d
    error('PCA initialization error: coeff should be %d x k, but got %d x %d.', ...
        d, size(coeff,1), size(coeff,2));
end

if size(coeff, 2) < k
    error('PCA returned only %d components, fewer than k=%d.', size(coeff,2), k);
end

H = coeff(:, 1:k);


%% Size checks
if size(H,1) ~= d
    error('Invalid H size.');
end

if size(G,1) ~= d || size(G,2) ~= d
    error('Invalid G size.');
end

if size(F,1) ~= n || size(F,2) ~= n
    error('Invalid F size.');
end

%% G-normalize H columns
for i = 1:k
    denom = sqrt(H(:,i)' * G * H(:,i));
    if denom > 0
        H(:,i) = H(:,i) / denom;
    end
end

%% Initialize S
S = F;
S = max(S, 0);
S(1:n+1:end) = 0;

end
