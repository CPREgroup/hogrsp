function H_new = update_H_spectral(X, S, Lf, H_old, G, cfg)
%UPDATE_H_SPECTRAL 固定 S，基于广义特征分解更新 H

[d, n] = size(X);
k = cfg.opt.subspace_dim;

alpha = cfg.opt.alpha;
eta = cfg.opt.eta;
lambda = cfg.opt.lambda;

%% 构造 M
I_n = eye(n);
M = (I_n - S) * (I_n - S)';

%% 构造 D_H
row_norm = sqrt(sum(H_old.^2, 2));
D_diag = 1 ./ (2 * sqrt(row_norm.^2 + cfg.opt.eps_DH));
D_H = diag(D_diag);

%% 构造 Q
Q = X * M * X' + eta * X * Lf * X' + alpha * D_H;

% 数值对称化
Q = (Q + Q') / 2;
G = (G + G') / 2;

%% 广义特征值分解 Q h = mu G h
[V, D] = eig(Q, G);
mu = diag(D);

% 去除异常值
valid = isfinite(mu) & isreal(mu);
mu = real(mu(valid));
V = real(V(:, valid));

if numel(mu) < k
    error('有效广义特征值数量少于 k，请检查 Q 和 G。');
end

% 按特征值从小到大排序
[mu_sorted, order] = sort(mu, 'ascend');
V_sorted = V(:, order);

Vk = V_sorted(:, 1:k);
mu_k = mu_sorted(1:k);

%% G-归一化
for i = 1:k
    denom = sqrt(Vk(:,i)' * G * Vk(:,i));
    if denom > 0
        Vk(:,i) = Vk(:,i) / denom;
    end
end

%% 计算尺度 s_i
s2 = 1 - mu_k ./ (2 * lambda);
s2 = max(s2, 0);

if cfg.opt.use_scale_floor
    s2 = max(s2, cfg.opt.scale_floor^2);
end

s = sqrt(s2);

%% 诊断输出：检查 soft orthogonal 是否导致 H 退化
H_tmp = Vk * diag(s);
orth_tmp = norm(H_tmp' * G * H_tmp - eye(k), 'fro')^2;

fprintf('[H update] mu_min=%.4e, mu_max=%.4e, lambda=%.4e\n', ...
    min(mu_k), max(mu_k), lambda);

fprintf('[H update] s_min=%.4e, s_max=%.4e, num_floor=%d/%d\n', ...
    min(s), max(s), sum(s <= cfg.opt.scale_floor * 1.01), k);

fprintf('[H update] orth=%.4e\n', orth_tmp);

%% 更新 H
H_new = H_tmp;

end