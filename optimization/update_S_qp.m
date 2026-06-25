function S = update_S_qp(X, H, F, cfg)
%UPDATE_S_QP Update S column by column with H fixed.
%
% Subproblem:
%   min ||Y - YS||_F^2 + gamma ||S - F||_F^2
%   s.t. S >= 0, diag(S)=0

gamma = cfg.opt.gamma;

Y = H' * X;  % k x n
[~, n] = size(Y);

S = zeros(n, n);

use_quadprog = cfg.opt.use_quadprog;

if use_quadprog
    options = optimoptions('quadprog', ...
        'Display', cfg.opt.qp_display, ...
        'Algorithm', 'interior-point-convex');
end

for j = 1:n
    idx = true(n, 1);
    idx(j) = false;

    Y_tilde = Y(:, idx);       % k x (n-1)
    y_j = Y(:, j);             % k x 1
    f_tilde = F(idx, j);       % (n-1) x 1

    R = Y_tilde' * Y_tilde + gamma * eye(n-1);
    q = Y_tilde' * y_j + gamma * f_tilde;

    if use_quadprog
        % quadprog standard form:
        %   min 1/2 u' Hqp u + fqp' u
        % current form:
        %   min u' R u - 2 q' u
        Hqp = 2 * R;
        fqp = -2 * q;

        lb = zeros(n-1, 1);

        try
            u = quadprog(Hqp, fqp, [], [], [], [], lb, [], [], options);
        catch
            warning('quadprog failed at column %d; using nonnegative fallback.', j);
            u = max(R \ q, 0);
        end

        if isempty(u)
            warning('quadprog returned empty at column %d; using nonnegative fallback.', j);
            u = max(R \ q, 0);
        end
    else
        % Fast fallback: solve the unconstrained system and clamp to nonnegative.
        u = max(R \ q, 0);
    end

    s_j = zeros(n, 1);
    s_j(idx) = u;
    s_j(j) = 0;

    S(:, j) = s_j;
end

S = max(S, 0);
S(1:n+1:end) = 0;

end
