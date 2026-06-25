function [loss, terms] = compute_objective(X, H, S, F, Lf, G, cfg)

alpha = cfg.opt.alpha;
gamma = cfg.opt.gamma;
eta = cfg.opt.eta;
lambda = cfg.opt.lambda;

Y = H' * X;

%% 1. self-representation loss
rec_mat = Y - Y * S;
terms.rec = norm(rec_mat, 'fro')^2;

%% 2. l2,1 sparse loss
row_norm = sqrt(sum(H.^2, 2));
terms.sparse = sum(row_norm);

%% 3. graph consistency loss
terms.graph = norm(S - F, 'fro')^2;

%% 4. Laplacian loss
terms.lap = trace(H' * X * Lf * X' * H);

%% 5. soft orthogonal loss
k = size(H, 2);
orth_mat = H' * G * H - eye(k);
terms.orth = norm(orth_mat, 'fro')^2;

%% total
loss = terms.rec ...
    + alpha * terms.sparse ...
    + gamma * terms.graph ...
    + eta * terms.lap ...
    + lambda * terms.orth;

end