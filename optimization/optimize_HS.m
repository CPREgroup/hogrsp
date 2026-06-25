function [H, S, loss_history] = optimize_HS(X, F, Lf, H, S, G, cfg)
%OPTIMIZE_HS Alternately update S and H until convergence.

max_iter = cfg.opt.max_iter;
tol = cfg.opt.tol;

loss_history.total = zeros(max_iter, 1);
loss_history.rec = zeros(max_iter, 1);
loss_history.sparse = zeros(max_iter, 1);
loss_history.graph = zeros(max_iter, 1);
loss_history.lap = zeros(max_iter, 1);
loss_history.orth = zeros(max_iter, 1);

prev_loss = inf;

fprintf('\n========== Start Alternating Optimization ==========\n');

for iter = 1:max_iter

    % Update S with H fixed.
    S = update_S_qp(X, H, F, cfg);

    % Update H with S fixed.
    H = update_H_spectral(X, S, Lf, H, G, cfg);

    % Evaluate the objective.
    [loss, terms] = compute_objective(X, H, S, F, Lf, G, cfg);

    loss_history.total(iter) = loss;
    loss_history.rec(iter) = terms.rec;
    loss_history.sparse(iter) = terms.sparse;
    loss_history.graph(iter) = terms.graph;
    loss_history.lap(iter) = terms.lap;
    loss_history.orth(iter) = terms.orth;

    fprintf('[Iter %03d] total=%.6e | rec=%.3e | sparse=%.3e | graph=%.3e | lap=%.3e | orth=%.3e\n', ...
        iter, loss, terms.rec, terms.sparse, terms.graph, terms.lap, terms.orth);

    % Stop when the relative objective change is small enough.
    if iter > 1
        rel_change = abs(prev_loss - loss) / max(abs(prev_loss), 1e-12);
        if rel_change < tol
            fprintf('[INFO] Converged at iter %d, relative change = %.3e\n', iter, rel_change);
            break;
        end
    end

    prev_loss = loss;
end

% Remove unused preallocated entries.
fields = fieldnames(loss_history);
for i = 1:numel(fields)
    loss_history.(fields{i}) = loss_history.(fields{i})(1:iter);
end

fprintf('========== Optimization Finished ==========\n');

end
