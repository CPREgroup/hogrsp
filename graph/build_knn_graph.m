function C = build_knn_graph(X, knn_k, sigma)
%BUILD_KNN_GRAPH Build a first-order KNN Gaussian graph.
%
% X is d x n, where each column is one sample.

[d, n] = size(X); %#ok<ASGLU>

if knn_k >= n
    error('knn_k must be smaller than the number of samples n.');
end

X_sample = X';  % n x d

dist_mat = pdist2(X_sample, X_sample, 'euclidean');

C = zeros(n, n);

for i = 1:n
    [~, idx] = sort(dist_mat(i, :), 'ascend');

    % Skip idx(1), which is the sample itself.
    neigh = idx(2:knn_k+1);

    for jj = 1:numel(neigh)
        j = neigh(jj);
        C(i, j) = exp(-dist_mat(i, j)^2 / (2 * sigma^2));
    end
end

C = (C + C') / 2;
C(1:n+1:end) = 0;

max_val = max(C(:));
if max_val > 0
    C = C / max_val;
end

end
