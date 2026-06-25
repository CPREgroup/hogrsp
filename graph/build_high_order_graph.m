function [F, Lf] = build_high_order_graph(C, T)
%BUILD_HIGH_ORDER_GRAPH Build F=sum_{t=1}^T C^t and its Laplacian.

n = size(C, 1);

if size(C,1) ~= size(C,2)
    error('C must be a square matrix.');
end

C = max(C, 0);
C = (C + C') / 2;
C(1:n+1:end) = 0;

row_sum = sum(C, 2);
row_sum(row_sum == 0) = 1;
C_norm = C ./ row_sum;

F = zeros(n, n);
C_power = C_norm;

for t = 1:T
    C_t = (C_power + C_power') / 2;
    C_t(1:n+1:end) = 0;
    F = F + C_t;

    C_power = C_power * C_norm;
end

F = max(F, 0);
F = (F + F') / 2;
F(1:n+1:end) = 0;

max_val = max(F(:));
if max_val > 0
    F = F / max_val;
end

D = diag(sum(F, 2));
Lf = D - F;

end
