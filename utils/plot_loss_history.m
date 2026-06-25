function plot_loss_history(loss_history, cfg, output_dir)
%PLOT_LOSS_HISTORY Plot and save optimization loss curves.

if nargin < 3 || isempty(output_dir)
    output_dir = pwd;
end

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

iters = 1:numel(loss_history.total);
figure_name = sprintf('Loss Curve - %s', cfg.exp_name);

fig = figure('Name', figure_name, 'Color', 'w');

subplot(2, 1, 1);
plot(iters, loss_history.total, '-o', 'LineWidth', 1.6, 'MarkerSize', 4);
grid on;
xlabel('Iteration');
ylabel('Total loss');
title(sprintf('%s Total Loss', cfg.dataset_name), 'Interpreter', 'none');

subplot(2, 1, 2);
hold on;
plot(iters, loss_history.rec, '-o', 'LineWidth', 1.2, 'MarkerSize', 3);
plot(iters, loss_history.sparse, '-s', 'LineWidth', 1.2, 'MarkerSize', 3);
plot(iters, loss_history.graph, '-^', 'LineWidth', 1.2, 'MarkerSize', 3);
plot(iters, loss_history.lap, '-d', 'LineWidth', 1.2, 'MarkerSize', 3);
plot(iters, loss_history.orth, '-x', 'LineWidth', 1.2, 'MarkerSize', 4);
hold off;
grid on;
xlabel('Iteration');
ylabel('Loss term');
title('Loss Terms', 'Interpreter', 'none');
legend({'rec', 'sparse', 'graph', 'lap', 'orth'}, 'Location', 'best');

safe_name = regexprep(char(cfg.exp_name), '[^\w.-]', '_');
png_file = fullfile(output_dir, [safe_name, '_loss.png']);
fig_file = fullfile(output_dir, [safe_name, '_loss.fig']);

saveas(fig, png_file);
saveas(fig, fig_file);

fprintf('[INFO] Loss curve saved: %s\n', png_file);

end
