function plot_total_loss_curve(loss_history, cfg, output_dir)
%PLOT_TOTAL_LOSS_CURVE Plot and save the total optimization loss curve.

if nargin < 3 || isempty(output_dir)
    output_dir = pwd;
end

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

iters = 1:numel(loss_history.total);

fig = figure( ...
    'Name', sprintf('Total Loss Curve - %s', cfg.exp_name), ...
    'Color', 'w');

plot(iters, loss_history.total, '-o', ...
    'LineWidth', 1.6, ...
    'MarkerSize', 4);
grid on;
set(gca, 'FontSize', 12);
xlabel('number of iterations', 'FontSize', 13);
ylabel('Objective function value', 'FontSize', 13);

safe_name = regexprep(char(cfg.exp_name), '[^\w.-]', '_');
png_file = fullfile(output_dir, [safe_name, '_total_loss.png']);
fig_file = fullfile(output_dir, [safe_name, '_total_loss.fig']);

saveas(fig, png_file);
saveas(fig, fig_file);

fprintf('[INFO] Total loss curve saved: %s\n', png_file);

end
