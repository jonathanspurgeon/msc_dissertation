function times = plot_figures_sgmres(A, b, m, tol, maxit, truncation, k)
% plot_figures_gmres_sdr_variants(A, b, m, tol, maxit, truncation, k)
%
% Compare GMRES-SDR variants:
% - sGMRES (standard cycle)
% - k-truncated sGMRES
% - sGMRES-DR
% - MATLAB built-in GMRES
%
% Inputs:
%   A, b        : system
%   m           : inner iterationsZ
%   tol         : stopping tolerance
%   maxit       : max restarts
%   truncation  : truncation parameter (for truncated variant)
%   k           : recycling dim (for DR variant)
%
% Output:
%   times       : [t_std; t_trunc; t_dr; t_builtin]

n = size(b,1);
if nargin < 7, k = 20; end
if nargin < 6, truncation = 2; end

figure; hold on;

%% ---- Standard sGMRES (standard cycle)

fprintf('Running sGMRES (standard cycle)...\n');
param = struct( ...
    'max_it', m, ...
    'max_restarts', maxit, ...
    'tol', tol, ...
    'cycle_type', 'standard', ...
    'verbose', 1, ...
    's', min(n,ceil(2*m*log(n)/log(m))), ...
    'pert', 0 ...
    );

tic
[~, out_std] = sgmres_copy(A, b, param);
t_std = toc;

semilogy(out_std.sres, '-o', 'LineWidth', 1.5, 'MarkerSize', 5);

%% ---- k-truncated sGMRES
fprintf('Running k-truncated sGMRES...\n');
param.cycle_type = 'truncated';
param.t = truncation;

tic
[~, out_trunc] = sgmres_copy(A, b, param);
t_trunc = toc;

semilogy(out_trunc.sres, '-s', 'LineWidth', 1.5, 'MarkerSize', 5);

%% ---- sGMRES-DR (SDR cycle with recycling)
fprintf('Running sGMRES-DR...\n');
param.cycle_type = 'sdr';
param.k = k;

tic
[~, out_dr] = sgmres_copy(A, b, param);
t_dr = toc;

semilogy(out_dr.sres, '-d', 'LineWidth', 1.5, 'MarkerSize', 5);

%% ---- MATLAB built-in GMRES
fprintf('Running built-in MATLAB GMRES...\n');
tic
[~, ~, ~, ~, resvec] = gmres(A, b, m, tol, maxit);
t_builtin = toc;

semilogy(resvec, '-x', 'LineWidth', 1.5, 'MarkerSize', 5);

%% ---- Finalize plot
xlabel('Iteration / Restart');
ylabel('Residual Norm (log scale)');
title('SGMRES Variants Residual Norm Comparison');
legend({ ...
    sprintf('sGMRES (standard) [%.2fs]', t_std), ...
    sprintf('k-truncated sGMRES [%.2fs]', t_trunc), ...
    sprintf('sGMRES-DR [%.2fs]', t_dr), ...
    sprintf('MATLAB GMRES [%.2fs]', t_builtin)}, ...
    'Location', 'best');
set(gca, 'YScale', 'log');
grid on; hold off;

%% ---- Output timing vector
times = [t_std; t_trunc; t_dr; t_builtin];

end
