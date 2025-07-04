f_func = @(x) 3*sin(3*pi*x) - 2;
p = 1; q = 2;
N = 1e4; % for good resolution

[A, b] = PDE_cases('poisson', N, p, q, f_func);

[L,U] = ilu(A);
PA = @(x) U\(L\(A *x));
Pb = U\(L\b);

tol = 1e-8;
m = 300;
maxit = 5;

n = size(Pb,1);

param = struct( ...
    'max_it', m, ...
    'max_restarts', maxit, ...
    'tol', tol, ...
    'cycle_type', 'sdr', ...
    'verbose', 1, ...
    's', min(n,ceil(2*m*log(n)/log(m))), ...
    'pert', 0, ...
    'k', 20 ...
    );

tic
[u_gmres, ~] = sgmres(PA, Pb, param);
t_dr = toc;

h = 1 / (N + 1);
x_grid = linspace(h, 1 - h, N)';

u_exact = (1/(9*pi^2))*sin(3*pi*x_grid) + x_grid.^2 + 1;

error = norm(u_gmres - u_exact, inf);
fprintf('Infinity norm of error: %e\n', error);

plot(x_grid, u_gmres, 'b-', 'LineWidth', 1.5);
hold on;
plot(x_grid, u_exact, 'r--', 'LineWidth', 1.5);
legend('GMRES Solution', 'Exact Solution');
xlabel('x');
ylabel('u(x)');
title('Comparison of GMRES Solution with Exact Solution');
grid on;
hold off;


