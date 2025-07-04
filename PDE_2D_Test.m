function f_val = gaussian_sources(x, y)
    % Four Gaussian sources at specified locations with small standard deviation
    % Sources at: (0.25, 0.75), (0.25, 0.25), (0.75, 0.25), (0.75, 0.75)
    
    % Standard deviation (small to keep sources localized)
    sigma = 0.05;
    
    % Amplitude of each Gaussian
    amplitude = 1;
    
    % Source locations
    sources = [0.25, 0.75;   % Source 1
               0.25, 0.25;   % Source 2  
               0.75, 0.25;   % Source 3
               0.75, 0.75];  % Source 4
    
    % Initialize output
    f_val = zeros(size(x));
    
    % Add each Gaussian source
    for i = 1:size(sources, 1)
        x_center = sources(i, 1);
        y_center = sources(i, 2);
        
        % Gaussian function: A * exp(-((x-x0)^2 + (y-y0)^2)/(2*sigma^2))
        f_val = f_val + amplitude * exp(-((x - x_center).^2 + (y - y_center).^2) / (2 * sigma^2));
    end
end

N = 100;  % Higher resolution to capture Gaussian sources
a = 1; b = 1; c = 0; d = 0; e = -1;

% Right-hand side function with Gaussian sources
f_func = @gaussian_sources;

% Solve with default zero Dirichlet BCs
[K, f_vec] = PDE_2D_Case(N, a, b, c, d, e, f_func);

n = size(K,1);
m=300; maxit = 5; tol = 1e-10; k =20;
param = struct( ...
    'max_it', m, ...
    'max_restarts', maxit, ...
    'tol', tol, ...
    'cycle_type', 'sdr', ...
    'verbose', 1, ...
    's', min(n,ceil(2*m*log(n)/log(m))), ...
    'pert', 0, ...
    'k', k ...
    );

% Solve the linear system
tic
[u_vec, ~] = sgmres(K, f_vec, param);
t_dr = toc

% Reshape solution to grid
U = reshape(u_vec, N, N);

% Display results
fprintf('Example 1: Gaussian Sources with Zero Dirichlet BCs\n');
fprintf('System size: %d x %d\n', size(K,1), size(K,2));
fprintf('Max solution value: %.4f\n', max(u_vec));

% Create coordinate grids for plotting
h = 1/(N+1);
x = h:h:1-h;
y = h:h:1-h;
[X, Y] = meshgrid(x, y);

% Plot the source function
figure;
F_plot = gaussian_sources(X, Y);
surf(X, Y, F_plot);
title('Source Function f(x,y) - Four Gaussian Sources');
xlabel('x'); ylabel('y'); zlabel('f(x,y)');
colorbar;
view(45, 30);

% Plot solution
figure;
surf(X, Y, U');
title('Solution u(x,y) - Response to Gaussian Sources');
xlabel('x'); ylabel('y'); zlabel('u');
colorbar;
view(45, 30);

% Plot contour of solution
figure;
contourf(X, Y, U', 20);
title('Solution Contours - Four Gaussian Source Response');
xlabel('x'); ylabel('y');
colorbar;
axis equal;
