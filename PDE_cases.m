function [A, b] = PDE_cases(pde_type, N, p, q, f, varargin)
    % pde_type: string, one of 'pure_diffusion', 'reaction_diffusion', 'convection_diffusion'
    % N: number of interior points
    % p, q: Dirichlet BCs at x=0 and x=1
    % f: function handle for f(x)
    % varargin: additional parameters (like c or gamma)

    switch lower(pde_type)
        case 'poisson'
            [A, b] = solve_poisson(N, p, q, f);
            desc = '-u_{xx} = f(x)';
        
        case 'advection'
            if length(varargin) < 1
                error('Please provide parameter c for reaction-diffusion.');
            end
            c = varargin{1};
            [A, b] = solve_modified_poisson(N, p, q, f, c);
            desc = sprintf('-u_{xx} + %.2f u = f(x)', c);
        
        case 'convection'
            if length(varargin) < 1
                error('Please provide parameter gamma for convection-diffusion.');
            end
            gamma = varargin{1};
            [A, b] = solve_convection_diffusion(N, p, q, f, gamma);
            desc = sprintf('-u_{xx} + %.2f u_x = f(x)', gamma);
        
        otherwise
            error('Unknown PDE type. Choose from: pure_diffusion, reaction_diffusion, convection_diffusion');
    end
end


function [A, b] = solve_poisson(N, p, q, f)
    % N: number of interior points
    % p: u(0) = p (left boundary)
    % q: u(1) = q (right boundary)
    % f: function handle for f(x)

    % Step size
    h = 1 / (N + 1);
    
    % Interior grid points
    x = linspace(h, 1 - h, N)'; % column vector

    % Construct A (NxN tridiagonal matrix)

    i = [1:N, 1:N-1, 2:N]; 
    j = [1:N, 2:N, 1:N-1]; 

    main_diag = (2 / h^2) * ones(1, N);
    off_diag  = (-1 / h^2) * ones(1, N-1);

    A = sparse(i, j, [main_diag, off_diag, off_diag], N, N);
    
    % Construct right-hand side vector b
    b = f(x); % f evaluated at interior points
    
    % Adjust for boundary conditions
    b(1)   = b(1)   + p / h^2;
    b(end) = b(end) + q / h^2;
end

function [A, b] = solve_modified_poisson(N, p, q, f, c)
    % N: number of interior points
    % p: Dirichlet BC at x=0
    % q: Dirichlet BC at x=1
    % f: function handle for f(x)
    % c: scalar constant

    % Step size
    h = 1 / (N + 1);

    % Interior grid points
    x = linspace(h, 1 - h, N)';  % column vector

    % Construct A (NxN tridiagonal matrix)
    main_diag = (2 / h^2 + c) * ones(1, N);
    off_diag = (-1 / h^2) * ones(1, N-1);

    i = [1:N, 1:N-1, 2:N]; 
    j = [1:N, 2:N, 1:N-1]; 
    A = sparse(i, j, [main_diag, off_diag, off_diag], N, N);

    % Construct b
    b = f(x);  % Evaluate f at interior points

    % Adjust b for Dirichlet boundary conditions
    b(1) = b(1) + p / h^2;
    b(end) = b(end) + q / h^2;
end

function [A, b] = solve_convection_diffusion(N, p, q, f, gamma)
    % N: number of interior points
    % p: Dirichlet BC at x=0
    % q: Dirichlet BC at x=1
    % f: function handle f(x)
    % gamma: convection coefficient (real number)

    % Step size
    h = 1 / (N + 1);
    
    % Interior grid
    x = linspace(h, 1 - h, N)';  % column vector

    % Finite difference coefficients
    % -u_xx -> (2 / h^2) on diagonal, (-1 / h^2) on off-diagonals
    % gamma*u_x -> (-gamma / (2h)) on lower diag, (+gamma / (2h)) on upper diag

    main_diag = (2 / h^2) * ones(1, N);
    lower_diag = (-1 / h^2 - gamma / (2 * h)) * ones(1, N-1);
    upper_diag = (-1 / h^2 + gamma / (2 * h)) * ones(1, N-1);

    % Assemble tridiagonal matrix A

    i = [1:N, 1:N-1, 2:N]; 
    j = [1:N, 2:N, 1:N-1]; 
    A = sparse(i, j, [main_diag, upper_diag, lower_diag], N, N);

    % RHS vector
    b = f(x);  % evaluate f at interior points

    % Adjust for Dirichlet BCs
    b(1)   = b(1)   + (1 / h^2 + gamma / (2 * h)) * p;
    b(end) = b(end) + (1 / h^2 - gamma / (2 * h)) * q;
end

