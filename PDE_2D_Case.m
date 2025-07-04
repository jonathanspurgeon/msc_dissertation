function [K, f_vec] = PDE_2D_Case(N, a, b, c, d, e, f_func, bc_type, bc_values)
% SOLVE_PDE_FD Solves -au_xx - bu_yy + cu_x + du_y + eu = f(x,y) using finite differences
%
% Inputs:
%   N        - Number of interior grid points in each direction (scalar)
%   a, b, c, d, e - PDE coefficients (scalars)
%   f_func   - Right-hand side function handle f(x,y) or matrix of values
%   bc_type  - Boundary condition type: 'dirichlet', 'neumann', or 'mixed'
%   bc_values - Structure containing boundary values:
%               .left, .right, .bottom, .top (vectors of length N+2)
%
% Outputs:
%   K        - System matrix (sparse, size N^2 x N^2)
%   f_vec    - Right-hand side vector (size N^2 x 1)
%
% The resulting system is: K * u_vec = f_vec
% where u_vec contains the interior grid points in column-major order
    if nargin < 8
        bc_type = 'dirichlet';
    end
    
    if nargin < 9 || isempty(bc_values)
        % Default: zero Dirichlet boundary conditions
        bc_values.left = zeros(N+2, 1);
        bc_values.right = zeros(N+2, 1);
        bc_values.bottom = zeros(N+2, 1);
        bc_values.top = zeros(N+2, 1);
    end
    % Grid spacing
    h = 1/(N+1);
    
    % Create coordinate vectors for interior points
    x = linspace(h, 1-h, N);
    y = linspace(h, 1-h, N);
    [X, Y] = meshgrid(x, y);
    
    % Create sparse matrices A, B, and I
    I = speye(N);
    
    % Matrix A for second derivative (1/h^2 * tridiagonal[-1, 2, -1])
    A = (1/h^2) * spdiags([-ones(N,1), 2*ones(N,1), -ones(N,1)], [-1, 0, 1], N, N);
    
    % Matrix B for first derivative (1/(2h) * tridiagonal[-1, 0, 1])
    B = (1/(2*h)) * spdiags([-ones(N,1), zeros(N,1), ones(N,1)], [-1, 0, 1], N, N);
    
    % Construct system matrix K using Kronecker products
    % K = a*I⊗A + b*A⊗I + c*I⊗B + d*B⊗I + e*I⊗I
    K = a * kron(I, A) + b * kron(A, I) + c * kron(I, B) + d * kron(B, I) + e * kron(I, I);
    
    % Evaluate right-hand side
    if isa(f_func, 'function_handle')
        F = f_func(X, Y);
    else
        F = f_func; % Assume it's already a matrix
    end
    
    % Convert F to vector (column-major order)
    f_vec = F(:);
    
    % Apply boundary conditions
    [K, f_vec] = apply_boundary_conditions(K, f_vec, N, h, bc_type, bc_values, a, b, c, d);
    
end

function [K, f_vec] = apply_boundary_conditions(K, f_vec, N, h, bc_type, bc_values, a, b, c, d)
% Apply boundary conditions to the system matrix and RHS vector
    
    switch lower(bc_type)
        case 'dirichlet'
            [K, f_vec] = apply_dirichlet_bc(K, f_vec, N, h, bc_values, a, b, c, d);
        case 'neumann'
            [K, f_vec] = apply_neumann_bc(K, f_vec, N, h, bc_values, a, b, c, d);
        case 'mixed'
            [K, f_vec] = apply_mixed_bc(K, f_vec, N, h, bc_values, a, b, c, d);
        otherwise
            error('Unsupported boundary condition type. Use: dirichlet, neumann, or mixed');
    end
end

function [K, f_vec] = apply_dirichlet_bc(K, f_vec, N, h, bc_values, a, b, c, d)
% Apply Dirichlet boundary conditions
    
    % Extract boundary values
    u_left = bc_values.left(2:N+1);    % Interior points on left boundary
    u_right = bc_values.right(2:N+1);  % Interior points on right boundary  
    u_bottom = bc_values.bottom(2:N+1); % Interior points on bottom boundary
    u_top = bc_values.top(2:N+1);      % Interior points on top boundary
    
    % Modify RHS vector for boundary contributions
    for j = 1:N
        for i = 1:N
            idx = (j-1)*N + i; % Linear index for point (i,j)
            
            % Left boundary (i=1)
            if i == 1
                f_vec(idx) = f_vec(idx) + (a/h^2) * u_left(j) - (c/(2*h)) * u_left(j);
            end
            
            % Right boundary (i=N)
            if i == N
                f_vec(idx) = f_vec(idx) + (a/h^2) * u_right(j) + (c/(2*h)) * u_right(j);
            end
            
            % Bottom boundary (j=1)
            if j == 1
                f_vec(idx) = f_vec(idx) + (b/h^2) * u_bottom(i) - (d/(2*h)) * u_bottom(i);
            end
            
            % Top boundary (j=N)
            if j == N
                f_vec(idx) = f_vec(idx) + (b/h^2) * u_top(i) + (d/(2*h)) * u_top(i);
            end
        end
    end
end

function [K, f_vec] = apply_neumann_bc(K, f_vec, N, h, bc_values, a, b, c, d)
% Apply Neumann boundary conditions (simplified implementation)
    warning('Neumann BC implementation is simplified. Full implementation requires ghost points.');
    % This is a placeholder - full Neumann BC implementation would require
    % modification of the system matrix and careful handling of ghost points
end

function [K, f_vec] = apply_mixed_bc(K, f_vec, N, h, bc_values, a, b, c, d)
% Apply mixed boundary conditions
    warning('Mixed BC implementation is simplified. Specify which boundaries have which type.');
    % This would combine Dirichlet and Neumann conditions on different boundaries
end
