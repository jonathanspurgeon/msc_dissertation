%%
fprintf('SiO2 \n \n');  

% Download SiO2.mat problem from https://sparse.tamu.edu/

% Load l.h.s. matrix.
t = open('./Matrices/SiO2.mat');
A = t.Problem.A;
clear t

% no preconditioner.
n = size(A,1);

% Define r.h.s.
b = randn(n,1);
b = b / norm(b);

% low-precision (reduced memory) rand-GMRES 
%m = 400;
%plot_figures_sgmres(A, b, m, 1e-10, 5)


tol = 1e-10;
m = 400;
maxit = 5;
plot_figures_sgmres(A, b, m , tol, maxit)
%%
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
[u_gmres, ~] = sgmres_copy(A, b, param);
t_dr = toc

%%
fprintf('low-precision (reduced memory) rand-GMRES... \n');  
tic
sol = randgmres(A,b, m, tol, maxit);
CPUTIME1 = toc
%% 
fprintf('atmosmodd \n \n');  

% Download atmosmodd.mat problem from https://sparse.tamu.edu/

% Load l.h.s. matrix.
t = open('./Matrices/atmosmodd.mat');
A = t.Problem.A;
clear t

% no preconditioner.
n = size(A,1);

% Define r.h.s.
b = randn(n,1);
b = b / norm(b);

% low-precision (reduced memory) rand-GMRES 
m = 300;
plot_figures_sgmres(A, b, m, 1e-10, 5)

%% 
fprintf('vas_stokes_1M \n \n');  

% Download atmosmodd.mat problem from https://sparse.tamu.edu/

% Load l.h.s. matrix.
t = open('./Matrices/vas_stokes_1M.mat');
A = t.Problem.A;
clear t

n = size(A,1);
[L,U] = ilu(A);
PA =@(x) U\(L\(A *x));

m = 100;          % max Arnoldi cycle length
nrestarts = 10;   % max number of restarts
tol = 1e-6;       % residual tolerance

% Create rhs and precondition
rng('default')
b = randn(n,1);
Pb = U\(L\b);
% we normalize Pb to compare different methods which may use relative or absolute residual
bet = norm(Pb);   
Pb = Pb/bet;

plot_figures_sgmres(PA, Pb, m, tol, nrestarts)

%% 
fprintf('Case I: Poisson Equation \n \n');  

% Load l.h.s. matrix.
N = 1e6;
p = 0;
q = 1;
f = @(x) ones(size(x));
[A, b] = PDE_cases('poisson', N, p, q, f);
%b = b / norm(b);

% no preconditioner.
[L,U] = ilu(A);
PA = @(x) U\(L\(A *x));

m = 100;          % max Arnoldi cycle length
nrestarts = 10;   % max number of restarts
tol = 1e-10;       % residual tolerance

% Create rhs and precondition
Pb = U\(L\b);
% we normalize Pb to compare different methods which may use relative or absolute residual
bet = norm(Pb);   
Pb = Pb/bet;

% low-precision (reduced memory) rand-GMRES 
plot_figures_sgmres(PA, Pb, m, tol, nrestarts)

%% 
fprintf('Case II: Advection Equation \n \n');  

% Load l.h.s. matrix.
N = 1e4;
p = 0;
q = 1;
f = @(x) ones(size(x));
[A, b] = PDE_cases('advection', N, p, q, f, 1e4);
b = b / norm(b);

m = 100;          % max Arnoldi cycle length
nrestarts = 10;   % max number of restarts
tol = 1e-10;       % residual tolerance

n = size(b,1);

plot_figures_sgmres(A, b, m, tol, nrestarts)
