function [x,out] = sgmres_copy(A,b,param)
% [x,out] = GMRES_SDR(A,b,param)
%
% A function which solves a linear system Ax = b using GMRES
% with sketching and deflated restarting, optionally using:
%   - recycling with Ritz extraction (default)
%   - truncated Arnoldi sketched GMRES
%   - standard Arnoldi sketched GMRES
%
% INPUT:
%   A       : matrix or function handle
%   b       : RHS
%   param   : struct controlling all parameters including:
%       .cycle_type : 'sdr' (default), 'truncated', 'standard'
%
% OUTPUT:
%   x       : approximate solution
%   out     : struct containing diagnostic information

if isnumeric(A)
    A = @(v) A*v;
end

n = size(b,1);

% Defaults
param = default_param_fields(param, n);

if param.verbose
    disp(['  using sketching dimension of s = ' num2str(param.s)])
end

% Initialize solution and residual
if ~isfield(param,'x0') || isempty(param.x0)
    x = zeros(n,1);
    r = b;
else
    x = param.x0;
    r = b - A(x);
end

% Tracking
res = zeros(1, param.max_restarts + 1);
res(1) = norm(r);
iters = 0;
sres = NaN; % initially undefined

% Outer restarts
for restart = 1:param.max_restarts
    % ------------------------------------------
    % SWITCH BASED ON param.cycle_type
    % ------------------------------------------
    switch lower(param.cycle_type)
        case {'sdr', 'recycling'}
            [e, r, cycle_out] = srgmres_cycle(A, r, param);

            % Update recycling for next cycle
            param.U = cycle_out.U;
            param.SU = cycle_out.SU;
            param.SAU = cycle_out.SAU;

            % Potentially adjust sketch distortion
            param.sketch_distortion = cycle_out.sketch_distortion;

        case {'truncated'}
            [e, cycle_out] = sgmres_truncated_cycle(A, r, param);
            r = r - A(e); % compute updated residual

        case {'standard'}
            [e, cycle_out] = sgmres_standard_cycle(A, r, param);
            r = r - A(e); % compute updated residual

        otherwise
            error('param.cycle_type must be ''sdr'', ''truncated'', or ''standard''.');
    end

    % Update solution
    sres = [sres, cycle_out.sres];
    tmp = x + e;
    res(restart + 1) = norm(r);
    stag = cycle_out.stag;
    conv = 0;
    
    % Check accuracy of error estimator.
    if (sres(cycle_out.m + 1) < param.tol/4) && res(restart + 1) > 2*param.tol
        stag = 1;
    end
    
    % only display if the output flag is not used
    if param.verbose >=1 && res(restart + 1) < param.tol
        x = tmp;
        conv = 1;
        fprintf('\nsgmres-%s converged at outer(inner) iteration %.2d(%.2d)', param.cycle_type, restart, cycle_out.m+1);
        fprintf(' to relative residual %.1d. \n \n', res(restart+1));
        break
    elseif param.verbose >=1  &&  restart>1 && res(restart + 1) > res(restart)*(1-100*eps)
        stag = 1;
        fprintf('\nsgmres-%s stopped at outer(inner) iteration %.2d(%.2d)', param.cycle_type,restart, cycle_out.m +1);
        fprintf(' because the method stagnated with relative residual %.1d. \n \n', res(restart+1));
        break
    elseif param.verbose >=1 && restart == param.max_restarts
        x = tmp;           
        if stag == 1
            fprintf('\nsgmres-%s stopped at outer(inner) iteration %.2d(%.2d)', param.cycle_type, restart, cycle_out.m+1);
            fprintf(' because the method stagnated with relative residual %.1d. \n \n', res(restart+1));
            break
        else
            fprintf('\nsgmres-%s stopped with relative residual %.1d', param.cycle_type, res(restart+1));
            fprintf(' because the maximum number of iterations was reached. \n \n');
            break
        end
    elseif param.verbose >=1
        fprintf('\nsgmres-%s: outer iteration %.2d; residual %.1d. \n \n', param.cycle_type, restart, res(restart+1));
    end
    
    stag = 0;
    x = tmp;

end

% Pack outputs
out = cycle_out;
out.residuals = res;
out.iters = iters;
out.sres = sres;
out.stag = stag;
out.conv = conv;

end

% ------------------------------------------------
function param = default_param_fields(param, n)
% Fill in missing fields with defaults for consistency

fields_defaults = {
    'verbose', 1;
    'U', []; 'SU', []; 'SAU', [];
    'max_it', 400;
    'max_restarts', min(ceil(n/50), 5);
    'tol', 1e-6;
    'ssa', 0;
    't', 2;
    'k', 10;
    's', min(n, 8*(50 + 10));
    'reorth', 0;
    'lssolver', '5reorth';
    'hS', []; % set below
    'sketch_distortion', 1.4;
    'ls_solve', 'mgs';
    'svd_tol', 1e-15;
    'harmonic', 1;
    'd', 1;
    'cycle_type', 'sdr'; % <--- CYCLE SWITCH KNOB
    'lowprecision', 0;
    'pert', 0;
};

for i = 1:size(fields_defaults,1)
    field = fields_defaults{i,1};
    def = fields_defaults{i,2};
    if ~isfield(param,field) || isempty(param.(field))
        param.(field) = def;
    end
end

% Default embedding
if isempty(param.hS)
    param.hS = srft(n, param.s);
end

end

function [dx, out] = sgmres_truncated_cycle(A, r0, param)
% [dx, out] = sgmres_truncated_cycle(A, r0, param)
%
% A single cycle of *sketched GMRES with t-truncated Arnoldi* without recycling,
% consistent with gmres_sdr pipeline structure.
%
% INPUT:
%   A       : matrix or function handle
%   r0      : initial residual
%   param   : struct with fields:
%       param.max_it       : max inner iterations (m)
%       param.t            : truncation window (k)
%       param.tol          : tolerance for stopping
%       param.hS           : sketching operator handle
%       param.verbose      : verbosity
%
% OUTPUT:
%   dx      : approximate solution update
%   out     : struct with fields:
%       out.m              : # inner iterations used
%       out.skres          : sketched residual history
%       out.mv             : # matvecs
%       out.ip             : # inner products
%       out.sv             : # sketches

% Initialize counters
mv = 0;
ip = 0;
sv = 0;

% Unpack parameters
m = param.max_it;
t = param.t;
tol = param.tol;
hS = param.hS;
stag = 0;
sketch_distortion = param.sketch_distortion;

n = size(r0,1);
Sr0 = hS(r0);
sv = sv + 1;
[Q,R] = qr([],0);
beta = norm(r0);
ip = ip + 1;
if beta == 0
    dx = zeros(n,1);
    out.m = 0;
    out.sres = 0;
    out.mv = mv;
    out.ip = ip;
    out.sv = sv;
    return
end

V = zeros(n,m);
AV = zeros(n,m);
C = zeros(param.s, m);

V(:,1) = r0 / beta;
AV(:,1) = A(V(:,1)); mv = mv + 1;
C(:,1) = hS(AV(:,1)); sv = sv + 1;

sres = zeros(1,m);
sres(1) = beta;

for j = 1:m-1

    w = AV(:,j);
    
    % t-truncated Arnoldi orthogonalization
    j_start = max(1, j - t + 1);
    for i = j_start:j
        h = V(:,i)' * w;
        ip = ip + 1;
        w = w - V(:,i) * h;
    end

    w_norm = norm(w);
    ip = ip + 1;
    if w_norm == 0
        if param.verbose >= 1
            disp('Break: happy breakdown detected.');
        end
        break
    end

    V(:,j+1) = w / w_norm;
    AV(:,j+1) = A(V(:,j+1)); mv = mv + 1;
    C(:,j+1) = hS(AV(:,j+1)); sv = sv + 1;

    % Least squares solve using QR
    [Q,R] = qrupdate_gs(C(:,1:j+1),Q,R);
    y = R\(Q'*Sr0);

    sres(j+1) = norm(Sr0 - C(:,1:j+1)*y);

    if param.verbose >= 1 && (sres(j+1) < tol/4  ...
                || mod(j+1,floor(m/5)) == 0 || j == m-1 || stag == 1)
        fprintf('smgres-truncated: inner iteration %.2d; ',j+1)
        fprintf('estimated residual %.1d;\n',sres(j+1))
    end
    if (sres(j+1) < tol/4 || j == m-1 || stag == 1)
        break
    end
   
    % If the residual estimate is small enough (or we reached the max
    % number of iterations), then we form the full approximation
    % correction (without explicitly forming [U V(:,1:j)])
    if sres(j+1) < tol/sketch_distortion || j == m-1
        e = V(:,1:j+1) * y;

        % Compute true residual
        r = r0 - A(e);
        mv = mv + 1;
        
        nrmr = norm(r);
        ip = ip + 1;

        % potentially increase sketch_distortion
        if nrmr/sres(j+1) > sketch_distortion
            sketch_distortion = nrmr/sres(j+1);
            if param.verbose >= 1
                % please stop commenting these out! use verbose=0
                disp(['  sketch distortion increased to ' num2str(sketch_distortion)])
            end
        end
        
        if nrmr < sres(j+1)/2
            stag = 1;
        end

        if sres(j+1) < tol/4 || j == m-1 || stag == 1
            % only display if the output flag is not used
            if param.verbose>=1 && stag == 1
                fprintf('sgmres-truncated: inner iteration %.2d; ', j+1)
                fprintf('stagnated.\n')
            end
            break
        end
    end
end

dx = V(:,1:j+1) * y;

% Pack output
out.m = j+1;
out.sres = sres(1:j+1);
out.stag = stag;

end


function [e,r,out] = srgmres_cycle(A,r0,param)

max_it = param.max_it;
tol = param.tol;
hS = param.hS;
t = param.t;
U = param.U;
k = param.k;
d = param.d;

sketch_distortion = param.sketch_distortion;

% Reset count parameters for each new cycle
mv = 0;
ip = 0;
sv = 0;

if isempty(U)
    SW = [];
    SAW = [];
else
    % In the special case when the matrix does not change, 
    % we can re-use SU from previous problem,
    if param.pert == 0
        SW = param.SU;
        SAW = param.SAU;
        mv = mv + 0;
    else
        SW = param.SU;
        if isempty(U)
            SAW = [];
        else
            SAW = hS(A(U));
            mv = mv + size(U,2);
            sv = sv + size(U,2);
        end
    end
end

% Arnoldi for (A,b)
Sr = hS(r0);
sv = sv + 1;
if param.ssa
    nrm = norm(Sr);
else
    nrm = norm(r0);
    ip = ip + 1;
end
SV(:,1) = Sr/nrm;
V(:,1) = r0/nrm;
stag = 0;

% NOTE: Interestingly, the vectors for which the distortion 
% norm(Sv)/norm(v) is largest away from 1, happen to be the
% residual vectors after each restart (at least with dct).
% What happens is that within a circle, norm(Sr)/norm(r) 
% typically starts to deviate more and more from 1 and 
% as the next cycle is restarted with a residual vector,
% it has large distortion. 
%
%nrmV = norm(V(:,1));
%nrmSV = norm(SV(:,1));

% Initialize QR factorization of SAW (recycling subspace)
if strcmp(param.ls_solve,'mgs') % modified GS
    [Q,R] = qr(SAW,0);
end
if strcmp(param.ls_solve,'hh') % Householder
    [W,R,QtSr] = qrupdate_hh(SAW,[],[],Sr);
end

d_it = 0; sres = []; 

for j = 1:max_it

    w = A(V(:,j));
    mv = mv + 1;

    if param.ssa == 0     % standard t-truncated Arnoldi
        for i = max(j-t+1,1):j
            H(i,j) = V(:,i)'*w;
            ip = ip + 1;
            w = w - V(:,i)*H(i,j);
        end
        H(j+1,j) = norm(w);
        ip = ip + 1;
        V(:,j+1) = w/H(j+1,j);
        SV(:,j+1) = hS(V(:,j+1));
        sv = sv + 1;
        % No need to sketch A*V since S*A*V = (S*V)*H
        %SAV = SV(:,1:j+1)*H(1:j+1,1:j);
        SAV(:,j) = SV(:,1:j+1)*H(1:j+1,j); 
    end

    if param.ssa == 1     % sketched t-truncated Arnoldi
        sw = hS(w); sv = sv + 1;
        SAV(:,j) = sw;

        % quasi-orthogonalise against U
        if size(param.U,2)>0
            coeffs = pinv(param.SU)*sw;
            w = w - param.U*coeffs;
            sw = sw - param.SU*coeffs;
        end
        
        % get coeffs with respect to previous t vectors
        ind = max(j-t+1,1):j;
        coeffs = SV(:,ind)'*sw;

        w = w - V(:,ind)*coeffs;
        %w = w - submatxmat(V,coeffs,min(ind),max(ind)); 

        sw = sw - SV(:,ind)*coeffs;
        nsw = norm(sw);
        SV(:,j+1) = sw/nsw; V(:,j+1) = w/nsw;
        H(ind,j) = coeffs; H(j+1,j) = nsw;
    end


    if param.ssa == 2     % sketch-and-select
        sw = hS(w); sv = sv + 1;
        SAV(:,j) = sw;
        % the following two lines perform the select operation
        coeffs = pinv(SV(:,1:j))*sw;
        [coeffs,ind] = maxk(abs(coeffs),t);
        w = w - V(:,ind)*coeffs;
        sw = sw - SV(:,ind)*coeffs;
        nsw = norm(sw);
        SV(:,j+1) = sw/nsw; V(:,j+1) = w/nsw;
        H(ind,j) = coeffs; H(j+1,j) = nsw;
    end
       
    % Every d iterations, compute the sketched residual 
    % If sres is small enough, compute full residual
    % If this is small enough, break the inner loop
    if true

        d_it = d_it + 1;

        % TODO: Both could be updated column-wise
        SW = [ param.SU, SV(:,1:j) ];
        SAW = [ param.SAU, SAV(:,1:j) ];

        if ~isempty(U)
            %keyboard
        end
    
        % Incrementally extend QR factorization and get LS coeffs
        if strcmp(param.ls_solve,'mgs')
            [Q,R] = qrupdate_gs(SAW,Q,R);
            y = R\(Q'*Sr);
        end
        if strcmp(param.ls_solve,'hh')
            [W,R,QtSr] = qrupdate_hh(SAW,W,R,QtSr);
            y = triu(R)\(QtSr);
        end
        if strcmp(param.ls_solve,'pinv')
            y = pinv(SAW)*Sr;
        end
        if strcmp(param.ls_solve,'\')
            y = SAW\Sr;
        end

        % Compute residual estimate (without forming full approximation)
        sres(d_it) = norm(Sr - SAW*y);
        % only display if the output flag is not used
        if param.verbose >= 1 && (sres(d_it) < tol/4  ...
                || mod(j+1,floor(max_it/5)) == 0 || j == max_it || stag == 1)
            fprintf('gmres-sdr: inner iteration %.2d; ',j+1)
            fprintf('estimated residual %.1d;\n',sres(d_it))
        end
        if (sres(d_it) < tol/4 || stag == 1)
            break
        end
       
        % If the residual estimate is small enough (or we reached the max
        % number of iterations), then we form the full approximation
        % correction (without explicitly forming [U V(:,1:j)])
        if sres(d_it) < tol/sketch_distortion || j == max_it
            if size(U,2) > 0
                e = U*y(1:size(U,2),1) + V(:,1:j)*y(size(U,2)+1:end,1);
            else
                e = V(:,1:j)*y(size(U,2)+1:end,1);
            end

            % Compute true residual
            r = r0 - A(e);
            mv = mv + 1;
            
            nrmr = norm(r);
            ip = ip + 1;

            % potentially increase sketch_distortion
            if nrmr/sres(d_it) > sketch_distortion
                sketch_distortion = nrmr/sres(d_it);
                if param.verbose >= 1
                    % please stop commenting these out! use verbose=0
                    disp(['  sketch distortion increased to ' num2str(sketch_distortion)])
                end
            end
            
            if nrmr < sres(d_it)/2
                stag = 1;
            end

            if sres(d_it) < tol/4 || j == max_it || stag == 1
                % only display if the output flag is not used
                if param.verbose>=1 && stag == 1
                    fprintf('sgmres-sdr: inner iteration %.2d; ', j+1)
                    fprintf('stagnated.\n')
                end
                break
            end
          

        end
    end
end

if size(U,2) > 0
    e = U*y(1:size(U,2),1) + V(:,1:j)*y(size(U,2)+1:end,1);
else
    e = V(:,1:j)*y(size(U,2)+1:end,1);
end

% Compute economic SVD of SW or SAW
if param.harmonic
    [Lfull,Sigfull,Jfull] = svd(SAW,'econ');  % harmonic
else    
    [Lfull,Sigfull,Jfull] = svd(SW,'econ');   % non-harmonic
end

if param.verbose >= 2
    fprintf('  cond(SAU) = %4.1e\n', cond(param.SAU))
    fprintf('  cond(SV) = %4.1e\n', cond(SV(:,1:j)))
    fprintf('  full subspace condition number = %4.1e\n', Sigfull(1,1)/Sigfull(end,end))
end

% Truncate SVD
ell = find(diag(Sigfull) > param.svd_tol*Sigfull(1,1), 1, 'last');
k = min(ell,k);
L = Lfull(:,1:ell);
Sig = Sigfull(1:ell,1:ell);
J = Jfull(:,1:ell);
if param.harmonic
    HH = L'*SW*J;   % harmonic
else
    HH = L'*SAW*J;  % non-harmonic
end

% update augmentation space using QZ
if isreal(HH) && isreal(Sig)
    [AA, BB, Q, Z] = qz(HH,Sig,'real'); % Q*A*Z = AA, Q*B*Z = BB
else
    [AA, BB, Q, Z] = qz(HH,Sig);
end
ritz = ordeig(AA,BB);
if param.harmonic
    [~,ind] = sort(abs(ritz),'descend');  % harmonic
else
    [~,ind] = sort(abs(ritz),'ascend');   % non-harmonic
end

select = false(length(ritz),1);
select(ind(1:k)) = 1;
[AA,BB,~,Z] = ordqz(AA,BB,Q,Z,select);
if k>0 && k<size(AA,1) && (AA(k+1,k)~=0 || BB(k+1,k)~=0)  % don't tear apart 2x2 diagonal blocks
    keep = k+1;
else
    keep = k;
end

if param.verbose >= 2
    disp(['  recycling subspace dimension k = ' num2str(keep)])
end

% cheap update of recycling subspace without explicitly constructing [U V(:,1:j)]
JZ = J*Z(:,1:keep);
if size(U,2) > 0
    out.U = U*JZ(1:size(U,2),:) + V(:,1:j)*JZ(size(U,2)+1:end,:);
else
    out.U = V(:,1:j)*JZ(size(U,2)+1:end,:);
end

out.SU = SW*JZ;
out.SAU = SAW*JZ;
out.hS = hS;
out.k = keep;
out.m = j;
out.sres = sres;
out.sketch_distortion = sketch_distortion;
out.stag = stag;


end

function [xout, output] = sgmres_standard_cycle(A, r0, params)
% Clean, clear, fast standard_cycle implementation consistent with randgmres structure.
% Using the corrected sres update:
% y = H(1:j,1:j) \ g(1:j);
% sres(j+1) = norm(P(:,1) - P*[0; y; zeros(m-j-1,1)]);
% yout = [y; zeros(m-j,1)]; dx = Q * yout;

m = params.max_it;
tol = params.tol;
Theta = params.hS;
lssolver = params.lssolver;
lowprecision = params.lowprecision;

n = length(r0);
k = params.s;

% Initialize
if lowprecision
    Q = zeros(n, m, 'single');
else
    Q = zeros(n, m, 'double');
end
S = zeros(k, m);
P = zeros(k, m);

cs = zeros(m, 1);
sn = zeros(m, 1);
g = zeros(m+1, 1);

% Initial sketch
s = Theta(r0);
p = s;
P(:, 1) = p;
rnorm = norm(s);
s = s / rnorm;
q = r0 / rnorm;
g(1) = rnorm;

sres = zeros(1, m+1);
sres(1) = rnorm;
Hres = zeros(1,m+1);
Hres(1) = rnorm;

stability = zeros(1,m+1);
stability(1) = s'*s - 1;

stag = 0;

for j = 1:m
    Q(:, j) = q;
    S(:, j) = s;
    
    if lowprecision
        w = A(double(q));
    else
        w = A(q);
    end
    
    p = Theta(w);
    P(:, j+1) = p;
    
    r = leastsquares(S, p, lssolver, j);
    r(j+1:m) = 0;
    
    if lowprecision
        q = single(w) - Q * single(r);
        s = Theta(double(q));
    else
        q = w - Q * r;
        s = Theta(q);
    end
    
    r(j+1) = norm(s);
    q = q / r(j+1);
    s = s / r(j+1);
    
    H(1:j+1, j) = r(1:j+1);
    
    % Apply previous Givens rotations
    for i = 1:j-1
        temp = cs(i) * H(i, j) + sn(i) * H(i+1, j);
        H(i+1, j) = -sn(i) * H(i, j) + cs(i) * H(i+1, j);
        H(i, j) = temp;
    end
    
    % Compute and apply new Givens rotation
    [cs(j), sn(j)] = givens(H(j, j), H(j+1, j));
    
    temp = cs(j) * g(j);
    g(j+1) = -sn(j) * g(j);
    g(j) = temp;
    
    H(j, j) = cs(j) * H(j, j) + sn(j) * H(j+1, j);
    H(j+1, j) = 0;
    
    % Compute y, sres, dx consistently with randgmres
    Hres(j+1) = abs(g(j+1));
            
    % Measure orthogonality of the Krylov basis.
    stability(j+1) = sqrt(stability(j)^2 + (s'*s-1)^2 + 2*norm(s'*S)^2);

    y = H(1:j, 1:j) \ g(1:j);
    sres(j+1) = norm(P(:, 1) - P * [0; y; zeros(m - j - 1, 1)]);
    
    % Check for error stagnation.
    if Hres(j+1) < sres(j+1)/2
        stag = 1;
    end
    
    % only display if the output flag is not used
    if params.verbose >= 1 && (sres(j+1) < tol/4  ...
            || mod(j+1,floor(m/5)) == 0 || j == m ...
            || stability(j+1) > 1 || stag == 1)
        fprintf('sgmres-standard: inner iteration %.2d; ',j+1)
        fprintf('estimated residual %.1d; ',sres(j+1))
        fprintf('stability measure %.1d;\n',stability(j+1))
    end
    
    if sres(j+1) < tol/4 || j == m ...
            || stability(j+1) > 1 || stag == 1
        % only display if the output flag is not used
        if params.verbose>=1 && stag == 1
            fprintf('sgmres-standard: inner iteration %.2d; ',j+1)
            fprintf('stagnated.\n')
        end
        if params.verbose>=1 && stability(j+1) > 1
            fprintf('sgmres-standard: inner iteration %.2d; ',j+1)
            fprintf('the orthogonality of Krylov basis was lost.\n')
        end
        break
    end
end

xout = Q(:, 1:j) * y;
output.sres = sres(1:j+1);
output.m = j;
output.stag = stag;

end


%% Solving nearly orthogonal least-squares problem
function r = leastsquares(S,p,lssolver,initer)
    if strcmp(lssolver,'3reorth')
        j = 3;
    elseif strcmp(lssolver,'5reorth')
        j = 5;
    elseif strcmp(lssolver,'20reorth')
        j = 20;
    else
        j = 0;
    end
    if j ~= 0
        % Richardson iterations
        r = (p'*S)';
        p = p - S*r;
        for i=1:j-1
            dr = (p'*S)';
            p = p - S*dr;
            r = r + dr;
        end
    else
        % Conjugate Gradient
        Stemp = S(:,1:initer);
        [rtemp,~,~,~,~] = pcg(@(x) ((Stemp*x)'*Stemp)',Stemp'*p,1.0e-14,20);
        r = [rtemp; zeros(size(S,2) - initer,1)];
    end
end

function [cs, sn] = givens(a, b)
% Standard stable Givens rotation
if b == 0
    cs = 1; sn = 0;
else
    if abs(b) > abs(a)
        t = a / b;
        sn = 1 / sqrt(1 + t^2);
        cs = t * sn;
    else
        t = b / a;
        cs = 1 / sqrt(1 + t^2);
        sn = t * cs;
    end
end
end



