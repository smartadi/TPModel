%% =========================================================================
%  TP Model Transformation Framework — MATLAB Code
%  Source: Baranyi, P. (2019)
%  "Extracting LPV and qLPV structures from state-space functions:
%   a TP model transformation based framework"
%  IEEE Transactions on Fuzzy Systems, DOI: 10.1109/TFUZZ.2019.2908770
%
%  REQUIREMENTS:
%    TPtool MATLAB Toolbox (download via Wikipedia: "TP model based control")
%    Functions used: hosvd(), genhull(), tprods(), ndim_unfold()
%
%  CONTENTS:
%    1. Inverted Pendulum Model
%    2. Helper: get_grid_point (Getgrid)
%    3. Example 1 — Global & Local Linearisation       (Method 2)
%    4. Example 2 — Extended TP / qLPV Transformation  (Method 4)
%    5. Example 3 — TP Transform of Known S(p1,p2)     (Method 1)
%    6. Example 4 — Modified Parameter Space           (Section VI)
%% =========================================================================


%% =========================================================================
%  SECTION 1 — INVERTED PENDULUM MODEL  (Equations 58-64)
%% =========================================================================

function DX = model(X)
% Full nonlinear state-space model of the inverted pendulum.
% X = [theta, dtheta, s, ds, u]   (state + input concatenated)
% Returns DX = [dtheta, ddtheta, ds, dds]

    th  = X(1);  dth = X(2);
    x   = X(3);  dx  = X(4);
    u   = X(5);

    si = sin(th);  co = cos(th);

    Mc = 3;   mp = 0.2;  l = 0.31;  b = 0.1;  g = 9.88;

    s1l = l * (Mc + mp*si*si);
    s2l = Mc + mp*si*si;

    T1 =  (Mc + mp)*g*si / s1l;
    T2 =  b*co*dx         / s1l;
    T3 = -mp*l*si*co*dth*dth / s1l;
    T4 = -co*u             / s1l;

    P1 = -mp*g*si*co / s2l;
    P2 = -b*dx        / s2l;
    P3 =  mp*l*si*dth*dth / s2l;
    P4 =  u            / s2l;

    DX = [dth,  T1+T2+T3+T4,  dx,  P1+P2+P3+P4]';
end


function S = modelS(P)
% Parameter-dependent system matrix S(p1, p2).
% P = [theta, dtheta]   — scheduling parameters
% Returns S ∈ R^{4×5}  such that  xdot ≈ S * [x; u]

    dth = P(2);
    si  = sin(P(1));
    co  = cos(P(1));

    Mc = 3;  mp = 0.2;  l = 0.31;  b = 0.1;  g = 9.88;

    s1l = l*(Mc + mp*si*si);
    s2l = Mc + mp*si*si;

    T1 =  (Mc + mp)*g*si / s1l;
    T2 =  b*co            / s1l;
    T3 = -mp*l*si*co      / s1l;
    T4 = -co               / s1l;

    P1v = -mp*g*si*co / s2l;
    P2v = -b           / s2l;
    P3v =  mp*l*si*dth*dth / s2l;
    P4v =  1.0         / s2l;

    S = [0,             1,  0,   0,   0;
         0,  T1/dth + T3,  0,  T2,  T4;
         0,             0,  0,   1,   0;
         0,  P1v/dth+P3v, 0, P2v, P4v];
end


%% =========================================================================
%  SECTION 2 — HELPER: GETGRID  (Paper Section VIII)
%% =========================================================================

function g = Getgrid(m, M, Omega, O)
% Compute one grid point given multi-index m (1-based vector),
% grid size M (scalar), half-intervals Omega, and centre O.
%
% Formula (per dimension i):
%   g(i) = O(i) - Omega(i) + (m(i)-1) * 2*Omega(i) / (M-1)

    a = length(m);
    g = zeros(1, a);
    for i = 1:a
        step = (m(i) - 1) * 2 * Omega(i) / (M - 1);
        g(i) = O(i) - Omega(i) + step;
    end
end


%% =========================================================================
%  SECTION 3 — LINEARISATION FUNCTION  (Method 2 / Paper Section IV)
%% =========================================================================

function S = linear(Omega, M, O)
% Numerically reconstruct global linear mapping  xdot ≈ S * [x; u]
% using pseudo-inverse over a sampling grid.
%
% Parameters:
%   Omega — half-intervals for each input dimension (5-element vector)
%   M     — grid density per dimension
%   O     — centre of the grid (5-element vector)
%
% Returns:
%   S — system matrix R^{4×5}  (least-squares best fit)

    for m1 = 1:M
      for m2 = 1:M
        for m3 = 1:M
          for m4 = 1:M
            for m5 = 1:M
                m = [m1 m2 m3 m4 m5];
                g = (Getgrid(m, M, Omega, O))';
                G(m1,m2,m3,m4,m5,:)   = g;
                DXT(m1,m2,m3,m4,m5,:) = model(g);
            end
          end
        end
      end
    end

    Xu  = ndim_unfold(G,   6);   % unfold along last (input) dimension
    DXu = ndim_unfold(DXT, 6);   % unfold along last (output) dimension

    S = DXu * pinv(Xu);          % Equation 16: S = F^[I+1] * pinv(G^[I+1])
end


%% =========================================================================
%  SECTION 4 — EXAMPLE 1: GLOBAL & LOCAL LINEARISATION
%% =========================================================================

%% -- Global linearisation (large Omega) --
clear;
Omega = [1.6, 1.6, 1, 1, 1];
M = 5;
O = [0, 0, 0, 0, 0];    % centre of grid

S_global = linear(Omega, M, O);
disp('Global linearisation matrix S:');
disp(S_global);

%% -- Local linearisation at equilibrium (theta ≈ 0, dtheta ≈ 0) --
%  Shrink Omega to near-zero for theta and dtheta
Omega_local = [1e-10, 1e-10, 1, 1, 1];

S_local = linear(Omega_local, M, O);
disp('Local linearisation at equilibrium (Eq. 65):');
disp(S_local);
% Expected result (from paper):
%  [  0       1      0      0       0    ]
%  [ 33.75    0      0      0.11  -1.08  ]
%  [  0       0      0      1       0    ]
%  [ -0.65    0      0     -0.03    0.33 ]


%% =========================================================================
%  SECTION 5 — EXAMPLE 2: EXTENDED TP / qLPV TRANSFORMATION  (Method 4)
%% =========================================================================
%
%  Goal: extract  xdot = S(p) * [x; u]  without knowing S(p) analytically.
%  All 5 variables [x1..x4, u] act as scheduling parameters.
%
%  Key modification to LINEAR: fix linearisation centre at grid point O,
%  using a tiny LOmega so each local linearisation is essentially a Jacobian.

clear;
D      = 3;                          % local linearisation grid density
r      = 1e-10;                      % tiny half-interval (Jacobian approx.)
M      = 10;                         % parameter grid density
LOmega = [r, r, r, r, r];           % local linearisation half-intervals
Omega  = [1.6, 1.6, 1, 1, 1];      % parameter space half-intervals
O      = [0, 0, 0, 0, 0];           % centre

for m1 = 1:M
  for m2 = 1:M
    for m3 = 1:M
      for m4 = 1:M
        for m5 = 1:M
            m = [m1 m2 m3 m4 m5];
            g = (Getgrid(m, M, Omega, O))';

            % Linearise at this grid point (centre = g, tiny subspace)
            S = linear(LOmega, D, g);

            ST(m1,m2,m3,m4,m5,:,:) = S;
        end
      end
    end
  end
end

% Extract SNNN convex hull TP model via HOSVD
[S_core, U, sv] = hosvd(ST, [1, 1, 1, 1, 1, 0, 0], 1e-11);

for i = 1:5
    Uc{i}  = genhull(U{i}, 'snnn');
    figure(i);
    plot(Uc{i}, 'LineWidth', 3);
    title(['Weighting functions — p', num2str(i)]);
    xlabel('Parameter value');
    ylabel('w(p)');
    Ucp{i} = pinv(Uc{i});
end

Sc = tprods(S_core, Ucp);

% Result: f(x,u) ≈ Sc ×₁w₁(x1) ×₂w₂(x2) ×₃w₃(x3) ×₄w₄(x4) ×₅w₅(u) * [x;u]
disp('Example 2 complete. Core tensor size:');
disp(size(Sc));


%% =========================================================================
%  SECTION 6 — EXAMPLE 3: TP TRANSFORM OF KNOWN S(p1,p2)  (Method 1)
%% =========================================================================
%
%  Scheduling parameters: p1 = x1 = theta,  p2 = x2 = dtheta
%  S(p) is directly sampled from the closed-form modelS() function.

clear;
M     = 50;                     % grid density (even to avoid sin(0)/0)
Omega = [1.6, 1.6];
O     = [0, 0];

% Sample S(p) over 2D parameter grid
for m1 = 1:M
  for m2 = 1:M
      m = [m1, m2];
      g = (Getgrid(m, M, Omega, O))';
      ST(m1, m2, :, :) = modelS(g);
  end
end

% HOSVD: decompose over p1 and p2 dimensions only (not S matrix dims)
[S_core, U, sv] = hosvd(ST, [1, 1, 0, 0], 1e-11);

for i = 1:2
    Uc{i}  = genhull(U{i}, 'snnn');
    figure(i);
    plot(Uc{i}, 'LineWidth', 3);
    title(['Weighting functions — p', num2str(i)]);
    xlabel('Parameter value');
    ylabel('w(p)');
    Ucp{i} = pinv(Uc{i});
end

Sc = tprods(S_core, Ucp);

% Result: xdot = Sc ×₁w₁(x1) ×₂w₂(x2) * [x; u]
% Number of fuzzy rules = size(Uc{1},2) * size(Uc{2},2)  → paper: 3×5 = 15
fprintf('Rules p1: %d,  p2: %d,  Total: %d\n', ...
    size(Uc{1},2), size(Uc{2},2), size(Uc{1},2)*size(Uc{2},2));


%% =========================================================================
%  SECTION 7 — EXAMPLE 4: MODIFIED PARAMETER SPACE  (Section VI)
%% =========================================================================
%
%  New parameters: p1 = sin(x1),  p2 = cos(x1),  p3 = x2
%  Demonstrates flexible reparametrisation before TP transformation.

function S = modelS_new(P)
% System matrix with modified parameters.
% P = [sin(theta), cos(theta), dtheta]

    dth = P(3);
    si  = P(1);   % sin(theta)  — passed directly
    co  = P(2);   % cos(theta)  — passed directly

    Mc = 3;  mp = 0.2;  l = 0.31;  b = 0.1;  g = 9.88;

    s1l = l*(Mc + mp*si*si);
    s2l = Mc + mp*si*si;

    T1 =  (Mc + mp)*g*si / s1l;
    T2 =  b*co            / s1l;
    T3 = -mp*l*si*co      / s1l;
    T4 = -co               / s1l;

    P1v = -mp*g*si*co      / s2l;
    P2v = -b                / s2l;
    P3v =  mp*l*si*dth*dth / s2l;
    P4v =  1.0              / s2l;

    S = [0,             1,  0,    0,   0;
         0,  T1/dth + T3,  0,   T2,  T4;
         0,             0,  0,    1,   0;
         0, P1v/dth+P3v,  0,  P2v, P4v];
end

clear;
M     = 50;
Omega = [1.6, 1.6, 0.6];    % half-intervals for [sin(x1), cos(x1), x2]
O     = [0, 0, 0];

% Sample S over 3D parameter grid
for m1 = 1:M
  for m2 = 1:M
    for m3 = 1:M
        m = [m1, m2, m3];
        g = (Getgrid(m, M, Omega, O))';
        ST(m1, m2, m3, :, :) = modelS_new(g);
    end
  end
end

% HOSVD with CNO hull type (paper uses CNO for Example 4)
[S_core, U, sv] = hosvd(ST, [1, 1, 1, 0, 0], 1e-11);

for i = 1:3
    Uc{i}  = genhull(U{i}, 'cno');
    figure(i);
    plot(Uc{i}, 'LineWidth', 3);
    titles = {'p_1 = sin(x_1)', 'p_2 = cos(x_1)', 'p_3 = x_2'};
    title(titles{i});
    xlabel('Parameter value');
    ylabel('w(p)');
    Ucp{i} = pinv(Uc{i});
end

Sc = tprods(S_core, Ucp);

% Result: xdot = Sc ×₁w₁(sin x1) ×₂w₂(cos x1) ×₃w₃(x2) * [x; u]
% Paper reports 18 rules  (3×2×3)
fprintf('Rules p1: %d,  p2: %d,  p3: %d,  Total: %d\n', ...
    size(Uc{1},2), size(Uc{2},2), size(Uc{3},2), ...
    size(Uc{1},2)*size(Uc{2},2)*size(Uc{3},2));
