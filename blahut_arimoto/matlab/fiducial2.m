% Gaussian variate
n = 100;
x = normrnd(0,1,n,1); 

% Pivotal quantity: 
% t(x,m) = sum_i (xi-m)^2 ~ chi2(n)
% Fiducial sample 
N = 1e5; 
t = chi2rnd(n,N,1);

% Solve for m
% We have: (m-mx)^2 + vx = t/n => m = mx +/- sqrt(t/n-vx)
mx = mean(x); 
vx = mean((x-mx).^2); 
t = t/n - vx; 
%%%t = t(find(t>=0)); %% Discard negative values
t = max(t, 0); 
Np = length(t); 
M = mx + sign(rand(Np,1)-.5).*sqrt( t ); 

%% Fiducial estimates
mu = mean(M), 
vmu = mean((M-mu).^2), 

mx, 
1/n, 

toto = vmu + vx, %% SHould equate 1 

% Display 
clf,
hist(M, 101); 
