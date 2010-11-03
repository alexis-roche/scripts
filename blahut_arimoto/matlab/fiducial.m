

% Gaussian variate N(0,1)
n = 1e6; 
x = normrnd(0,1,n,1); 

% Sampled parameter space
s = 1/sqrt(n); 
m = [-5:.01:5]*s; 
P = zeros(size(m)); 
T = zeros(size(m)); 

% Test stat: t(x,m) = [mean(x) - m]^2
% Critical region: t >= t_obs
% t(x,m)= chi^2/n
for i = 1:length(m), 
	
	T(i) = (mean(x) - m(i))^2;
	P(i) = 1 - chi2cdf( n*T(i), 1 ); 
	
end,

% Display 
clf,
plot( m, P ); 
hold,
plot( m, T/max(T), 'g' ); 
plot( m, exp(- n * (m-mean(x)).^2 ) , 'r'); 
