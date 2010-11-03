function [m, ok, k, kk] = conset( x, a, m0 )

% Minimize C(m) = sum_i [x-m]^2 
% Confidence set of the form I = {m, C(m)<=k}

m = mean(x); 

n = length(x); 
z = (x-m).^2; 
M = mean(z);
S = sqrt( (mean(z.^2) - M^2) / n );
%%S = std(z) / sqrt(n);

C0 = mean((x-m0).^2); 
ok = normcdf( C0, M, S ) <= a; 
k = norminv(a, M, S); 

%% DEBUG 
%%ok = chi2cdf( n*C0, n ) <= a; 
kk = chi2inv(a, n)/n; 
%%ok = C0 <= kk; 

% Display 
%%b = std(x)/sqrt(n); 
%%mm = [m-10*b:.1*b:m+10*b];
mm = [-2:.1:2]; 
C = [];  
for i=1:length(mm), 
	C(i) = mean((x-mm(i)).^2); 
end, 
clf, 
plot(mm, C), 
hold, 
plot(mm, ones(length(mm))*k, 'r'),
plot(mm, ones(length(mm))*kk, 'g'),


