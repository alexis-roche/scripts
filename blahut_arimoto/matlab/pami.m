function [p, px, py, r, d, MI] = bami( lda, nbins, niter ) 

% Scheme: 
%   Step 1: p(x,y) = [1/z(lda)] * exp( -lda*d(x,y) ) * pi(x) pi(y)
%   Step 2: pi(x) = int p(x,y) dy, same for pi(y)
%
% Here: d(x,y) = [x-y]^2

% Params
if nargin < 3, niter = 5; end, 

% Pre-compute the distance matrix
D = repmat([0:nbins-1]',1,nbins) - repmat([0:nbins-1],nbins,1); 
D = D.^2; 

% Init with uniform marginals
px = ones(nbins, 1)/nbins; 
py = ones(1, nbins)/nbins; 

%% Loop start 
for i=1:niter,

	% Compute the reference distribution 
	m = repmat(px, 1, nbins) .* repmat(py, nbins, 1); 
	
	% Step 1 (max ent)
	p = m .* exp(-lda*D); 
	p = p / sum(p(:)); %% Normalization equivalent to computing z(lda)

	% Compute mutual information and average distance
	r = max(eps, p./max(eps, m)); 
	aux = p .* log( r );
	MI(i) = sum(aux(:));  

	% Step 2 (marginalization) 
	px = sum( p, 2 ); 
	py = sum( p, 1 ); 
	
	% Compute the average distance
	d(i) = sum( p(:).*D(:) ); 
	
	% Display 
	figure(1), 
	hisplay( p ); 
	pause(.01); 
	figure(2), 
	plot([1:i], d)
	figure(3), 
	plot([1:i], MI)
	
end, 
