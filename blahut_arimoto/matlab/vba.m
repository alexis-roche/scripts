function [p, px, py, r, MI] = bami( lda, nbins, niter ) 

% Scheme: 
%   Step 1: p(x,y) = [1/z(lda)] * exp( -lda*d(x,y) ) * pi(x) pi(y)
%   Step 2: pi(x) = int p(x,y) dy, same for pi(y)
%
% Here: d(x,y) = [x-y]^2

% Params
if nargin < 3, niter = 5; end, 
if length(lda)~=5, error('lda must be of length 5'); end,

% Pre-computations
offset = (nbins-1)/2;
Dx = repmat([0:nbins-1]',1,nbins) - offset; 
Dxx = Dx.^2; 
Dy = repmat([0:nbins-1],nbins,1) - offset;
Dyy = Dy.^2;
Dxy = Dx .* Dy; 


% Init with uniform marginals
px = ones(nbins, 1)/nbins; 
py = ones(1, nbins)/nbins; 
%%px = rand(nbins, 1); px = px/sum(px); 
%%py = rand(1, nbins); py = py/sum(py); 

%% Loop start 
for i=1:niter,

	% Compute the reference distribution 
	m = repmat(px, 1, nbins) .* repmat(py, nbins, 1); 
	
	% Step 1 (max ent)
	aux = lda(1)*Dx + lda(2)*Dy + lda(3)*Dxx + lda(4)*Dyy + lda(5)*Dxy; 
	p = m .* exp(-aux); 
	p = p / sum(p(:)); %% Normalization equivalent to computing z(lda)

	% Compute mutual information and average distance
	r = max(eps, p./max(eps, m)); 
	aux = p .* log( r );
	MI(i) = sum(aux(:));  

	% Step 2 (marginalization) 
	px = sum( p, 2 ); 
	py = sum( p, 1 ); 
	
	% Display 
	figure(1), 
	hisplay( p ); 
	pause(.01); 
	figure(2), 
	plot([1:i], MI)
	
end, 
