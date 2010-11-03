function [p, px, py, da, lda, MI] = bami( d, nbins, niter ) 

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
%%px = rand(nbins, 1); px = px/sum(px); 
%%py = rand(1, nbins); py = py/sum(py); 

%% Loop start 
lda = 10; 
for i=1:niter,

	% Step 1
	err = 1; 
	nsubiter = 0; 
	step = 1e-4; 
	while err > 0.01 & nsubiter < 1000, 
	
		err0 = err; 
		
		p = exp(-lda*D) .* repmat(px, 1, nbins); 
		p = p .* repmat(py, nbins, 1); 
		p = p / sum(p(:)); %% Normalization equivalent to computing z(lda)

		q = p .* D; 
		da = sum(q(:));
		
		err = d-da;
		lda = min( max( eps, lda - err * step), 1e2 ); %% Unlikely to work very well...
		err = abs(err); 
		
		if  err > err0, 
			step = step/2;
		else, 
			step = step*2; 
		end,
		
		nsubiter = nsubiter + 1; 

	end, 
	error = err;
	L(i) = lda;
	
	% Step 2 (marginalization) 
	px = sum( p, 2 ); 
	py = sum( p, 1 ); 
	
	% Compute mutual information and average distance
	q = repmat(px, 1, nbins) .* repmat(py, nbins, 1); 
	q = p .* log( max(eps, p) ./ max(eps,q) );
	MI(i) = sum(q(:));  
	
	% Display 
	figure(1), 
	hisplay( p ); 
	%%pause(.01); 
	figure(2), 
	plot([1:i], MI)
	figure(3), 
	plot([1:i], L)

end, 
