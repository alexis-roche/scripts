function [p, px, p1, px1] = baa( py, lda, niter ) 

% Tentative Blahut-Arimoto algorithm 
%
% Scheme: 
%   Step 1: p(x|y) = [1/z(lda,y)] * exp( -lda*t(x,y) ) * pi(x)
%   Step 2: pi(x) = int py(y) p(x|y) dy
%
% Here: t(x,y) = [x-y]^2

% Params
if nargin < 3, niter = 5; end, 
nbins = length(py); 
if size(py,1) > 1, py = py'; end, %% Force row vector 

T = repmat([0:nbins-1]',1,nbins) - repmat([0:nbins-1],nbins,1); 
T = T.^2; 

% Init
px = ones(nbins, 1)/nbins; %% Uniform marginal to start
%%px = rand(nbins,1); px = px/sum(px); 
px1 = px; 

%% Loop start 
for i=1:niter,

	% Step 1
	p = exp(-lda*T) .* repmat(px, 1, nbins); 
	p = p ./ repmat(sum(p), nbins, 1);
	if i==1, 
		p1 = p; 
	end,
 
	% Step 2 (marginalization) 
	px = sum( p .* repmat(py, nbins, 1), 2 ); 

	% Display 
	figure(1), 
	hisplay( p ); 
	pause(.01);

end, 

% Display
%%im = round( length(py)*rand ); 
%%clf, 
%%plot( p1(:,im) ); 
%%hold,
%%plot( p(:,im), 'r' ); 
%%plot( px1, ':' ); 
%%plot( px, 'r:' ); 




