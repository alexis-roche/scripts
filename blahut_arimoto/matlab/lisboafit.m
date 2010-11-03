function [P, L, Pi, Pj] = lisboafit( H, Hi, Hj, niter, tiny, init)

if nargin < 3, error('Three arguments needed.'), end,
if nargin < 4, niter = 10; end, 
if nargin < 5, tiny = eps; end, 
if nargin < 6, init = 0; end, 

if size(Hi,1)==1, Hi = Hi'; end, 
if size(Hj,2)==1, Hj = Hj'; end, 

% Bounds
Im = length(Hi); 
Jm = length(Hj); 

% Constants
na = sum(H(:));
Ki = Hi - sum(H,2); 
Kj = Hj - sum(H,1); 
nb = sum(Ki);
nc = sum(Kj); 
lda = na + nb + nc; 

% Initialization 
if init == 0, 
	P = (1/(Im*Jm))*ones(Im, Jm); 
elseif init == 1, 
	P = rand(Im,Jm); P = P/sum(P(:)); 
elseif init == 2, 
	P = H + eps; P = P/sum(P(:)); 
else, 
	P = Hi*Hj + eps; P = P/sum(P(:));
end,
L = zeros(niter,1); 

% Loop 
iter = 0; 
while iter<niter, 
	
	% Display
	hisplay(P),
	pause, 	

	% E-step 
	Pi = sum(P,2); 
	aux = Ki ./ max(Pi,tiny); 
	%%%aux = Ki ./ Pi;
	Hn = H + P.*repmat(aux, 1, Jm); 
	
	%% DEBUG TEST
	test1 = max( abs( sum(Hn,2)-Hi ) ),
	
	% M-step 
	aux = sum(Hn,1); 
	P = Hn .* repmat((aux+Kj)./max(aux,tiny), Im, 1); 
	%%%P = Hn .* repmat((aux+Kj)./aux, Im, 1);
	
	%% DEBUG TEST
	test2 = max( abs( sum(P,1)-aux-Kj ) ),

	% Normalization to unit integral
	P = P/lda;	

	% Iteration increment
	iter = iter + 1;
	
	% Log likelihood
	Pi = sum(P,2); 
	Pj = sum(P,1); 
	L(iter) = sum(sum( H.*log(max(P,tiny)) )) + ... 
		sum( Ki.*log(max(Pi,tiny)) ) + sum( Kj.*log(max(Pj,tiny)) ); 
		
end,

Pi = sum(P,2); 
Pj = sum(P,1); 

%% DISPLAYS
a = 1; 

figure, 
marginal_plot( Pi, Hi, sum(H,2), a ); 

figure, 
marginal_plot( Pj, Hj, sum(H,1), a ); 

function marginal_plot( P, P0, Pa, a )
plot( (P/sum(P)).^a, 'o-'), 
hold, 
plot( (P0/sum(P0)).^a, 'r:'),
plot( (Pa/sum(Pa)).^a, 'go'), 

