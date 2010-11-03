function nplot(x, y)

xn = x/sum(x); 

if nargin == 1, 
	plot(xn), 
else, 
	plot(xn, y),
end,
