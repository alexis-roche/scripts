function hisplay(H,a);

% SYNTAX: hisplay(H);
%
% Displays a joint histogram as a 2D color image. 

if nargin < 2, a = 0.25; end,

Y = H / max(max(H));

%%Y = round(255*(Y.^a));

Y = abs(255*(Y.^a));

image(Y);
set(gca,'YDir','normal');


xlabel('Target'), 
ylabel('Source')
