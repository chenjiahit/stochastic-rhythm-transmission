clear

%% Parameters: edit this block
p.N = 8;
p.Tu = 5;
p.rhoBar = 10;
p.B = 3;
p.epsilon = 1;
p.phi = 0;
gammaGrid = linspace(0,2,501);
% gamma=0 is plotted only as the limiting value eta -> 0; stationary
% downstream abundance requires gamma > 0.

outDir = fullfile(fileparts(mfilename('fullpath')),'results');
if ~exist(outDir,'dir'), mkdir(outDir); end
k = p.N/p.Tu;
% RList = [4*k./(rhoList*p.epsilon^2),0];
RList = 4*k./(p.rhoBar*p.epsilon^2);
nCurves = numel(RList);
eta = zeros(numel(gammaGrid),nCurves);
omegaPeak = eta;
gc = zeros(1,nCurves);
labels = cell(1,nCurves);
[~,~,gc0] = ring_threshold(p,p.rhoBar);
for j = 1:nCurves
    [e,w,etaU] = exact_efficiency(p,gammaGrid,RList(j));
    eta(:,j) = e(:); omegaPeak(:,j) = w(:);
    gc(j) = gc0*sqrt(1+RList(j));
    if j<=numel(p.rhoBar)
        labels{j} = sprintf('rhoBar = %g, R = %.3g',p.rhoBar(j),RList(j));
    else
        labels{j} = 'Zero readout noise (R = 0 limit)';
    end
end

% fig = figure('Color','w','Position',[90,90,1120,620]);
colors = lines(nCurves);
for j = 1:nCurves
    plot(gammaGrid,eta(:,j),'LineWidth',1.8,'Color',colors(j,:)); hold on;
end
% xlabel('Degradation rate gamma');
% ylabel('Oscillation efficiency eta');
% title(sprintf('Finite-N efficiency: N=%d, T_u=%.1f, epsilon=%.2g', ...
%       p.N,p.Tu,p.epsilon));
% legend(labels,'Location','southeast');
% xlim([0,max(gammaGrid)]); ylim([0,min(1,etaU+0.05)]);
grid off; box off; set(gca,'FontSize',12);

function [gc,R,gc0] = ring_threshold(p,rhoBar)
% Exact finite-N threshold, Eqs. (24)-(25), NOT the large-N approximation.
k = p.N/p.Tu;
R = 4*k./(rhoBar*p.epsilon^2);
if p.N <= 3
    gc0 = Inf;
elseif p.N == 4
    gc0 = sqrt(2)*k;
else
    a = pi/p.N;
    % 1-cos(a) = 2*sin(a/2)^2 avoids cancellation for large N.
    K = 16*k^2*sin(a)^2*cos(a)*(2*sin(a/2)^2);
    gc0 = sqrt(K);
end
gc = gc0*sqrt(1+R);
end

function [eta,omegaPeak,etaU] = exact_efficiency(p,gammas,R)
% Global maximization of the rational spectrum: enumerate ALL stationary
% points, plus omega = 0. No frequency grid or unimodality assumption.
% R can be zero to represent the zero-readout-noise limit.
k = p.N/p.Tu;
theta = 2*pi/p.N;
c = 4*k^2*sin(theta/2)^2; % mu^2 + nu^2
beta = -cos(theta);
xu = max(0,sqrt(2-2*beta)-1);
qu = (1+xu)/(xu^2+2*beta*xu+1);
etaU = max(0,1-1/qu);
eta = zeros(size(gammas));
omegaPeak = zeros(size(gammas));
for j = 1:numel(gammas)
    gamma = gammas(j);
    if gamma == 0
        % Limit gamma -> 0 at fixed burst parameters; no stationary X at 0.
        continue
    end
    a = gamma^2/c;
    % x = omega^2/c; G(omega)/G(0) = a*n(x)/((1+R)*d(x)).
    n = [R, 1+2*R*beta, 1+R];
    d = conv([1,a],[1,2*beta,1]);
    left = conv(polyder(n),d);
    right = conv(n,polyder(d));
    m = max(numel(left),numel(right));
    coeff = [zeros(1,m-numel(left)),left] ...
           -[zeros(1,m-numel(right)),right];
    r = roots(coeff/max(abs(coeff)));
    realRoot = abs(imag(r)) <= 1e-8*(1+abs(real(r)));
    x = [0; real(r(realRoot & real(r)>0))];
    q = (1+x)./(x.^2+2*beta*x+1);
    H = a./(a+x).*(q+R)/(1+R);
    [peak,which] = max(H);
    eta(j) = max(0,1-1/peak);
    omegaPeak(j) = sqrt(c*x(which));
end
end
