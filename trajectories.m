clear

%% Parameters: edit this block
p.N = 8;
p.Tu = 5;
p.rhoBar = 10;
p.B = 3;
p.epsilon = 1;
p.phi = 0;
gammas = [0.1,1.5];
T = 120;
dt = 0.05;
% burn = max(20*p.Tu,12/min(gammas));
burn = 0;

outDir = fullfile(fileparts(mfilename('fullpath')),'results');
if ~exist(outDir,'dir'), mkdir(outDir); end
[gc,R] = ring_threshold(p,p.rhoBar);
[t,U,X,S,stats] = simulate_ring_bursts(p,gammas,T,dt,burn);
meanU = p.rhoBar*p.B;
meanX = meanU./gammas;

%% Plot normalized traces: every theoretical mean is one
colors = lines(numel(gammas));
fig = figure('Color','w','Position',[80,60,1150,600]);
ax = gobjects(1,numel(gammas)+1);
ax(1) = subplot(numel(gammas)+1,1,1);
% stairs(t,U/meanU,'Color',[0.12,0.12,0.12],'LineWidth',1.1);
stairs(t,U,'Color',[0.12,0.12,0.12],'LineWidth',1.1); hold on;
% yline(1,':','Color',[.6,.6,.6]); ylabel('U');
% title(sprintf('Uniform ring: N=%d, T_u=%.1f, R=%.3g, gamma_c=%.3g', ...
%       p.N,p.Tu,R,gc));
grid off; box off;
for j = 1:numel(gammas)
    ax(j+1) = subplot(numel(gammas)+1,1,j+1);
    % plot(t,X(:,j)/meanX(j),'Color',colors(j,:),'LineWidth',1);
    plot(t,X(:,j),'Color',colors(j,:),'LineWidth',1); hold on;
    % yline(1,':','Color',[.6,.6,.6]); ylabel('X');
    % if gammas(j)>gc, label='positive efficiency'; else, label='zero efficiency'; end
    % title(sprintf('gamma = %.3g: %s',gammas(j),label));
    grid off; box off;
end
% xlabel('Time (h)');
linkaxes(ax,'x');
% xlim([0,min(T,12*p.Tu)]);
set(ax,'FontSize',11);

function [t,U,X,S,stats] = simulate_ring_bursts(p,gammas,T,dt,burn)
% Exact event-driven simulation; only the observation grid is discretized.
% State labels are 0,...,N-1. All downstream channels share the SAME
% upstream path, burst times and burst marks (a deliberate coupling).
k = p.N/p.Tu;
gammas = gammas(:).';
t = (0:floor(T/dt))'*dt;
U = zeros(size(t));
X = zeros(numel(t),numel(gammas));
S = zeros(size(t));
state = p.N/2; % stationary upstream initialization
% x = p.rhoBar*p.B./gammas; % residual transient removed by burn-in
x = 0; % residual transient removed by burn-in
now = -burn;
sample = 1;
stats.bursts = 0;
stats.ringJumps = 0;
stats.burstMass = 0;
while sample <= numel(t)
    rho = p.rhoBar*(1+p.epsilon*cos(2*pi*state/p.N+p.phi));
    rho = max(0,rho); % remove possible roundoff at epsilon = 1
    totalRate = k+rho;
    next = now-log(max(rand,realmin))/totalRate;
    while sample <= numel(t) && t(sample) < next
        X(sample,:) = x.*exp(-gammas*(t(sample)-now));
        U(sample) = p.B*rho;
        S(sample) = state;
        sample = sample+1;
    end
    if sample > numel(t), break; end
    x = x.*exp(-gammas*(next-now));
    now = next;
    if rand < k/totalRate
        state = mod(state+1,p.N);
        if now >= 0, stats.ringJumps = stats.ringJumps+1; end
    else
        burst = -p.B*log(max(rand,realmin));
        x = x+burst;
        if now >= 0
            stats.bursts = stats.bursts+1;
            stats.burstMass = stats.burstMass+burst;
        end
    end
end
stats.burnIn = burn;
stats.duration = t(end);
end

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
