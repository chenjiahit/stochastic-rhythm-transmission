clear

%% Parameters: edit this block
p.N = 8;
p.Tu = 5;
p.rhoBar = 10;
p.B = 3;
p.epsilon = 1;
p.phi = 0;
gammas = [0.1,1.5];
dt = 0.01;
T = 1e4; % long observation improves the empirical spectral estimate
segmentLength = 1e4; % duration 163.84 h; 50% overlap
burn = max(20*p.Tu,12/min(gammas));

outDir = fullfile(fileparts(mfilename('fullpath')),'results');
if ~exist(outDir,'dir'), mkdir(outDir); end
[gc,R] = ring_threshold(p,p.rhoBar);
[t,U,X,~,stats] = simulate_ring_bursts(p,gammas,T,dt,burn);
stationaryMeans = [p.rhoBar*p.B,p.rhoBar*p.B./gammas];
centered = bsxfun(@minus,[U,X],stationaryMeans);
[omega,P,welchInfo] = welch_angular(centered,dt,segmentLength);
[GuTheory,GTheory,Gu0,D] = ring_spectra(p,omega,gammas);
G0 = (Gu0+D)./gammas.^2;
theory = [GuTheory,GTheory];
reference = [Gu0,G0];
sampleMeans = mean([U,X],1);
sampleVariances = var([U,X],1,1);
mu = 2*(p.N/p.Tu)*sin(pi/p.N)^2;
nu = (p.N/p.Tu)*sin(2*pi/p.N);
A = (p.rhoBar*p.B*p.epsilon)^2;
theoryVariances = [A/2, ...
    A*(gammas+mu)./(2*gammas.*((gammas+mu).^2+nu^2))+D./(2*gammas)];
omegaLimit = 3*(2*pi/p.Tu);
mask = omega>0 & omega<=omegaLimit;

%% Raw spectra: direct comparison tests absolute normalization as well
fig = figure('Color','w','Position',[70,60,1150,600]);
for j = 1:numel(gammas)+1
    subplot(2,ceil((numel(gammas)+1)/2),j);
    semilogy(omega(mask),P(mask,j),'Color',[.65,.72,.82],'LineWidth',0.8); hold on;
    semilogy(omega(mask),theory(mask,j),'Color',[.05,.25,.65],'LineWidth',1.7);
    plot(0,reference(j),'o','Color',[.75,.15,.12],'MarkerFaceColor',[.75,.15,.12]);
    if j==1
        title('Upstream driving U_t'); ylabel('G_u(omega)');
    else
        title(sprintf('Downstream: gamma = %.3g',gammas(j-1)));
        ylabel('G(omega)');
    end
    xlabel('Angular frequency omega (rad)');
    xlim([0,omegaLimit]); grid on; box off;
    legend('Simulation: Welch','Exact continuous-time PSD','Exact PSD at zero', ...
           'Location','best');
    set(gca,'FontSize',10);
end

%% Normalized theoretical spectra: peaks above one have positive efficiency
maxnum = 4001; itv = 200;
omegaFine = linspace(0,omegaLimit,maxnum)';
[GuFine,GFine] = ring_spectra(p,omegaFine,gammas);
figNormalized = figure('Color','w','Position',[110,100,1020,570]);
plot(omegaFine,GuFine/Gu0,'k','LineWidth',1.5); hold on;
colors = lines(numel(gammas));
labels = {'Upstream: G_u / G_u(0)'};
for j = 1:numel(gammas)
    plot(omegaFine,GFine(:,j)/G0(j),'Color',colors(j,:),'LineWidth',1.5); hold on
    labels{end+1} = sprintf('Downstream: gamma = %.3g',gammas(j)); %#ok<SAGROW>
end
yline(1,':','Color',[.5,.5,.5],'HandleVisibility','off');
xlabel('Angular frequency omega (rad)');
ylabel('Spectrum / zero-frequency spectrum');
title(sprintf('Exact spectral transmission: R = %.3g, gamma_c = %.3g',R,gc));
legend(labels,'Location','northeast');
grid off; box off;
set(gca,'FontSize',12); xlim([0,omegaLimit]);
% Finite sampling aliases continuous-time power; finite windows smooth it.
% These errors are small in the displayed band for the defaults, but the
% Welch curve is not expected to coincide exactly with the analytical PSD.
% Do not estimate efficiency by taking the maximum of a noisy periodogram.

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
state = randi(p.N)-1; % stationary upstream initialization
x = p.rhoBar*p.B./gammas; % residual transient removed by burn-in
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

function [Gu,G,Gu0,D] = ring_spectra(p,omega,gammas)
% Two-sided continuous-time PSD evaluated at nonnegative ANGULAR frequency.
% G(omega) = integral C(t)*exp(-1i*omega*t) dt.
% Lorentzian representation is equivalent to Eq. (23) and is stable.
k = p.N/p.Tu;
theta = 2*pi/p.N;
mu = 2*k*sin(theta/2)^2;
nu = k*sin(theta);
A = (p.rhoBar*p.B*p.epsilon)^2;
omega = omega(:);
Gu = (A*mu/2)*(1./(mu^2+(omega-nu).^2) ...
                     +1./(mu^2+(omega+nu).^2));
Gu0 = A/(2*k);
D = 2*p.rhoBar*p.B^2;
G = bsxfun(@rdivide,Gu+D,bsxfun(@plus,omega.^2,gammas(:).'.^2));
end

function [omega,P,info] = welch_angular(x,dt,L)
% Toolbox-free Welch estimate with 50% overlap and a periodic Hann window.
% Input must already have its stationary mean removed. No per-window
% demeaning: that operation would artificially suppress low frequencies.
% P is TWO-SIDED density restricted to omega >= 0: do NOT double bins.
% If a one-sided pwelch result were used, divide positive bins by two.
% No additional 2*pi factor: S_f(f) = G(2*pi*f) under the paper's convention.
hop = L/2;
starts = 1:hop:(size(x,1)-L+1);
w = 0.5-0.5*cos(2*pi*(0:L-1)'/L);
P = zeros(L/2+1,size(x,2));
for a = starts
    F = fft(bsxfun(@times,x(a:a+L-1,:),w),L,1);
    P = P+abs(F(1:L/2+1,:)).^2;
end
fs = 1/dt;
P = P/(numel(starts)*fs*sum(w.^2));
omega = 2*pi*(0:L/2)'*fs/L;
info.segments = numel(starts);
info.segmentDuration = L*dt;
info.frequencySpacing = 2*pi/(L*dt);
info.meanRemoved = 'Analytical stationary mean, not window means';
end
