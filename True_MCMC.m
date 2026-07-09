%% MCMC-Based Road Parameter Estimation for Suspension Mode Selection
% -------------------------------------------------------------------------
% Realistic window-based pipeline:
%
%   Road Profile Generator
%       -> Quarter-Car Vehicle Response
%       -> Noisy Sensor Measurements
%       -> MCMC Estimator of theta = [Phi0, w]^T
%       -> Suspension Mode Selector
%       -> Validation Metrics and Plots
%
% Important conceptual point:
%   The MCMC estimator receives only the measured vehicle response over each
%   time window. It does not receive the true road profile. It estimates the
%   statistical road parameters that are most compatible with the measured
%   chassis acceleration and suspension deflection.
%
% -------------------------------------------------------------------------

clear; close all; clc;
rng(7);

%% ------------------------------------------------------------------------
%  0. OUTPUT DIRECTORY
% -------------------------------------------------------------------------

outDir = fullfile(pwd, 'MCMC_Project_OnlineEstimator_Plots');

if ~exist(outDir, 'dir')
    mkdir(outDir);
end

%% ------------------------------------------------------------------------
%  1. BASE SIMULATION SETTINGS
% -------------------------------------------------------------------------

dt      = 0.005;        % simulation time step [s]
fs      = 1/dt;         % sampling frequency [Hz]
T_total = 30;           % total simulation time [s]
t       = 0:dt:T_total;
Nt      = numel(t);

%% ------------------------------------------------------------------------
%  2. ROAD GENERATION PARAMETERS
% -------------------------------------------------------------------------

road.Nwaves   = 240;       % delta_Omega (stepsize between wavenumbers)
road.OmegaMin = 0.08;      % Omega_1 [rad/m]
road.OmegaMax = 65;        % Omega_N [rad/m]
road.Omega0   = 1.0;       % Omega_0 [rad/m]

%% ------------------------------------------------------------------------
%  3. QUARTER-CAR MODEL PARAMETERS
% -------------------------------------------------------------------------

car.ms = 300;              % sprung mass [kg]
car.mu = 45;               % unsprung mass [kg]
car.ks = 16000;            % suspension stiffness [N/m]
car.kt = 190000;           % tire stiffness [N/m]

% Nominal suspension used while collecting the measured response
car.cs = 1500;             % [N s/m]

% Candidate suspension modes
modes.Firm.name    = "Firm/Cruise";
modes.Firm.cs      = 2200;
modes.Firm.ks      = car.ks;

modes.Normal.name  = "Normal";
modes.Normal.cs    = 1500;
modes.Normal.ks    = car.ks;

modes.Comfort.name = "Comfort/Raised";
modes.Comfort.cs   = 1100;
modes.Comfort.ks   = car.ks;

%% ------------------------------------------------------------------------
%  4. SENSOR MODEL: Adding a zero-mean, std to the measured chassis accel. &
%  suspension deflection
% -------------------------------------------------------------------------

sensor.sigmaAcc  = 0.12;       % chassis acceleration noise [m/s^2]
sensor.sigmaDefl = 0.0015;     % suspension deflection noise [m]

%% ------------------------------------------------------------------------
%  5. MCMC SETTINGS
% -------------------------------------------------------------------------

mcmc.nSamples     = 1200;                               % Number of MCMC iterations per estimation window. More samples means better posterior exploration, but slower runtime.
mcmc.burnIn       = 300;                                % First 300 samples are discarded because the chain may still be moving away from the initial guess.

% Proposal is applied to q = [log10(Phi0), w]; Larger values explore faster but may get rejected more often.
mcmc.proposalStd  = [0.035, 0.025];

% Uniform prior bounds
mcmc.logPhiBounds = [-6.4, -3.4];                       % prior bounds for log10(Phi0): 10e-6.4 <= Phi_allowed <= 10e-3.4
mcmc.wBounds      = [1.45, 2.55];                       % prior bounds for w as: 1.45<=w<=2.55

% Feature likelihood tuning.
% Feature vector:
%   [log RMS chassis acceleration,
%    log RMS suspension deflection,
%    log RMS chassis acceleration in 0.5-3 Hz band,
%    log RMS chassis acceleration in 3-15 Hz band]
like.sigmaFeat = [0.22, 0.24, 0.30, 0.35];

%% ------------------------------------------------------------------------
%  6. ESTIMATION WINDOW SETTINGS
% -------------------------------------------------------------------------

windowLength_s = 4.0;   %[s] of measured data
windowStep_s   = 2.0;   %[s] of window moving forward (windows overlap by windowLength_s - windowStep_s)

Nw = round(windowLength_s/dt);     % converts windowLength to Number of Samples (Nw)
Ns = round(windowStep_s/dt);       % converts windowStep to Number of Samples (Ns)

startIdx = 1:Ns:(Nt-Nw+1);         % starting index of the sampling
nWindows = numel(startIdx);        % number of sampling indices

%% ------------------------------------------------------------------------
%  7. DEFINE TRUE DRIVING SCENARIO
% -------------------------------------------------------------------------
% Default Scenario: (change parameters below to change the scenario)


%   0-10 s   : urban/average road
%   10-20 s  : highway/smooth road
%   20-30 s  : rural damaged/rough road with localized potholes

segments(1).name = "Urban";
segments(1).t0   = 0;
segments(1).t1   = 10;
segments(1).v    = 45/3.6;        % [m/s]
segments(1).Phi0 = 16e-6;         % Class C
segments(1).w    = 2.00;
segments(1).potholes = false;

segments(2).name = "Highway";
segments(2).t0   = 10;
segments(2).t1   = 20;
segments(2).v    = 80/3.6;       % [m/s]
segments(2).Phi0 = 2e-6;          % Class A/B
segments(2).w    = 2.05;
segments(2).potholes = false;

segments(3).name = "Rural damaged";
segments(3).t0   = 20;
segments(3).t1   = 30;
segments(3).v    = 20/3.6;        % [m/s]
segments(3).Phi0 = 70e-6;         % Class D
segments(3).w    = 1.90;
segments(3).potholes = true;

% Smooth speed transition settings
speedTransition.t_accel_start = 8.0;     % start accelerating before highway
speedTransition.t_accel_end   = 12.0;    % reach highway speed
speedTransition.t_decel_start = 18.0;    % start slowing before rough road
speedTransition.t_decel_end   = 22.0;    % reach rural speed

%% ------------------------------------------------------------------------
%  7B. BUILD SMOOTH SPEED (Transition) PROFILE
% -------------------------------------------------------------------------


% Speeds change like step functions between road classes. This section adds
% smooth transitions between them, taking from the settings of the last
% section above the initial and final velocities, and fixing zero
% derivative in the beginning and end of each transition section.

v_urban   = segments(1).v;
v_highway = segments(2).v;
v_rural   = segments(3).v;

v_profile = zeros(size(t));

for k = 1:Nt
    tk = t(k);

    if tk < speedTransition.t_accel_start
        v_profile(k) = v_urban;

    elseif tk <= speedTransition.t_accel_end
        alpha = smoothStep( ...
            (tk - speedTransition.t_accel_start) / ...
            (speedTransition.t_accel_end - speedTransition.t_accel_start) );

        v_profile(k) = (1-alpha)*v_urban + alpha*v_highway;

    elseif tk < speedTransition.t_decel_start
        v_profile(k) = v_highway;

    elseif tk <= speedTransition.t_decel_end
        alpha = smoothStep( ...
            (tk - speedTransition.t_decel_start) / ...
            (speedTransition.t_decel_end - speedTransition.t_decel_start) );

        v_profile(k) = (1-alpha)*v_highway + alpha*v_rural;

    else
        v_profile(k) = v_rural;
    end
end

% Travelled distance profile
s_profile = cumtrapz(t, v_profile);

%% ------------------------------------------------------------------------
%  8. GENERATE TRUE ROAD PROFILE
% -------------------------------------------------------------------------

zR_true     = zeros(size(t));

Phi0_true_t = zeros(size(t));
w_true_t    = zeros(size(t));
label_true  = strings(size(t));

previousEndValue = 0;

for i = 1:numel(segments)

    if i < numel(segments)
        idx = t >= segments(i).t0 & t < segments(i).t1;
    else
        idx = t >= segments(i).t0 & t <= segments(i).t1;
    end

    tloc = t(idx) - segments(i).t0;
    
    % Local travelled distance, shifted to start from zero
    sloc = s_profile(idx);
    sloc = sloc - sloc(1);
    
    theta_i.Phi0 = segments(i).Phi0;
    theta_i.w    = segments(i).w;
    
    zloc = generateRoadProfileFromDistance(sloc, theta_i, road, 100+i);

    % Make road height continuous at segment transition
    zloc = zloc - zloc(1) + previousEndValue;

    if segments(i).potholes
        zloc = addPotholes(tloc, zloc, ...
            [2.0, 5.2, 7.8], ...
            [-0.035, -0.025, -0.045], ...
            [0.20, 0.28, 0.22]);
    end

    zR_true(idx) = zloc(:).';
    previousEndValue = zloc(end);

    
    Phi0_true_t(idx) = segments(i).Phi0;
    w_true_t(idx)    = segments(i).w;
    label_true(idx)  = segments(i).name;
end

% Remove global offset only
zR_true = zR_true - mean(zR_true);

%% ------------------------------------------------------------------------
%  9. SIMULATE TRUE VEHICLE RESPONSE UNDER NOMINAL SUSPENSION
% -------------------------------------------------------------------------

[x_true, y_clean] = simulateQuarterCarConstantCs(t, zR_true, car);

% Noisy sensor measurements
y_meas = y_clean;
y_meas(:,1) = y_meas(:,1) + sensor.sigmaAcc  * randn(Nt,1);
y_meas(:,2) = y_meas(:,2) + sensor.sigmaDefl * randn(Nt,1);

%% ------------------------------------------------------------------------
%  10. WINDOW-BY-WINDOW MCMC ESTIMATION
% -------------------------------------------------------------------------

est.tmid        = zeros(nWindows,1);
est.Phi0_mean   = zeros(nWindows,1);
est.w_mean      = zeros(nWindows,1);
est.Phi0_map    = zeros(nWindows,1);
est.w_map       = zeros(nWindows,1);
est.acceptRate  = zeros(nWindows,1);
est.class       = strings(nWindows,1);
est.mode        = strings(nWindows,1);
est.samples     = cell(nWindows,1);

Phi0_true_win   = zeros(nWindows,1);
w_true_win      = zeros(nWindows,1);
class_true_win  = strings(nWindows,1);
mode_true_win   = strings(nWindows,1);

fprintf('\nRunning window-based MCMC road estimator...\n');

for iw = 1:nWindows

    k0 = startIdx(iw);
    k1 = k0 + Nw - 1;

    tw = t(k0:k1);
    yw = y_meas(k0:k1,:);
    vw = mean(v_profile(k0:k1));

    est.tmid(iw) = mean(tw);

    % True window values, used only for validation
    Phi0_true_win(iw)  = median(Phi0_true_t(k0:k1));
    w_true_win(iw)     = median(w_true_t(k0:k1));
    class_true_win(iw) = classifyRoad(Phi0_true_win(iw));
    mode_true_win(iw)  = selectSuspensionMode(Phi0_true_win(iw));

    % Initial guess: previous window estimate, or neutral class C initial
    if iw == 1
        q0 = [log10(16e-6), 2.0];
    else
        q0 = [log10(est.Phi0_mean(iw-1)), est.w_mean(iw-1)];
    end

    result = runMCMCWindow_OnlineResponse( ...
        tw - tw(1), yw, vw, car, road, mcmc, like, q0);

    est.samples{iw}    = result.samples;
    est.Phi0_mean(iw)  = result.Phi0_mean;
    est.w_mean(iw)     = result.w_mean;
    est.Phi0_map(iw)   = result.Phi0_map;
    est.w_map(iw)      = result.w_map;
    est.acceptRate(iw) = result.acceptRate;

    est.class(iw) = classifyRoad(est.Phi0_mean(iw));
    est.mode(iw)  = selectSuspensionMode(est.Phi0_mean(iw));

    fprintf(['Window %02d/%02d | t = %5.2f s | ', ...
             'Phi0 true = %.2e | Phi0 est = %.2e | ', ...
             'w true = %.2f | w est = %.2f | ', ...
             'class = %s | mode = %s | acc = %.2f\n'], ...
        iw, nWindows, est.tmid(iw), ...
        Phi0_true_win(iw), est.Phi0_mean(iw), ...
        w_true_win(iw), est.w_mean(iw), ...
        est.class(iw), est.mode(iw), est.acceptRate(iw));
end

%% ------------------------------------------------------------------------
%  11. VALIDATION METRICS
% -------------------------------------------------------------------------

phiErrLog = log10(est.Phi0_mean) - log10(Phi0_true_win);
wErr      = est.w_mean - w_true_win;

classCorrect = est.class == class_true_win;
modeCorrect  = est.mode  == mode_true_win;

meanAbsLogPhiErr = mean(abs(phiErrLog));
meanAbsWErr      = mean(abs(wErr));
classAccuracy    = mean(classCorrect);
modeAccuracy     = mean(modeCorrect);
meanAcceptRate   = mean(est.acceptRate);

fprintf('\nValidation summary:\n');
fprintf('Mean abs log10(Phi0) error: %.3f decades\n', meanAbsLogPhiErr);
fprintf('Mean abs w error:           %.3f\n', meanAbsWErr);
fprintf('Road class accuracy:        %.1f %%\n', 100*classAccuracy);
fprintf('Mode selection accuracy:    %.1f %%\n', 100*modeAccuracy);
fprintf('Mean MCMC acceptance rate:  %.2f\n', meanAcceptRate);

%% ------------------------------------------------------------------------
%  12. BUILD SELECTED MODE SCHEDULE AND SIMULATE ADAPTIVE RESPONSE
% -------------------------------------------------------------------------
% This is an evaluation step. The estimator has selected a mode per window.
% We now simulate how the same road would affect the car if the selected
% suspension damping were applied.

cs_adaptive = car.cs * ones(size(t));

for iw = 1:nWindows
    k0 = startIdx(iw);
    k1 = k0 + Nw - 1;

    cs_adaptive(k0:k1) = modeToDamping(est.mode(iw), modes);
end

% Fill any untouched samples with nearest previous value
for k = 2:Nt
    if cs_adaptive(k) == car.cs && k > startIdx(end)+Nw-1
        cs_adaptive(k) = cs_adaptive(k-1);
    end
end

[x_adapt, y_adapt] = simulateQuarterCarVariableCs(t, zR_true, car, cs_adaptive);

% Compare with fixed-normal response
rmsAcc_normal  = rms(y_clean(:,1));
rmsDefl_normal = rms(y_clean(:,2));
maxDefl_normal = max(abs(y_clean(:,2)));

rmsAcc_adapt   = rms(y_adapt(:,1));
rmsDefl_adapt  = rms(y_adapt(:,2));
maxDefl_adapt  = max(abs(y_adapt(:,2)));

fprintf('\nResponse comparison:\n');
fprintf('Fixed Normal RMS body acceleration:     %.4f m/s^2\n', rmsAcc_normal);
fprintf('Adaptive RMS body acceleration:         %.4f m/s^2\n', rmsAcc_adapt);
fprintf('Fixed Normal RMS suspension deflection: %.5f m\n', rmsDefl_normal);
fprintf('Adaptive RMS suspension deflection:     %.5f m\n', rmsDefl_adapt);
fprintf('Fixed Normal max suspension deflection: %.5f m\n', maxDefl_normal);
fprintf('Adaptive max suspension deflection:     %.5f m\n', maxDefl_adapt);

%% ------------------------------------------------------------------------
%  13. PLOTS
% -------------------------------------------------------------------------

fig = figure('Name','Road profile and true scenario','Color','w');
subplot(3,1,1);
plot(t, zR_true, 'LineWidth', 1.1);
grid on;
ylabel('$z_R(t)$ [m]', 'Interpreter','latex');
title('Generated Road Profile');

subplot(3,1,2);
plot(t, v_profile*3.6, 'LineWidth', 1.1);
grid on;
ylabel('Speed [km/h]');
title('Vehicle Speed');

subplot(3,1,3);
semilogy(t, Phi0_true_t, 'LineWidth', 1.1);
grid on;
ylabel('$\Phi_0$ true', 'Interpreter','latex');
xlabel('Time [s]');
title('True Road Roughness Parameter');
saveFigurePDF(fig, outDir, '01_road_profile_true_scenario');


%% Plot 2: Measured vehicle response with overlaid sensor error

acc_error  = y_meas(:,1) - y_clean(:,1);
defl_error = y_meas(:,2) - y_clean(:,2);

% Scale error signals only for visualization (if necessary)
acc_error_vis  = 1 * acc_error;
defl_error_vis = 1 * defl_error;

fig = figure('Name','Measured vehicle response with overlaid sensor error','Color','w');

subplot(2,1,1);
plot(t, y_clean(:,1), 'LineWidth', 1.15); hold on;
plot(t, y_meas(:,1), '.', 'MarkerSize', 2);
plot(t, acc_error_vis, 'g', 'LineWidth', 0.9);
grid on;
ylabel('$\ddot z_s$ [m/s$^2$]', 'Interpreter','latex');
legend({'clean','measured','error'}, ...
    'Interpreter','latex', 'Location','northwest');
title('Chassis Acceleration Measurement');

subplot(2,1,2);
plot(t, y_clean(:,2), 'LineWidth', 1.15); hold on;
plot(t, y_meas(:,2), '.', 'MarkerSize', 2);
plot(t, defl_error_vis, 'g', 'LineWidth', 0.9);
grid on;
ylabel('$z_s-z_u$ [m]', 'Interpreter','latex');
xlabel('Time [s]');
legend({'clean','measured','error'}, ...
    'Interpreter','latex', 'Location','northwest');
title('Suspension Deflection Measurement');

saveFigurePDF(fig, outDir, '02_noisy_vehicle_measurements_with_error_overlay');

%%


fig = figure('Name','MCMC parameter estimates','Color','w');
subplot(2,1,1);
semilogy(est.tmid, Phi0_true_win, 'o-', 'LineWidth', 1.2); hold on;
semilogy(est.tmid, est.Phi0_mean, 's-', 'LineWidth', 1.2);
grid on;
ylabel('$\Phi_0$', 'Interpreter','latex');
legend('true','estimated');
title('Road Roughness Parameter Estimation');

subplot(2,1,2);
plot(est.tmid, w_true_win, 'o-', 'LineWidth', 1.2); hold on;
plot(est.tmid, est.w_mean, 's-', 'LineWidth', 1.2);
grid on;
ylabel('$w$', 'Interpreter','latex');
xlabel('Time [s]');
legend('true','estimated');
title('Waviness Parameter Estimation');
saveFigurePDF(fig, outDir, '03_mcmc_parameter_estimates');


fig = figure('Name','Estimation errors','Color','w');
subplot(2,1,1);
bar(est.tmid, phiErrLog);
grid on;
ylabel('$\Delta \log_{10}(\Phi_0)$', 'Interpreter','latex');
title('Log-Roughness Estimation Error');

subplot(2,1,2);
bar(est.tmid, wErr);
grid on;
ylabel('$\Delta w$', 'Interpreter','latex');
xlabel('Time [s]');
title('Waviness Estimation Error');
saveFigurePDF(fig, outDir, '04_estimation_errors');


fig = figure('Name','Road class and selected mode','Color','w');
subplot(2,1,1);
stairs(est.tmid, categorical(est.class), 'LineWidth', 1.5); hold on;
stairs(est.tmid, categorical(class_true_win), '--', 'LineWidth', 1.2);
grid on;
ylabel('Road Class');
legend('estimated','true');
title('Estimated Road Class');

subplot(2,1,2);
stairs(est.tmid, categorical(est.mode), 'LineWidth', 1.5); hold on;
stairs(est.tmid, categorical(mode_true_win), '--', 'LineWidth', 1.2);
grid on;
ylabel('Suspension Mode');
xlabel('Time [s]');
legend(["selected","desired"], 'Location','best');
title('Selected Suspension Mode');
saveFigurePDF(fig, outDir, '05_class_and_mode_selection');

%% Plot 6: Estimator diagnostic summary

fig = figure('Name','Estimator diagnostic summary','Color','w');

subplot(3,1,1);
bar(est.tmid, phiErrLog);
grid on;
ylabel('$\Delta \log_{10}(\Phi_0)$', 'Interpreter','latex');
title('Log-Roughness Estimation Error');

subplot(3,1,2);
bar(est.tmid, wErr);
grid on;
ylabel('$\Delta w$', 'Interpreter','latex');
title('Waviness Estimation Error');

subplot(3,1,3);
plot(est.tmid, est.acceptRate, 'o-', 'LineWidth', 1.3); hold on;
% yline(0.2, '--', 'Lower useful range', 'LabelHorizontalAlignment','left');
% yline(0.6, '--', 'Upper useful range', 'LabelHorizontalAlignment','left');
grid on;
ylim([0 1]);
ylabel('Acceptance rate');
xlabel('Time [s]');
title('MCMC Acceptance Rate');

saveFigurePDF(fig, outDir, '06_estimator_diagnostic_summary');


%% Plot 7: Adaptive mode response, compact version

acc_diff  = y_adapt(:,1) - y_clean(:,1);
defl_diff = y_adapt(:,2) - y_clean(:,2);

fig = figure('Name','Adaptive suspension response compact comparison','Color','w');

subplot(3,1,1);
plot(t, acc_diff, 'LineWidth', 1.0);
grid on;
ylabel('$\Delta \ddot z_s$ [m/s$^2$]', 'Interpreter','latex');
title('Body Acceleration Difference: Adaptive - Fixed Normal');

subplot(3,1,2);
plot(t, defl_diff, 'LineWidth', 1.0);
grid on;
ylabel('$\Delta d$ [m]', 'Interpreter','latex');
title('Suspension Deflection Difference: Adaptive - Fixed Normal');

subplot(3,1,3);
plot(t, cs_adaptive, 'LineWidth', 1.2);
grid on;
ylabel('$c_s$ [N s/m]', 'Interpreter','latex');
xlabel('Time [s]');
title('Selected Damping Schedule');

saveFigurePDF(fig, outDir, '07_adaptive_response_difference_compact');


%% Plot 8: Segment-wise performance comparison

nSeg = numel(segments);

segNames = strings(nSeg,1);

rmsAcc_normal_seg  = zeros(nSeg,1);
rmsAcc_adapt_seg   = zeros(nSeg,1);

rmsDefl_normal_seg = zeros(nSeg,1);
rmsDefl_adapt_seg  = zeros(nSeg,1);

maxDefl_normal_seg = zeros(nSeg,1);
maxDefl_adapt_seg  = zeros(nSeg,1);

for i = 1:nSeg
    idx = t >= segments(i).t0 & t <= segments(i).t1;

    segNames(i) = segments(i).name;

    rmsAcc_normal_seg(i)  = rms(y_clean(idx,1));
    rmsAcc_adapt_seg(i)   = rms(y_adapt(idx,1));

    rmsDefl_normal_seg(i) = rms(y_clean(idx,2));
    rmsDefl_adapt_seg(i)  = rms(y_adapt(idx,2));

    maxDefl_normal_seg(i) = max(abs(y_clean(idx,2)));
    maxDefl_adapt_seg(i)  = max(abs(y_adapt(idx,2)));
end

fig = figure('Name','Segment-wise adaptive performance metrics','Color','w');

subplot(3,1,1);
bar(categorical(segNames), [rmsAcc_normal_seg, rmsAcc_adapt_seg]);
grid on;
ylabel('RMS acc. [m/s^2]');
legend({'fixed normal','adaptive'}, 'Location','best');
title('Segment-wise Body Acceleration RMS');

subplot(3,1,2);
bar(categorical(segNames), [rmsDefl_normal_seg, rmsDefl_adapt_seg]);
grid on;
ylabel('RMS defl. [m]');
legend({'fixed normal','adaptive'}, 'Location','best');
title('Segment-wise Suspension Deflection RMS');

subplot(3,1,3);
bar(categorical(segNames), [maxDefl_normal_seg, maxDefl_adapt_seg]);
grid on;
ylabel('Max $|z_s-z_u|$ [m]', 'Interpreter','latex');
legend({'fixed normal','adaptive'}, 'Location','best');
title('Segment-wise Maximum Suspension Deflection');

saveFigurePDF(fig, outDir, '08_segment_wise_adaptive_performance');


% Posterior traces for representative windows
plotWindows = unique([1, ceil(nWindows/2), nWindows]);

fig = figure('Name','MCMC posterior traces','Color','w');
for ii = 1:numel(plotWindows)
    iw = plotWindows(ii);
    S = est.samples{iw};

    subplot(numel(plotWindows),2,2*ii-1);
    plot(10.^S(:,1), 'LineWidth', 0.8);
    grid on;
    ylabel('$\Phi_0$', 'Interpreter','latex');
    title(sprintf('Window %d trace: Phi0', iw));

    subplot(numel(plotWindows),2,2*ii);
    plot(S(:,2), 'LineWidth', 0.8);
    grid on;
    ylabel('$w$', 'Interpreter','latex');
    title(sprintf('Window %d trace: w', iw));
end
saveFigurePDF(fig, outDir, '09_mcmc_posterior_traces');


fig = figure('Name','MCMC posterior histograms','Color','w');
for ii = 1:numel(plotWindows)
    iw = plotWindows(ii);
    S = est.samples{iw};
    Sburn = S(mcmc.burnIn+1:end,:);

    subplot(numel(plotWindows),2,2*ii-1);
    histogram(10.^Sburn(:,1), 25);
    grid on;
    xlabel('$\Phi_0$', 'Interpreter','latex');
    title(sprintf('Window %d posterior: Phi0', iw));

    subplot(numel(plotWindows),2,2*ii);
    histogram(Sburn(:,2), 25);
    grid on;
    xlabel('$w$', 'Interpreter','latex');
    title(sprintf('Window %d posterior: w', iw));
end
saveFigurePDF(fig, outDir, '10_mcmc_posterior_histograms');

%% ------------------------------------------------------------------------
%  14. SUMMARY TABLE
% -------------------------------------------------------------------------

summaryTable = table( ...
    est.tmid(:), ...
    Phi0_true_win(:), est.Phi0_mean(:), ...
    w_true_win(:), est.w_mean(:), ...
    class_true_win(:), est.class(:), ...
    mode_true_win(:), est.mode(:), ...
    est.acceptRate(:), ...
    classCorrect(:), modeCorrect(:), ...
    'VariableNames', { ...
    'Time_s', ...
    'Phi0_true', 'Phi0_est', ...
    'w_true', 'w_est', ...
    'Class_true', 'Class_est', ...
    'Mode_desired', 'Mode_selected', ...
    'AcceptanceRate', ...
    'ClassCorrect', 'ModeCorrect'} );

disp(summaryTable);

fprintf('\nAll figures saved as PDFs in:\n%s\n', outDir);

%% ========================================================================
%  LOCAL FUNCTIONS
% ========================================================================

function zR = generateRoadProfileTime(t, v, theta, road, seed)
% Generate road profile using sinusoidal approximation:
%
% z_R(s) = sum_i A_i sin(Omega_i s - psi_i)
%
% A_i = sqrt(2 Phi(Omega_i) DeltaOmega)

    rng(seed);

    t = t(:);
    s = v .* t;

    Omega = linspace(road.OmegaMin, road.OmegaMax, road.Nwaves);
    dOmega = (road.OmegaMax - road.OmegaMin)/(road.Nwaves - 1);

    Phi = theta.Phi0 .* (Omega./road.Omega0).^(-theta.w);
    A   = sqrt(2 .* Phi .* dOmega);

    psi = 2*pi*rand(size(Omega));

    zR = zeros(size(s));

    for i = 1:numel(Omega)
        zR = zR + A(i)*sin(Omega(i)*s - psi(i));
    end

    zR = zR - mean(zR);
end

function z = addPotholes(t, z, potholeTimes, depths, durations)
% Add smooth negative cosine-shaped potholes.

    t = t(:);
    z = z(:);

    for i = 1:numel(potholeTimes)
        tc  = potholeTimes(i);
        D   = depths(i);
        dur = durations(i);

        idx = abs(t - tc) <= dur/2;

        tau = (t(idx) - (tc - dur/2)) / dur;
        shape = 0.5*(1 - cos(2*pi*tau));

        z(idx) = z(idx) + D*shape;
    end
end

function [x, y] = simulateQuarterCarConstantCs(t, zR, car)
% Simulate quarter-car with constant damping coefficient.

    csProfile = car.cs * ones(size(t));
    [x, y] = simulateQuarterCarVariableCs(t, zR, car, csProfile);
end

function [x, y] = simulateQuarterCarVariableCs(t, zR, car, csProfile)
% Simulate quarter-car with possibly time-varying damping coefficient.
%
% State:
%   x = [z_s, dz_s, z_u, dz_u]^T
%
% Output:
%   y(:,1) = chassis acceleration
%   y(:,2) = suspension deflection

    t = t(:);
    zR = zR(:);
    csProfile = csProfile(:);

    dt = t(2)-t(1);
    Nt = numel(t);

    x = zeros(Nt,4);
    y = zeros(Nt,2);

    for k = 1:Nt-1

        car1 = car;
        car2 = car;
        car3 = car;
        car4 = car;

        car1.cs = csProfile(k);
        car2.cs = 0.5*(csProfile(k)+csProfile(k+1));
        car3.cs = car2.cs;
        car4.cs = csProfile(k+1);

        zr1 = zR(k);
        zr4 = zR(k+1);
        zrm = 0.5*(zr1+zr4);

        xk = x(k,:)';

        f1 = quarterCarRHS(xk, zr1, car1);
        f2 = quarterCarRHS(xk + 0.5*dt*f1, zrm, car2);
        f3 = quarterCarRHS(xk + 0.5*dt*f2, zrm, car3);
        f4 = quarterCarRHS(xk + dt*f3, zr4, car4);

        x(k+1,:) = xk' + (dt/6)*(f1 + 2*f2 + 2*f3 + f4)';
    end

    for k = 1:Nt
        carK = car;
        carK.cs = csProfile(k);
        [acc_s, defl] = quarterCarOutputs(x(k,:)', zR(k), carK);
        y(k,:) = [acc_s, defl];
    end
end

function dx = quarterCarRHS(x, zR, car)

    zs  = x(1);
    dzs = x(2);
    zu  = x(3);
    dzu = x(4);

    ms = car.ms;
    mu = car.mu;
    ks = car.ks;
    cs = car.cs;
    kt = car.kt;

    ddzs = (-cs*(dzs-dzu) - ks*(zs-zu))/ms;
    ddzu = ( cs*(dzs-dzu) + ks*(zs-zu) - kt*(zu-zR))/mu;

    dx = [dzs; ddzs; dzu; ddzu];
end

function [acc_s, defl] = quarterCarOutputs(x, ~, car)

    zs  = x(1);
    dzs = x(2);
    zu  = x(3);
    dzu = x(4);

    acc_s = (-car.cs*(dzs-dzu) - car.ks*(zs-zu))/car.ms;
    defl  = zs - zu;
end

function result = runMCMCWindow_OnlineResponse(tw, yw, v, car, road, mcmc, like, q0)
% MCMC estimator using only measured vehicle response features.
%
% q = [log10(Phi0), w]

    nSamples = mcmc.nSamples;

    qChain  = zeros(nSamples,2);
    logPost = zeros(nSamples,1);

    featMeas = responseFeatureVectorMeasured(tw, yw);

    qCurrent = q0;
    lpCurrent = logPosteriorResponse(qCurrent, featMeas, v, car, road, mcmc, like);

    nAccept = 0;

    for j = 1:nSamples

        qProp = qCurrent + mcmc.proposalStd .* randn(1,2);

        lpProp = logPosteriorResponse(qProp, featMeas, v, car, road, mcmc, like);

        if log(rand) < (lpProp - lpCurrent)
            qCurrent = qProp;
            lpCurrent = lpProp;
            nAccept = nAccept + 1;
        end

        qChain(j,:) = qCurrent;
        logPost(j)  = lpCurrent;
    end

    S  = qChain(mcmc.burnIn+1:end,:);
    LP = logPost(mcmc.burnIn+1:end);

    [~, idxMAP] = max(LP);

    result.samples = qChain;
    result.logPost = logPost;

    result.Phi0_mean = mean(10.^S(:,1));
    result.w_mean    = mean(S(:,2));

    result.Phi0_map = 10.^S(idxMAP,1);
    result.w_map    = S(idxMAP,2);

    result.acceptRate = nAccept/nSamples;
end

function lp = logPosteriorResponse(q, featMeas, v, car, road, mcmc, like)
% Log posterior for road parameters given measured response features.

    logPhi = q(1);
    w      = q(2);

    % Uniform priors
    if logPhi < mcmc.logPhiBounds(1) || logPhi > mcmc.logPhiBounds(2) || ...
       w < mcmc.wBounds(1) || w > mcmc.wBounds(2)
        lp = -Inf;
        return;
    end

    theta.Phi0 = 10^logPhi;
    theta.w    = w;

    featPred = responseFeatureVectorPredicted(theta, v, car, road);

    e = featMeas - featPred;

    lp = -0.5 * sum((e ./ like.sigmaFeat).^2);
end

function feat = responseFeatureVectorMeasured(t, y)
% Measured response features from sensor signals.

    acc  = y(:,1);
    defl = y(:,2);

    acc  = acc(:)  - mean(acc(:));
    defl = defl(:) - mean(defl(:));

    rmsAcc  = sqrt(mean(acc.^2)) + eps;
    rmsDefl = sqrt(mean(defl.^2)) + eps;

    rmsAccLow  = sqrt(simpleBandVariance(t, acc, 0.5, 3.0)) + eps;
    rmsAccHigh = sqrt(simpleBandVariance(t, acc, 3.0, 15.0)) + eps;

    feat = log([rmsAcc, rmsDefl, rmsAccLow, rmsAccHigh]);
end

function feat = responseFeatureVectorPredicted(theta, v, car, road)
% Predicted response features from road PSD and quarter-car transfer functions.
%
% Instead of generating a synthetic road, this function predicts the output
% variance by integrating:
%
%   var_y = int |G_y(j v Omega)|^2 Phi_R(Omega) dOmega
%
% over the road wave-number range.

    Omega = logspace(log10(road.OmegaMin), log10(road.OmegaMax), 900);

    PhiRoad = theta.Phi0 .* (Omega./road.Omega0).^(-theta.w);

    omega = v .* Omega;          % temporal rad/s seen by vehicle

    [Gacc, Gdefl] = quarterCarFRF(omega, car);

    % Full-band variances
    varAcc  = trapz(Omega, abs(Gacc).^2  .* PhiRoad);
    varDefl = trapz(Omega, abs(Gdefl).^2 .* PhiRoad);

    % Frequency bands in Hz, converted to temporal rad/s
    fLow1 = 0.5; fLow2 = 3.0;
    fHi1  = 3.0; fHi2  = 15.0;

    omegaLow1 = 2*pi*fLow1;
    omegaLow2 = 2*pi*fLow2;
    omegaHi1  = 2*pi*fHi1;
    omegaHi2  = 2*pi*fHi2;

    idxLow  = omega >= omegaLow1 & omega <= omegaLow2;
    idxHigh = omega >= omegaHi1  & omega <= omegaHi2;

    varAccLow = trapz(Omega(idxLow), ...
        abs(Gacc(idxLow)).^2 .* PhiRoad(idxLow));

    varAccHigh = trapz(Omega(idxHigh), ...
        abs(Gacc(idxHigh)).^2 .* PhiRoad(idxHigh));

    rmsAcc     = sqrt(max(varAcc, eps));
    rmsDefl    = sqrt(max(varDefl, eps));
    rmsAccLow  = sqrt(max(varAccLow, eps));
    rmsAccHigh = sqrt(max(varAccHigh, eps));

    feat = log([rmsAcc, rmsDefl, rmsAccLow, rmsAccHigh]);
end

function [Gacc, Gdefl] = quarterCarFRF(omega, car)
% Frequency response from road displacement z_R to:
%   1) chassis acceleration ddot z_s
%   2) suspension deflection z_s - z_u

    ms = car.ms;
    mu = car.mu;
    ks = car.ks;
    cs = car.cs;
    kt = car.kt;

    A = [ ...
        0,       1,        0,        0;
       -ks/ms, -cs/ms,    ks/ms,    cs/ms;
        0,       0,        0,        1;
        ks/mu,  cs/mu, -(ks+kt)/mu, -cs/mu];

    B = [0; 0; 0; kt/mu];

    Cacc  = [-ks/ms, -cs/ms, ks/ms, cs/ms];
    Cdefl = [1, 0, -1, 0];

    Dacc  = 0;
    Ddefl = 0;

    Gacc  = zeros(size(omega));
    Gdefl = zeros(size(omega));

    I = eye(4);

    for k = 1:numel(omega)
        jw = 1i*omega(k);
        H = (jw*I - A)\B;

        Gacc(k)  = Cacc*H  + Dacc;
        Gdefl(k) = Cdefl*H + Ddefl;
    end
end

function varBand = simpleBandVariance(t, x, f1, f2)
% FFT-based band variance estimate.

    t = t(:);
    x = x(:) - mean(x(:));

    dt = t(2)-t(1);
    fs = 1/dt;
    N  = numel(x);

    X = fft(x);

    % One-sided power approximation
    P2 = abs(X/N).^2;
    P1 = P2(1:floor(N/2)+1);
    P1(2:end-1) = 2*P1(2:end-1);

    f = fs*(0:floor(N/2))/N;

    idx = f >= f1 & f <= f2;

    if any(idx)
        varBand = sum(P1(idx));
    else
        varBand = eps;
    end
end

function cls = classifyRoad(Phi0)
% Approximate ISO-style road class using log-midpoint boundaries.

    if Phi0 < 2e-6
        cls = "A";
    elseif Phi0 < 8e-6
        cls = "B";
    elseif Phi0 < 32e-6
        cls = "C";
    elseif Phi0 < 128e-6
        cls = "D";
    else
        cls = "E";
    end
end

function mode = selectSuspensionMode(Phi0)
% Mode selector based on estimated roughness.

    if Phi0 < 8e-6
        mode = "Firm/Cruise";
    elseif Phi0 < 40e-6
        mode = "Normal";
    else
        mode = "Comfort/Raised";
    end
end

function cs = modeToDamping(mode, modes)

    mode = string(mode);

    switch mode
        case "Firm/Cruise"
            cs = modes.Firm.cs;
        case "Normal"
            cs = modes.Normal.cs;
        case "Comfort/Raised"
            cs = modes.Comfort.cs;
        otherwise
            cs = modes.Normal.cs;
    end
end

function zR = generateRoadProfileFromDistance(s, theta, road, seed)
% Generate road profile using sinusoidal approximation with distance input:
%
% z_R(s) = sum_i A_i sin(Omega_i s - psi_i)
%
% This version allows non-constant vehicle speed because s(t) is provided
% directly from the travelled-distance profile.

    rng(seed);

    s = s(:);

    Omega = linspace(road.OmegaMin, road.OmegaMax, road.Nwaves);
    dOmega = (road.OmegaMax - road.OmegaMin)/(road.Nwaves - 1);

    Phi = theta.Phi0 .* (Omega./road.Omega0).^(-theta.w);
    A   = sqrt(2 .* Phi .* dOmega);

    psi = 2*pi*rand(size(Omega));

    zR = zeros(size(s));

    for i = 1:numel(Omega)
        zR = zR + A(i)*sin(Omega(i)*s - psi(i));
    end

    zR = zR - mean(zR);
end

function y = smoothStep(x)
% Smooth transition from 0 to 1 with zero slope at both ends.

    x = max(0, min(1, x));
    y = 3*x.^2 - 2*x.^3;
end

function saveFigurePDF(fig, outDir, fileName)
% Save MATLAB figure as PDF.

    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    pdfPath = fullfile(outDir, [fileName, '.pdf']);

    set(fig, 'Color', 'w');

    try
        exportgraphics(fig, pdfPath, ...
            'ContentType', 'vector', ...
            'BackgroundColor', 'white');
    catch
        set(fig, 'PaperPositionMode', 'auto');
        print(fig, pdfPath, '-dpdf', '-bestfit');
    end

    fprintf('Saved figure: %s\n', pdfPath);
end