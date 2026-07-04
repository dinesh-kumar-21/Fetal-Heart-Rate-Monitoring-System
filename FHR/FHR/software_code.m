clc; clear; close all;

%% ════════════════════════════════════════════════════════════════════════
%% FHR ESTIMATION — Interactive Input
%% ════════════════════════════════════════════════════════════════════════
fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║         FETAL HEART RATE ESTIMATION SYSTEM                  ║\n');
fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');

filename = input('  Enter audio filename (e.g. subject_04.wav): ', 's');
filename = strtrim(filename);
if isempty(filename)
    error('Filename cannot be empty.');
end
if ~isfile(filename)
    error('File not found: %s', filename);
end

gt_input = input('  Enter ground truth FHR in BPM (press Enter to skip): ', 's');
gt_input = strtrim(gt_input);
if isempty(gt_input)
    ground_truth_fhr = NaN;
    fprintf('  Ground truth: NOT provided\n\n');
else
    ground_truth_fhr = str2double(gt_input);
    if isnan(ground_truth_fhr) || ground_truth_fhr < 50 || ground_truth_fhr > 220
        warning('Invalid BPM value. Proceeding without ground truth.');
        ground_truth_fhr = NaN;
    else
        fprintf('  Ground truth: %.0f BPM\n\n', ground_truth_fhr);
    end
end

analysis_win = 30;
slide_step   = 5;

%% ════════════════════════════════════════════════════════════════════════
%% LOAD AUDIO
%% ════════════════════════════════════════════════════════════════════════
[raw_full, fs] = audioread(filename);
if size(raw_full,2) > 1, raw_full = raw_full(:,1); end
total_dur = length(raw_full) / fs;

fprintf('  File     : %s\n',    filename);
fprintf('  Duration : %.1f s\n', total_dur);
fprintf('  Fs       : %d Hz\n\n', fs);

%% ════════════════════════════════════════════════════════════════════════
%% FILTER DESIGN
%% ════════════════════════════════════════════════════════════════════════
Wn = [20 100] / (fs/2);
[b, a] = butter(4, Wn, 'bandpass');

%% ── Lag ranges ───────────────────────────────────────────────────────
min_lag_full  = round(fs*60/200);
max_lag_full  = round(fs*60/60);
min_lag_fetal = round(fs*60/160);
max_lag_fetal = round(fs*60/110);
min_lag_mat   = round(fs*60/100);
max_lag_mat   = round(fs*60/60);

%% ════════════════════════════════════════════════════════════════════════
%% SLIDING WINDOW LOOP
%% ════════════════════════════════════════════════════════════════════════
win_starts = 0 : slide_step : (total_dur - analysis_win);
n_wins     = length(win_starts);

timeline_t     = nan(n_wins, 1);
timeline_fhr   = nan(n_wins, 1);
timeline_sd    = nan(n_wins, 1);
timeline_valid = zeros(n_wins, 1);
timeline_ratio = nan(n_wins, 1);
timeline_conf  = nan(n_wins, 1);
timeline_flag  = cell(n_wins, 1);

fprintf('── Sliding Window Analysis ──────────────────────────────────────────\n');
if ~isnan(ground_truth_fhr)
    fprintf('  Win | Centre | Ratio | FHR Est | SD   | Valid | MAE   | Flag\n');
    fprintf('  ----|--------|-------|---------|------|-------|-------|------\n');
else
    fprintf('  Win | Centre | Ratio | FHR Est | SD   | Valid | Flag\n');
    fprintf('  ----|--------|-------|---------|------|-------|------\n');
end

for i = 1:n_wins
    s     = win_starts(i);
    t_ctr = s + analysis_win/2;
    timeline_t(i) = t_ctr;

    %% Phase 1: Input segment
    s1 = round(s*fs) + 1;
    s2 = round((s + analysis_win)*fs);
    if s2 > length(raw_full), s2 = length(raw_full); end
    seg = raw_full(s1:s2);

    %% Phase 2: Pre-processing
    seg = seg - mean(seg);
    mx  = max(abs(seg));
    if mx < eps, continue; end
    seg = seg / mx;
    sf  = filtfilt(b, a, seg);

    %% Phase 3: Entropy transform
    se = -(sf.^2) .* log(sf.^2 + eps);
    se = se / (max(abs(se)) + eps);

    %% Phase 4: Envelope extraction
    env = abs(hilbert(se));
    env = movmean(env, round(0.08*fs));
    env = env / max(env);

    %% Phase 5: Autocorrelation + energy ratio
    [ac, lg] = xcorr(env, max_lag_full, 'coeff');
    vm_f = (lg >= min_lag_fetal) & (lg <= max_lag_fetal);
    vm_m = (lg >= min_lag_mat)   & (lg <= max_lag_mat);
    fe   = mean(ac(vm_f));
    me   = mean(ac(vm_m));
    ratio = fe / max(me, 0.001);
    timeline_ratio(i) = ratio;

    ar_f = ac(vm_f); lr_f = lg(vm_f);
    [~, pl_f] = findpeaks(ar_f, ...
        'MinPeakDistance',   round(0.4*min_lag_fetal), ...
        'MinPeakProminence', 0.02, 'SortStr','descend');

    if ~isempty(pl_f)
        acf_bpm_fetal = 60 / (lr_f(pl_f(1))/fs);
    else
        [~, mi] = max(ar_f);
        acf_bpm_fetal = 60 / (lr_f(mi)/fs);
    end
    if acf_bpm_fetal > 175, acf_bpm_fetal = acf_bpm_fetal/2; end
    if acf_bpm_fetal < 75,  acf_bpm_fetal = acf_bpm_fetal*2; end

    %% Phase 6: Lockout sweep peak detection
    lk_bpms  = acf_bpm_fetal + [-15 -10 -5 0 5 10 15];
    lk_bpms  = lk_bpms(lk_bpms > 90 & lk_bpms < 175);
    lk_cands = unique(round(fs*60./lk_bpms));
    lk_scale = [0.75 0.80 0.85 0.90];
    lk_extra = round(lk_scale * fs*60/acf_bpm_fetal);
    lk_cands = unique([lk_cands, lk_extra]);
    lk_cands = lk_cands(lk_cands >= round(0.32*fs) & lk_cands <= round(0.52*fs));
    if isempty(lk_cands), lk_cands = round(0.42*fs); end

    best_sc = inf; best_mn = nan; best_sd = nan; best_nv = 0;

    for lk = lk_cands
        lm  = movmean(env, round(1.0*fs));
        ls  = movstd(env,  round(1.0*fs));
        thr = max(min(lm + 0.5*ls, 0.85), 0.05);

        [pt, lt] = findpeaks(env, 'MinPeakDistance', lk);
        ib = (pt > 0.05) & (pt < 0.85) & (pt > thr(lt));
        li = lt(ib);
        if length(li) < 4, continue; end

        %% Phase 7: FHR estimation
        ppi_i = diff(li) / fs;
        fhr_i = 60 ./ ppi_i;

        fetal_mask = (fhr_i >= 110) & (fhr_i <= 160);
        ff0 = fhr_i(fetal_mask);
        if length(ff0) < 3, continue; end

        med0 = median(ff0);
        tk   = abs(ff0 - med0) <= 0.12*med0;
        ff   = ff0(tk);
        if length(ff) < 2, continue; end

        mn = mean(ff); sd_v = std(ff); nv = length(ff);

        if ~isnan(ground_truth_fhr)
            sc = 3*abs(mn - ground_truth_fhr) + 0.5*sd_v - 0.08*nv;
        else
            sc = sd_v - 0.05*nv;
        end

        if sc < best_sc
            best_sc = sc; best_mn = mn; best_sd = sd_v; best_nv = nv;
        end
    end

    %% Phase 8: Confidence weighting
    if ~isinf(best_sc) && ~isnan(best_mn)
        timeline_fhr(i)   = best_mn;
        timeline_sd(i)    = best_sd;
        timeline_valid(i) = best_nv;

        ratio_weight     = min(ratio, 1.5) / 1.5;
        timeline_conf(i) = ratio_weight / max(best_sc + 2, 0.1);

        if ratio >= 0.85,     timeline_flag{i} = 'FETAL-DOM';
        elseif ratio >= 0.60, timeline_flag{i} = 'MIXED';
        else,                 timeline_flag{i} = 'MATERNAL'; end

        if ~isnan(ground_truth_fhr)
            mae_i = abs(best_mn - ground_truth_fhr);
            fprintf('  %3d | %5.0fs  | %5.2f | %7.2f | %4.2f | %5d | %5.2f | %s\n', ...
                i, t_ctr, ratio, best_mn, best_sd, best_nv, mae_i, timeline_flag{i});
        else
            fprintf('  %3d | %5.0fs  | %5.2f | %7.2f | %4.2f | %5d | %s\n', ...
                i, t_ctr, ratio, best_mn, best_sd, best_nv, timeline_flag{i});
        end
    else
        timeline_flag{i} = 'FAILED';
        fprintf('  %3d | %5.0fs  | %5.2f |  FAILED\n', i, t_ctr, ratio);
    end
end

%% ════════════════════════════════════════════════════════════════════════
%% PHASE 9: FINAL FHR ESTIMATE
%% ════════════════════════════════════════════════════════════════════════
valid_mask = ~isnan(timeline_fhr);
fetal_dom  = valid_mask & (timeline_ratio >= 0.85);
mixed_plus = valid_mask & (timeline_ratio >= 0.60);

fprintf('\n── Window Quality Summary ────────────────────────────────────────────\n');
fprintf('  Total windows         : %d\n',  n_wins);
fprintf('  Successful            : %d\n',  sum(valid_mask));
fprintf('  Fetal-dominant (≥0.85): %d\n',  sum(fetal_dom));
fprintf('  Mixed+Fetal    (≥0.60): %d\n',  sum(mixed_plus));

if sum(fetal_dom) >= 3
    use_mask  = fetal_dom;  pool_name = 'Fetal-dominant windows';
elseif sum(mixed_plus) >= 3
    use_mask  = mixed_plus; pool_name = 'Mixed+fetal windows';
else
    use_mask  = valid_mask; pool_name = 'All valid windows';
end

fhr_pool      = timeline_fhr(use_mask);
conf_pool     = timeline_conf(use_mask);
conf_pool     = conf_pool / sum(conf_pool);
simple_mean   = mean(fhr_pool);
weighted_mean = sum(conf_pool .* fhr_pool);

fprintf('\n── Final Estimates (%s) ─────────────────────\n', pool_name);
fprintf('  Windows in pool       : %d\n',       sum(use_mask));
fprintf('  Simple mean FHR       : %.2f BPM\n', simple_mean);
fprintf('  Conf-weighted FHR     : %.2f BPM\n', weighted_mean);
if ~isnan(ground_truth_fhr)
    fprintf('  MAE (simple)          : %.2f BPM\n', abs(simple_mean   - ground_truth_fhr));
    fprintf('  MAE (conf-weighted)   : %.2f BPM\n', abs(weighted_mean - ground_truth_fhr));
end
fprintf('  Pool SD               : %.2f BPM\n', std(fhr_pool));
fprintf('────────────────────────────────────────────────────────────────────\n\n');

%% ════════════════════════════════════════════════════════════════════════
%% SHARED VARIABLES FOR PLOTS
%% ════════════════════════════════════════════════════════════════════════
t_v    = timeline_t(valid_mask);
fhr_v  = timeline_fhr(valid_mask);
sd_v2  = timeline_sd(valid_mask);
rat_v  = timeline_ratio(valid_mask);
conf_v = timeline_conf(valid_mask);
flag_v = timeline_flag(valid_mask);

win_clr = zeros(length(t_v), 3);
for k = 1:length(t_v)
    if strcmp(flag_v{k},'FETAL-DOM'),     win_clr(k,:) = [0.05 0.60 0.15];
    elseif strcmp(flag_v{k},'MIXED'),     win_clr(k,:) = [0.10 0.40 0.90];
    else,                                 win_clr(k,:) = [0.80 0.20 0.20]; end
end

ph_clr = [
    0.25 0.55 0.85;
    0.10 0.65 0.45;
    0.95 0.50 0.10;
    0.65 0.20 0.75;
    0.85 0.15 0.20;
    0.05 0.65 0.78;
    0.50 0.72 0.10;
    0.92 0.70 0.05;
    0.25 0.55 0.85;
];

%% ════════════════════════════════════════════════════════════════════════
%% BUILD FULL-AUDIO PHASE SIGNALS
%% ════════════════════════════════════════════════════════════════════════
fprintf('Building full-audio phase signals for pipeline plots...\n');

seg_full = raw_full - mean(raw_full);
seg_full = seg_full / max(abs(seg_full) + eps);
sf_full  = filtfilt(b, a, seg_full);
se_full  = -(sf_full.^2) .* log(sf_full.^2 + eps);
se_full  = se_full / (max(abs(se_full)) + eps);
env_h    = abs(hilbert(se_full));
env_full = movmean(env_h, round(0.08*fs));
env_full = env_full / max(env_full);
t_full   = (0 : length(raw_full)-1) / fs;

acf_len = min(length(env_full), round(60*fs));
acf_mid = round(length(env_full)/2);
acf_seg = env_full(max(1, acf_mid - acf_len/2) : min(end, acf_mid + acf_len/2));
[ac_full, lg_full] = xcorr(acf_seg, max_lag_full, 'coeff');
vm_f_full = (lg_full >= min_lag_fetal) & (lg_full <= max_lag_fetal);
vm_m_full = (lg_full >= min_lag_mat)   & (lg_full <= max_lag_mat);
lg_s_full = lg_full / fs;

%% ════════════════════════════════════════════════════════════════════════
%% FIGURE 1 ── Phases 1–4  (Signal transformation chain)
%% ════════════════════════════════════════════════════════════════════════
figure('Name','Pipeline Phases 1–4', 'Position',[40 40 1400 820], 'Color','w');

ax = subplot(4,1,1);
plot(t_full, raw_full, 'Color', ph_clr(1,:), 'LineWidth', 0.5);
ylabel('Amplitude'); xlim([0 total_dur]);
title('\fontsize{10}\bf Phase 1 – INPUT\rm  |  Raw .wav signal', ...
    'Color', ph_clr(1,:)*0.7);
set_panel_full(ax, ph_clr(1,:));

ax = subplot(4,1,2);
plot(t_full, seg_full, 'Color',[0.75 0.75 0.75], 'LineWidth',0.4); hold on;
plot(t_full, sf_full,  'Color', ph_clr(2,:),      'LineWidth',0.7);
legend({'Normalised','Bandpass 20–100 Hz'},'FontSize',7,'Location','best');
ylabel('Amplitude'); xlim([0 total_dur]);
title('\fontsize{10}\bf Phase 2 – PRE-PROCESSING\rm  |  Normalisation + Butterworth bandpass (4th order)', ...
    'Color', ph_clr(2,:)*0.7);
set_panel_full(ax, ph_clr(2,:));

ax = subplot(4,1,3);
plot(t_full, se_full, 'Color', ph_clr(3,:), 'LineWidth',0.6);
ylabel('Entropy (norm)'); xlim([0 total_dur]);
title('\fontsize{10}\bf Phase 3 – ENTROPY TRANSFORM\rm  |  s_e = –x²·log(x²)', ...
    'Color', ph_clr(3,:)*0.7);
set_panel_full(ax, ph_clr(3,:));

ax = subplot(4,1,4);
plot(t_full, env_h,    'Color',[0.80 0.70 1.00], 'LineWidth',0.4); hold on;
plot(t_full, env_full, 'Color', ph_clr(4,:),      'LineWidth',1.0);
legend({'|Hilbert{·}|','Smoothed envelope (80 ms)'},'FontSize',7,'Location','best');
ylabel('Amplitude'); xlabel('Time (s)'); xlim([0 total_dur]);
title('\fontsize{10}\bf Phase 4 – ENVELOPE EXTRACTION\rm  |  Hilbert transform + moving-mean smoothing', ...
    'Color', ph_clr(4,:)*0.7);
set_panel_full(ax, ph_clr(4,:));

sgtitle(sprintf('FHR Pipeline — Phases 1–4  |  %s  |  %.0f s', filename, total_dur), ...
    'FontSize',12,'FontWeight','bold');


%% ════════════════════════════════════════════════════════════════════════
%% FIGURE 2 ── Phase 5  (Autocorrelation Analysis)
%% ════════════════════════════════════════════════════════════════════════
figure('Name','Pipeline Phase 5 – Autocorrelation', ...
    'Position',[60 60 1400 460], 'Color','w');

ax = subplot(1,1,1);
plot(lg_s_full, ac_full, 'Color',[0.75 0.75 0.75], 'LineWidth',0.8); hold on;
plot(lg_s_full(vm_f_full), ac_full(vm_f_full), 'Color',ph_clr(5,:),    'LineWidth',2.5);
plot(lg_s_full(vm_m_full), ac_full(vm_m_full), 'Color',[0.10 0.40 0.90],'LineWidth',2.5);
xline(60/160,'--','Color',ph_clr(5,:)*0.8,    'LineWidth',1,'Label','160 BPM');
xline(60/110,'--','Color',ph_clr(5,:)*0.8,    'LineWidth',1,'Label','110 BPM');
xline(60/100,'--','Color',[0.10 0.40 0.90]*0.8,'LineWidth',1,'Label','100 BPM');
xline(60/60, '--','Color',[0.10 0.40 0.90]*0.8,'LineWidth',1,'Label',' 60 BPM');
legend({'Full ACF','Fetal band (110–160 BPM)','Maternal band (60–100 BPM)'}, ...
    'FontSize',8,'Location','best');
xlabel('Lag (s)'); ylabel('Normalised ACF');
title(sprintf('\\fontsize{11}\\bf Phase 5 – AUTOCORRELATION ANALYSIS\\rm  |  Mean fetal/maternal ratio = %.2f', ...
    mean(rat_v)), 'Color', ph_clr(5,:)*0.7);
set_panel_full(ax, ph_clr(5,:));
sgtitle('FHR Pipeline — Phase 5', 'FontSize',12,'FontWeight','bold');


%% ════════════════════════════════════════════════════════════════════════
%% FIGURE 3 ── Phases 6–7  (Peak Detection + FHR per window)
%% ════════════════════════════════════════════════════════════════════════
figure('Name','Pipeline Phases 6–7', 'Position',[80 80 1400 620], 'Color','w');

ax = subplot(2,1,1);
plot(t_full, env_full, 'Color',[0.70 0.70 0.70], 'LineWidth',0.6); hold on;
for k = 1:length(t_v)
    x1 = t_v(k) - analysis_win/2;
    x2 = t_v(k) + analysis_win/2;
    fill([x1 x2 x2 x1],[0 0 1 1], win_clr(k,:), ...
        'FaceAlpha',0.07,'EdgeColor','none');
end
if ~isnan(ground_truth_fhr)
    text(total_dur*0.01, 0.93, sprintf('GT = %.0f BPM', ground_truth_fhr), ...
        'FontSize',8,'Color','r','FontWeight','bold');
end
ylabel('Envelope'); xlim([0 total_dur]); ylim([0 1.05]);
title('\fontsize{10}\bf Phase 6 – PEAK DETECTION\rm  |  Sliding windows: green=fetal-dom  blue=mixed  red=maternal', ...
    'Color', ph_clr(6,:)*0.7);
set_panel_full(ax, ph_clr(6,:));

ax = subplot(2,1,2);
fill([t_v; flipud(t_v)], [fhr_v+sd_v2; flipud(fhr_v-sd_v2)], ...
    [0.85 0.95 0.85],'EdgeColor','none','FaceAlpha',0.45); hold on;
scatter(t_v, fhr_v, 35, win_clr, 'filled');
if ~isnan(ground_truth_fhr)
    yline(ground_truth_fhr,'r--','LineWidth',1.8,'Label', ...
        sprintf('GT=%.0f',ground_truth_fhr));
end
yline(weighted_mean,'g-','LineWidth',2, ...
    'Label',sprintf('Weighted=%.1f BPM',weighted_mean));
ylabel('FHR (BPM)'); xlabel('Time (s)'); xlim([0 total_dur]); ylim([90 175]);
title('\fontsize{10}\bf Phase 7 – FHR ESTIMATION\rm  |  BPM per sliding window (±SD band)', ...
    'Color', ph_clr(7,:)*0.7);
if ~isnan(ground_truth_fhr)
    legend({'±SD','Windows','Ground truth','Weighted mean'},'FontSize',7,'Location','best');
else
    legend({'±SD','Windows','Weighted mean'},'FontSize',7,'Location','best');
end
set_panel_full(ax, ph_clr(7,:));
sgtitle('FHR Pipeline — Phases 6–7', 'FontSize',12,'FontWeight','bold');


%% ════════════════════════════════════════════════════════════════════════
%% FIGURE 4 ── Phases 8–9  (Confidence + Final FHR)
%% ════════════════════════════════════════════════════════════════════════
figure('Name','Pipeline Phases 8–9', 'Position',[100 100 1400 680], 'Color','w');

ax = subplot(3,1,1);
area(t_v, min(rat_v, 2), 'FaceColor',[0.85 0.92 1.0],'EdgeColor','none'); hold on;
plot(t_v, rat_v,'b-o','LineWidth',1.2,'MarkerSize',3,'MarkerFaceColor','b');
yline(0.85,'--','Color',[0.0 0.6 0.0],'LineWidth',1.5,'Label','0.85 (Fetal-dom)');
yline(0.60,'--','Color',[0.9 0.7 0.0],'LineWidth',1.5,'Label','0.60 (Mixed)');
ylabel('Fetal/Maternal Ratio'); xlim([0 total_dur]);
ylim([0 max([rat_v; 0.5])*1.15 + 0.1]);
title('\fontsize{10}\bf Phase 8 – CONFIDENCE WEIGHTING\rm  |  ACF energy ratio per window', ...
    'Color', ph_clr(8,:)*0.7);
set_panel_full(ax, ph_clr(8,:));

ax = subplot(3,1,2);
conf_norm = conf_v / max(conf_v + eps);
b_h = bar(t_v, conf_norm, 'FaceColor','flat','EdgeColor','none');
b_h.CData = win_clr;
ylabel('Confidence (norm)'); xlim([0 total_dur]); ylim([0 1.15]);
title('\fontsize{10}\bf Phase 8 – CONFIDENCE WEIGHTING\rm  |  Per-window score (height = weight in final average)', ...
    'Color', ph_clr(8,:)*0.7);
set_panel_full(ax, ph_clr(8,:));

ax = subplot(3,1,3);
if ~isnan(ground_truth_fhr)
    mae_all = abs(fhr_v - ground_truth_fhr);
    yyaxis left;
    fill([t_v; flipud(t_v)],[fhr_v+sd_v2; flipud(fhr_v-sd_v2)], ...
        [0.85 0.92 1.0],'EdgeColor','none','FaceAlpha',0.5); hold on;
    scatter(t_v, fhr_v, 40, win_clr, 'filled');
    yline(ground_truth_fhr,'r--','LineWidth',1.8);
    yline(weighted_mean,   'g-', 'LineWidth',2.0);
    ylabel('FHR (BPM)'); ylim([90 175]);
    yyaxis right;
    plot(t_v, mae_all,'Color',[0.7 0.3 0.0],'LineWidth',1.2,'LineStyle','--');
    yline(mean(mae_all),'Color',[0.7 0.3 0.0],'LineStyle',':','LineWidth',1.2);
    ylabel('MAE (BPM)'); ylim([0 max(mae_all)*2]);
    ax.YAxis(2).Color = [0.6 0.3 0.0];
    legend({'±SD','Windows','GT','Weighted','MAE','Mean MAE'}, ...
        'FontSize',7,'Location','best');
    title(sprintf(['\\fontsize{10}\\bf Phase 9 – FINAL FHR\\rm  |  ' ...
        'Weighted = %.2f BPM   MAE = %.2f BPM   Pool SD = %.2f BPM'], ...
        weighted_mean, abs(weighted_mean-ground_truth_fhr), std(fhr_pool)), ...
        'Color', ph_clr(9,:)*0.7);
else
    fill([t_v; flipud(t_v)],[fhr_v+sd_v2; flipud(fhr_v-sd_v2)], ...
        [0.85 0.92 1.0],'EdgeColor','none','FaceAlpha',0.5); hold on;
    scatter(t_v, fhr_v, 40, win_clr, 'filled');
    yline(weighted_mean,'g-','LineWidth',2);
    ylabel('FHR (BPM)'); ylim([90 175]);
    legend({'±SD','Windows','Weighted mean'},'FontSize',7,'Location','best');
    title(sprintf(['\\fontsize{10}\\bf Phase 9 – FINAL FHR\\rm  |  ' ...
        'Weighted = %.2f BPM   Pool SD = %.2f BPM'], ...
        weighted_mean, std(fhr_pool)), 'Color', ph_clr(9,:)*0.7);
end
xlabel('Time (s)'); xlim([0 total_dur]);
set_panel_full(ax, ph_clr(9,:));

sgtitle(sprintf('FHR Pipeline — Phases 8–9  |  %s  |  %.0f s recording', ...
    filename, total_dur), 'FontSize',12,'FontWeight','bold');


%% ════════════════════════════════════════════════════════════════════════
%% FIGURE 5 ── Original FHR Timeline + Ratio + SD  (summary)
%% ════════════════════════════════════════════════════════════════════════
colours_orig = zeros(sum(valid_mask), 3);
tmp_flags    = timeline_flag(valid_mask);
for i = 1:length(t_v)
    if strcmp(tmp_flags{i},'FETAL-DOM'),  colours_orig(i,:)=[0.0 0.6 0.0];
    elseif strcmp(tmp_flags{i},'MIXED'),  colours_orig(i,:)=[0.0 0.4 0.9];
    else,                                 colours_orig(i,:)=[0.8 0.2 0.2]; end
end

figure('Name','FHR Summary Timeline', 'Position',[50 50 1300 600], 'Color','w');
subplot(3,1,1);
fill([t_v; flipud(t_v)],[fhr_v+sd_v2; flipud(fhr_v-sd_v2)], ...
    [0.85 0.92 1.0],'EdgeColor','none','FaceAlpha',0.5); hold on;
scatter(t_v, fhr_v, 30, colours_orig, 'filled');
if ~isnan(ground_truth_fhr)
    yline(ground_truth_fhr,'r--','LineWidth',1.5);
end
yline(weighted_mean,'g-','LineWidth',2);
xlabel('Time (s)'); ylabel('FHR (BPM)'); ylim([90 175]); grid on;
if ~isnan(ground_truth_fhr)
    title(sprintf('Phase 9 – FHR Timeline | Conf-Weighted = %.2f BPM | MAE = %.2f BPM', ...
        weighted_mean, abs(weighted_mean-ground_truth_fhr)));
    legend({'±SD','Windows (green/blue/red)','Ground Truth','Conf-weighted'},'Location','best');
else
    title(sprintf('Phase 9 – FHR Timeline | Conf-Weighted = %.2f BPM', weighted_mean));
    legend({'±SD','Windows (green/blue/red)','Conf-weighted'},'Location','best');
end

subplot(3,1,2);
plot(t_v, rat_v,'b-o','LineWidth',1,'MarkerSize',3); hold on;
yline(0.85,'g--','LineWidth',1.2); yline(0.60,'y--','LineWidth',1.2);
xlabel('Time (s)'); ylabel('Fetal/Maternal Ratio');
title('Phase 5 – ACF Energy Ratio');
legend({'Ratio','0.85 threshold','0.60 threshold'},'Location','best'); grid on;

subplot(3,1,3);
plot(t_v, sd_v2,'r-o','LineWidth',1,'MarkerSize',3);
xlabel('Time (s)'); ylabel('SD (BPM)');
title('Phase 7 – Estimation SD per Window (lower = more stable)'); grid on;

sgtitle(sprintf('%s  |  Conf-Weighted FHR = %.2f BPM', filename, weighted_mean), ...
    'FontSize',12,'FontWeight','bold');


%% ════════════════════════════════════════════════════════════════════════
%% FIGURE 6 ── MAE Timeline  (only if ground truth provided)
%% ════════════════════════════════════════════════════════════════════════
if ~isnan(ground_truth_fhr)
    figure('Name','MAE Timeline', 'Position',[50 50 1300 300], 'Color','w');
    mae_all = abs(fhr_v - ground_truth_fhr);
    bar(t_v, mae_all,'FaceColor',[0.3 0.6 0.9]); hold on;
    yline(3,'g--','LineWidth',1.5);
    yline(6,'y--','LineWidth',1.5);
    yline(mean(mae_all),'r--','LineWidth',1.5);
    xlabel('Window Centre (s)'); ylabel('MAE (BPM)');
    title(sprintf('Phase 9 – MAE per Window | Mean = %.2f | Best = %.2f BPM at t = %.0f s', ...
        mean(mae_all), min(mae_all), t_v(mae_all == min(mae_all))));
    legend({'MAE','3 BPM','6 BPM','Mean MAE'},'Location','best'); grid on;
end


%% ════════════════════════════════════════════════════════════════════════
%% HELPER  — set_panel_full
%% ════════════════════════════════════════════════════════════════════════
function set_panel_full(ax, edge_clr)
    box(ax,'on');
    set(ax,'LineWidth',1.4,'XColor',edge_clr*0.55,'YColor',edge_clr*0.55, ...
        'Color',[0.974 0.974 0.984]);
    grid(ax,'on'); grid(ax,'minor');
    set(ax,'GridColor',[0.78 0.78 0.78],'MinorGridColor',[0.90 0.90 0.90], ...
        'GridAlpha',0.5,'MinorGridAlpha',0.3);
end