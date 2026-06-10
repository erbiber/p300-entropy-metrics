\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
   
import os
import sys
import numpy as np
import pandas as pd
import mne

from config import (
    DATA_ROOT, RESULTS_DIR, FILES,
    FILTER_LOW, FILTER_HIGH, FILTER_DESIGN,
    BASELINE, REJECT_THRESHOLD,
    DO_ICA, ICA_N_COMPONENTS, ICA_HIGHPASS_FOR_FIT, ICA_METHOD, ICA_RANDOM_STATE,
    TMIN_EPOCH, TMAX_EPOCH,
    EARLY_WINDOW, EARLY_WINDOW_200, EARLY_WINDOW_250,
    P300_WINDOW, BASELINE_CONTROL_WINDOW,
    EARLY_CHANNEL, P300_CHANNEL, EARLY_FALLBACK, P300_FALLBACK,
    STANDARD_CODES, TARGET_CODES,
    MIN_TRIALS_REQUIRED,
    get_channel_index, banner,
)

mne.set_log_level('WARNING')

                                                                             
def feature_block(trace):
                                                                     
    mean_amp = float(np.mean(trace))
    sd_amp = float(np.std(trace - mean_amp, ddof=0))
    rms_amp = float(np.sqrt(np.mean(trace ** 2)))
    return mean_amp, sd_amp, rms_amp

def window_mask(times, win):
    return (times >= win[0]) & (times <= win[1])

                                                                             
def process_subject(sub_id, log_rows):
\
\
\
       
    set_path = os.path.join(
        DATA_ROOT, f"sub-{sub_id}", "ses-P3", "eeg",
        f"sub-{sub_id}_ses-P3_task-P3_eeg.set",
    )
    if not os.path.exists(set_path):
        return None, None, None, "file_not_found"

                                 
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)
    raw.filter(FILTER_LOW, FILTER_HIGH, fir_design=FILTER_DESIGN, verbose=False)

    sfreq = raw.info['sfreq']
    n_samples_continuous = raw.n_times

                     
    n_ica_components = 0
    n_ica_excluded = 0
    if DO_ICA:
        try:
            raw_for_ica = raw.copy().filter(
                ICA_HIGHPASS_FOR_FIT, None,
                fir_design=FILTER_DESIGN, verbose=False)
            ica = mne.preprocessing.ICA(
                n_components=ICA_N_COMPONENTS,
                method=ICA_METHOD,
                random_state=ICA_RANDOM_STATE,
                max_iter='auto',
            )
            ica.fit(raw_for_ica, verbose=False)
            n_ica_components = ica.n_components_
            try:
                eog_idx, _ = ica.find_bads_eog(raw, verbose=False)
                ica.exclude = eog_idx
                n_ica_excluded = len(eog_idx)
            except Exception:
                ica.exclude = []
            raw = ica.apply(raw, verbose=False)
        except Exception as e:
            print(f"  [sub-{sub_id}] ICA failed, continuing without: {e}")

                        
    events, _ = mne.events_from_annotations(raw, verbose=False)

                                      
    epochs = mne.Epochs(
        raw, events, event_id=None,
        tmin=TMIN_EPOCH, tmax=TMAX_EPOCH,
        baseline=BASELINE,
        reject=dict(eeg=REJECT_THRESHOLD),
        preload=True, verbose=False,
    )
    if len(epochs) < MIN_TRIALS_REQUIRED:
        return None, None, None, f"insufficient_trials_{len(epochs)}"

    epoch_codes = epochs.events[:, 2]
    valid = np.isin(epoch_codes, STANDARD_CODES + TARGET_CODES)
    epochs = epochs[valid]
    if len(epochs) < MIN_TRIALS_REQUIRED:
        return None, None, None, f"insufficient_valid_trials_{len(epochs)}"

    data = epochs.get_data()                                     
    times = epochs.times
    ch_names = epochs.ch_names

    fz_name, fz_idx = get_channel_index(ch_names, EARLY_CHANNEL, EARLY_FALLBACK)
    pz_name, pz_idx = get_channel_index(ch_names, P300_CHANNEL, P300_FALLBACK)
    if fz_name is None or pz_name is None:
        return None, None, None, f"missing_channels_fz={fz_name}_pz={pz_name}"

                            
    m_early = window_mask(times, EARLY_WINDOW)
    m_early200 = window_mask(times, EARLY_WINDOW_200)
    m_early250 = window_mask(times, EARLY_WINDOW_250)
    m_p300 = window_mask(times, P300_WINDOW)
    m_basectrl = window_mask(times, BASELINE_CONTROL_WINDOW)

    n_trials, n_chan, _ = data.shape

                                     
    rows = []
    for i in range(n_trials):
        code = epochs.events[i, 2]
        cond = "Target" if code in TARGET_CODES else "Standard"
        fz_trace = data[i, fz_idx, :]
        pz_trace = data[i, pz_idx, :]

        m_fz, sd_fz, rms_fz = feature_block(fz_trace[m_early])
        m_fz200, sd_fz200, rms_fz200 = feature_block(fz_trace[m_early200])
        m_fz250, sd_fz250, rms_fz250 = feature_block(fz_trace[m_early250])
        m_basef, sd_basef, rms_basef = feature_block(fz_trace[m_basectrl])
        m_pz_e, sd_pz_e, rms_pz_e = feature_block(pz_trace[m_early])
        m_pz_b, sd_pz_b, rms_pz_b = feature_block(pz_trace[m_basectrl])
        p300_amp = float(np.mean(pz_trace[m_p300]))

        rows.append({
                     : f'sub-{sub_id}',
                       : int(i),
                       : cond,
                       : int(cond == "Target"),
                                 
                           : m_fz,
                         : sd_fz,
                          : rms_fz,
                                    
                               : m_fz200,
                             : sd_fz200,
                              : rms_fz200,
                               : m_fz250,
                             : sd_fz250,
                              : rms_fz250,
                                 
                          : m_basef,
                        : sd_basef,
                         : rms_basef,
                                 
                           : m_pz_e,
                         : sd_pz_e,
                          : rms_pz_e,
                                 
                          : m_pz_b,
                        : sd_pz_b,
                         : rms_pz_b,
                          
                      : p300_amp,
                        : fz_name,
                        : pz_name,
                        : n_chan,
        })
    df_trial = pd.DataFrame(rows)

                                                         
    early_rms_all = np.sqrt(np.mean(data[:, :, m_early] ** 2, axis=2))
    df_elec = pd.DataFrame(
        early_rms_all, columns=[f'rms_{c}' for c in ch_names])
    df_elec.insert(0, 'subject', f'sub-{sub_id}')
    df_elec.insert(1, 'trial_idx', np.arange(n_trials))
    df_elec.insert(2, 'condition',
                   ['Target' if c in TARGET_CODES else 'Standard'
                    for c in epochs.events[:, 2]])
    df_elec['p300_amp'] = df_trial['p300_amp'].values

                                                      
    is_t = df_trial['is_target'].values.astype(bool)
    ga = {
               : times,
                  : np.array(ch_names),
                       : data[is_t, pz_idx, :].mean(axis=0),
                         : data[~is_t, pz_idx, :].mean(axis=0),
                       : data[is_t, fz_idx, :].mean(axis=0),
                         : data[~is_t, fz_idx, :].mean(axis=0),
                          : data[is_t][:, :, m_p300].mean(axis=(0, 2)),
                            : data[~is_t][:, :, m_p300].mean(axis=(0, 2)),
                           : data[is_t][:, :, m_early].mean(axis=(0, 2)),
                             : data[~is_t][:, :, m_early].mean(axis=(0, 2)),
    }

                                 
    log_rows.append({
                 : f'sub-{sub_id}',
                  : sfreq,
                            : n_samples_continuous,
                           : len(df_trial),
                  : int(df_trial['is_target'].sum()),
                    : int((1 - df_trial['is_target']).sum()),
                         : fz_name,
                         : pz_name,
                    : n_chan,
                          : n_ica_components,
                        : n_ica_excluded,
    })
    return df_trial, df_elec, ga, None

                                                                             
if __name__ == "__main__":
    banner("01_extract_features.py — feature extraction from real epochs")
    print(f"Data:     {DATA_ROOT}")
    print(f"Results:  {RESULTS_DIR}")
    print(f"Filter:   {FILTER_LOW}-{FILTER_HIGH} Hz")
    print(f"Reject:   +/-{REJECT_THRESHOLD*1e6:.0f} uV")
    print(f"Baseline: {BASELINE}")
    print(f"Early:    {EARLY_WINDOW} + sensitivity {EARLY_WINDOW_200}, {EARLY_WINDOW_250}")
    print(f"P300:     {P300_WINDOW} at {P300_CHANNEL}")
    print(f"ICA:      {DO_ICA}\n")

    if not os.path.isdir(DATA_ROOT):
        print(f"ERROR: data root not found: {DATA_ROOT}")
        sys.exit(1)

    sub_dirs = sorted(d for d in os.listdir(DATA_ROOT) if d.startswith('sub-'))
    sub_ids = [d.split('-')[1] for d in sub_dirs]
    print(f"Found {len(sub_ids)} subjects.\n")

    all_trials, all_elec, exclusions, gas = [], [], {}, {}
    log_rows = []

    for sid in sub_ids:
        print(f"  Processing sub-{sid} ...")
        df_t, df_e, ga, err = process_subject(sid, log_rows)
        if df_t is None:
            exclusions[f'sub-{sid}'] = err
            print(f"    excluded ({err})")
        else:
            all_trials.append(df_t)
            all_elec.append(df_e)
            gas[f'sub-{sid}'] = ga
            n_t = int(df_t['is_target'].sum())
            print(f"    ok  ({len(df_t)} trials, {n_t} targets)")

    if not all_trials:
        print("No subjects processed."); sys.exit(1)

    df_all = pd.concat(all_trials, ignore_index=True)
    df_elec_all = pd.concat(all_elec, ignore_index=True)

    out1 = os.path.join(RESULTS_DIR, FILES['trial_features'])
    df_all.to_csv(out1, index=False)
    out2 = os.path.join(RESULTS_DIR, FILES['trial_features_per_elec'])
    df_elec_all.to_csv(out2, index=False)

    np.savez_compressed(
        os.path.join(RESULTS_DIR, FILES['figure1_data']),
        subjects=np.array(list(gas.keys())),
        gas=np.array([gas[k] for k in gas], dtype=object),
    )

          
    if exclusions:
        pd.DataFrame(
            [{'subject': s, 'reason': r} for s, r in exclusions.items()]
        ).to_csv(os.path.join(RESULTS_DIR, FILES['exclusions']), index=False)
    pd.DataFrame(log_rows).to_csv(
        os.path.join(RESULTS_DIR, FILES['preprocessing_log']), index=False)

    n_sub = df_all['subject'].nunique()
    n_trials = len(df_all)
    n_target = int((df_all['condition'] == 'Target').sum())
    n_std = int((df_all['condition'] == 'Standard').sum())

    print()
    banner(f"DONE: {n_sub} subjects, {n_trials} trials")
    print(f"  Targets: {n_target} ({100*n_target/n_trials:.1f}%)   "
          f"Standards: {n_std} ({100*n_std/n_trials:.1f}%)")
    print(f"  Mean trials/subject: {n_trials/n_sub:.1f}")
    print(f"  Trial features:        {out1}")
    print(f"  Per-electrode RMS:     {out2}")
    if exclusions:
        print(f"\n  Excluded {len(exclusions)} subjects (see logs/subject_exclusions.csv):")
        for s, r in exclusions.items():
            print(f"    {s}: {r}")
