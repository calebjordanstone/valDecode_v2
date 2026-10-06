## Notes: This script takes preprocessed, epoched EEG data and raw behavioural data and performs the following:
# 1) Drops bad epochs and those that are excluded in the behavioural and eye analysis, equalises the event counts 
# per trial/epoch type, and creates averages of 4 epochs of the same trial/epoch type to increase SNR in the decoding analyses
# 2) Multi-class decoding to extract decision-functions per trial,  time point, and subject
# 3) Binary decoding separaetly for high- and low-value trials per time point and subject
## Input data can be found in 
## Output data can be found in 

import mne
import re
import polars as pl
import numpy as np 
import scipy as sp 
from mne.decoding import SlidingEstimator, cross_val_multiscore
from sklearn import svm, model_selection
from pathlib import Path
%matplotlib qt

# load data
SOURCEPATH = "C:/Users/cstone/OneDrive - UNSW/Documents/Projects/my_experiments/val_decode_v2/data/sourcedata/"
EPOCHPATH = "C:/Users/cstone/OneDrive - UNSW/Documents/Projects/my_experiments/val_decode_v2/data/preprocessed/"
SAVEPATH = "C:/Users/cstone/OneDrive - UNSW/Documents/Projects/my_experiments/val_decode_v2/data/derivatives/"
extension = 'cue'
epoch_paths = sorted(Path(EPOCHPATH).glob(f'sub*/*{extension}.fif')) 

## set constants for analysis
freqs = np.geomspace(2, 35, 30)
n_cycles = 3 
analysis_window = [-0.1, 1] 
# define bands
delta = freqs < 4
theta = (freqs >= 4) & (freqs < 8)
alpha = (freqs >= 8) & (freqs < 13) 
beta = (freqs >= 13) & (freqs < 30)
gamma = freqs >= 30
# create classifier
svc = svm.LinearSVC(max_iter=2000) 
temp_decode = SlidingEstimator(svc, scoring="accuracy") # extend pipeline over time
skf = model_selection.StratifiedKFold(n_splits=4, shuffle=True) # define cross-validation
# create column names for output DF
cols = [f'dfun_{trig}' 
        for trig in np.arange(11, 171, 10)] # cue trigger values from 11 to 171

## loop through participants for analysis
for path in epoch_paths[19:]:

    ## load, clean and subset data
    subID = path.stem.split('_')[0]
    # exclude subjects with lots of missing/deviation eye trials
    if subID in ['sub-03', 'sub-11', 'sub-12', 
                 'sub-13', 'sub-16', 'sub-24', 
                 'sub-30', 'sub-36']: 
        continue

    epochs = mne.read_epochs(path) 
    # load behavioural data  
    beh_path = sorted(Path(SOURCEPATH).glob(f'**/{subID}*_beh.txt')) 
    beh = pl.read_csv(beh_path[0], separator="\t")
    #beh = beh_all.filter(pl.col("Subject") == subID)
    # reset trial number index
    beh = beh.with_columns((pl.col("RunningTrialNo") - 1).alias("RunningTrialNo2")) 
    # extract trial number of trials to exlcude
    filt_idx = beh.filter((pl.col('Accuracy') != 1) | 
                          (pl.col('RT') < 0.15)
                          ).select('RunningTrialNo2').to_series().to_list()
    # load bad epochs txt file
    bad_epochs_path = sorted(Path(EPOCHPATH).glob(f'**/{subID}*bad*epochs.txt')) 
    # get bad eeg epochs
    with open(bad_epochs_path[0], 'r') as file:
        eeg_bads_str = file.readlines()
    eeg_bads = re.findall(r"-?\d*\.?\d+", eeg_bads_str[0])
    eeg_bads = list(map(int, eeg_bads)) # convert to list of ints
    # get bad eye epochs
    with open(bad_epochs_path[1], 'r') as file:
        eye_bads_str = file.readlines()
    eye_bads = re.findall(r"-?\d*\.?\d+", eye_bads_str[0])
    eye_bads = list(map(int, eye_bads)) # convert to list of ints
    # combine bad epochs with list of trials to exclude, remove duplicates
    exclude = list(set(filt_idx + eye_bads + eeg_bads)) 
    # drop epochs 
    epochs.drop(exclude)
    # remove corresponding epochs from the behavioural data file
    beh_cleaned = beh.filter(~pl.col('RunningTrialNo2').is_in(exclude))
    # add new column to behavioural data file with new index
    beh_cleaned = beh_cleaned.with_columns(pl.Series('NewTrialIndex', np.arange(0, len(beh_cleaned))))
    # equalise event counts
    epochs, idx_drpd = epochs.equalize_event_counts(method='random', random_state=int('1234' + subID.split('-')[1])) 
    # remove the equalised event count dropped epochs from the behavioural data
    beh_cleaned = beh_cleaned.with_columns(pl.Series('NewTrialIndex', np.arange(0, len(beh_cleaned)))
                                           ).filter(~pl.col("NewTrialIndex").is_in(idx_drpd))
    # add final reset index 
    beh_cleaned = beh_cleaned.with_columns(pl.Series('NewTrialIndex', np.arange(0, len(beh_cleaned))))
    # add new columns to behavioural datafile for later
    beh_cleaned = beh_cleaned.with_columns(event_type=pl.lit(999), 
                                           event_group=pl.lit(999))
    # create averaged epochs
    epochs_av = epochs.copy()
    n_trials = 4 # number of trials to average
    epochs_av.selection = np.arange(0, len(epochs_av)) # renumber epochs to start at 0
    eeg_events_array = [] # create empty list for new events array
    epoch_array = [] # create empty list to store averaged epochs
    df_epoch_array = pl.DataFrame(schema={"event_type": float,
                                          "event_group": float,
                                          "RT": float})
    for event_type in epochs_av.event_id.keys():
        idxs = epochs_av[event_type].selection # find indicies of epochs that belong to the event type
        beh_cleaned[idxs, 'event_type'] = epochs_av.event_id[event_type]
        n_epochs = int(np.ceil(len(idxs) / n_trials))
        for epoch in range(n_epochs):
            eeg_events_array.append(epochs_av.event_id[event_type]) # add correct number of events to events array
        group = 0
        while len(idxs) > n_trials: # loop through epochs to extract as many averages as we can 
            this_selection = np.random.choice(idxs, 
                                              size=n_trials, 
                                              replace=False) # radomly select epochs of the same type to average
            av_epoch = epochs_av[this_selection].average(method='mean').get_data() # average epochs
            epoch_array.append(av_epoch[np.newaxis, :]) # save to list
            bool_array = list(map(lambda x: x not in this_selection, idxs)) # update idxs list to remove the epochs we just averaged together
            idxs = idxs[bool_array]
            # update behavioural datafiles
            beh_cleaned[this_selection, 'event_group'] = group
            df_epoch_array = pl.concat([df_epoch_array,
                                        beh_cleaned[this_selection].mean()['event_type', 'event_group', "RT"]
                                        ])
            group += 1
        # average remaining epochs
        av_epoch = epochs_av[idxs].average(method='mean').get_data() # average remaining epochs together
        epoch_array.append(av_epoch[np.newaxis, :])
        beh_cleaned[idxs, 'event_group'] = group
        df_epoch_array = pl.concat([df_epoch_array,
                            beh_cleaned[idxs].mean()['event_type', 'event_group', "RT"]
                            ])
    epoch_array = np.concatenate(epoch_array, axis=0)
    ## create dimensions for new events array for new info object for averaged epochs
    # create 1st dimension: event onset, by adding length of each epoch in ms * n_epochs (technically should be samples not ms)
    dim1 = np.linspace(0, 
                    (np.abs(epochs_av.tmin) + epochs_av.tmax)*1000*len(epoch_array), 
                    len(eeg_events_array), 
                    endpoint=False, 
                    dtype=int)
    # create 2nd dimension: signal value of the immediately preceding sample. Just setting to 0 for all
    dim2 = np.zeros(len(epoch_array))
    # cretae ful levent array by stacking dim1 and dim2 with eeg_events_array, which contains the event codes
    eeg_events_array = np.stack([dim1, dim2, eeg_events_array], 
                                axis=1)
    # create new info object
    info = mne.create_info(epochs_av.info.ch_names[0:64], 
                           epochs_av.info['sfreq'], 
                           ch_types='eeg')
    # create new epoch array for averaged epochs
    epoch_grp = mne.EpochsArray(data=epoch_array, 
                                info=info, 
                                events=eeg_events_array.astype(int), 
                                tmin=epochs_av.tmin, 
                                event_id=epochs_av.event_id)

    # save cleaned behavioural data files
    beh_cleaned.write_csv(SAVEPATH + f'beh_{subID}.csv')

    ## begin decoding analysis 
    # compute tfr
    power = epoch_grp.compute_tfr(
        method="morlet",
        freqs=freqs,
        n_cycles=n_cycles,
        average=False)

    # average over frequency bands
    power_data = power.get_data() 
    times = (power.times >= analysis_window[0]) & (power.times <= analysis_window[1])
    power_data = power_data[..., times] # trim data to times of interest
    power_ind_freqs = [power_data[:, :, band, :].mean(2) 
                        for band in [delta, theta, alpha, beta, gamma]] # average over all freqncies within a band
    
    # normalise data
    z_scrd_lst = []
    for band in power_ind_freqs:
        z_scrd = [sp.stats.zscore(band[:, :, t], axis=1) 
                    for t in range(0, band.shape[2])] # z-score across electodes for each epoch and time point, separately for each freqency band
        z_scrd_lst.append(np.moveaxis(np.array(z_scrd), 0, -1))
    
    # get data
    X = np.concatenate(z_scrd_lst, axis=1)
    # get indicies of classes
    power.selection = np.arange(0, len(power))
    y = power.events[:, 2]

    # Run multiclass decoding -------------------------------------------------------
    # create empty data frame to save results
    df_decfun = pl.DataFrame(schema={"dfun_11": float, 
                                     "dfun_21": float,
                                     "dfun_31": float, 
                                     "dfun_41": float,
                                     "dfun_51": float, 
                                     "dfun_61": float,
                                     "dfun_71": float, 
                                     "dfun_81": float,
                                     "dfun_91": float,
                                     "dfun_101": float,
                                     "dfun_111": float,
                                     "dfun_121": float,
                                     "dfun_131": float,
                                     "dfun_141": float,
                                     "dfun_151": float,
                                     "dfun_161": float,
                                     "event_type": float,
                                     "event_group": float,
                                     "RT": float,
                                     "y": int,
                                     "tpoint":int,
                                     "score":float,
                                     "subID":str,
                                     "cv_split":float})
    
    # run model
    for split in np.arange(1, 9):
        # generate new cv folds
        splits = list(skf.split(X, y))
        for train, test in splits:
            # run per time sample
            for t in range(X.shape[-1]):
                # fit model
                svc.fit(X[:, :, t][train], y[train])
                score = svc.score(X[:, :, t][test], y[test])
                dfun = svc.decision_function(X[:, :, t][test]) 
                # put results into dataframe
                df = pl.DataFrame(dfun, schema=cols)
                df = df.with_columns([(pl.Series(df_epoch_array[test]['event_type']).alias('event_type')),
                                      (pl.Series(df_epoch_array[test]['event_group']).alias('event_group')),
                                      (pl.Series(df_epoch_array[test]['RT']).alias('RT')),
                                      (pl.Series(y[test], dtype=int).alias('y')),
                                      (pl.lit(t, dtype=int).alias('tpoint')),
                                      (pl.lit(score, dtype=float).alias('score')),
                                      (pl.lit(subID, dtype=str).alias('subID')),
                                      (pl.lit(split, dtype=float).alias('cv_split'))])
                # save output to main dataframe
                df_decfun = pl.concat([df_decfun, df])

    # save csv 
    df_decfun = df_decfun.group_by(["event_type", "event_group", "y", "tpoint", "subID"]).mean() # average over 8 iterations of CV
    df_decfun.write_csv(SAVEPATH + f'dfun_{extension}_{subID}.csv')

