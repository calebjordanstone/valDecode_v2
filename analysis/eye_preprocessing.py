import polars as pl
import mne
import numpy as np 
from pathlib import Path
import os
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
%matplotlib qt

## set data paths
DATAPATH = "C:/Users/cstone/OneDrive - UNSW/Documents/Projects/my_experiments/val_decode_v2/data/sourcedata" 
SAVEPATH = "C:/Users/cstone/OneDrive - UNSW/Documents/Projects/my_experiments/val_decode_v2/data/preprocessed/"
srcDataEEG = sorted(Path(DATAPATH).glob('**/*.bdf'))

# I loop through the EEG data paths here just for convenience; could use anything that enables you to get the subIDs you need
for path in srcDataEEG[0:1]:

    # extract some info
    subID, task, modality = path.stem.split('_')

    # load eye tracking data 
    eye = pl.concat([pl.read_csv(i)  
            for i in sorted(Path(DATAPATH).glob(f'**/{subID}*.csv'))],
            how='vertical_relaxed').sort(by='block', descending=False)
    
    # create event dictionaries
    event_dict = {
            "hi/Co/to/le/iti": 10,
            "hi/Co/to/le/cue": 11,
            "hi/Co/to/le/tar": 13,
            "hi/Co/to/le/fdbc": 15,
            "hi/Co/to/le/fdbi": 17,
            "hi/Co/to/le/fdbm": 19,

            "hi/Co/to/ri/iti": 20,
            "hi/Co/to/ri/cue": 21,
            "hi/Co/to/ri/tar": 23,
            "hi/Co/to/ri/fdbc": 25,
            "hi/Co/to/ri/fdbi": 27,
            "hi/Co/to/ri/fdbm": 29,

            "hi/Co/aw/le/iti": 30,
            "hi/Co/aw/le/cue": 31,
            "hi/Co/aw/le/tar": 33,
            "hi/Co/aw/le/fdbc": 35,
            "hi/Co/aw/le/fdbi": 37,
            "hi/Co/aw/le/fdbm": 39,

            "hi/Co/aw/ri/iti": 40,
            "hi/Co/aw/ri/cue": 41,
            "hi/Co/aw/ri/tar": 43,
            "hi/Co/aw/ri/fdbc": 45,
            "hi/Co/aw/ri/fdbi": 47,
            "hi/Co/aw/ri/fdbm": 49,

            "hi/Ct/to/le/iti": 50,
            "hi/Ct/to/le/cue": 51,
            "hi/Ct/to/le/tar": 53,
            "hi/Ct/to/le/fdbc": 55,
            "hi/Ct/to/le/fdbi": 57,
            "hi/Ct/to/le/fdbm": 59,

            "hi/Ct/to/ri/iti": 60,
            "hi/Ct/to/ri/cue": 61,
            "hi/Ct/to/ri/tar": 63,
            "hi/Ct/to/ri/fdbc": 65,
            "hi/Ct/to/ri/fdbi": 67,
            "hi/Ct/to/ri/fdbm": 69,

            "hi/Ct/aw/le/iti": 70,
            "hi/Ct/aw/le/cue": 71,
            "hi/Ct/aw/le/tar": 73,
            "hi/Ct/aw/le/fdbc": 75,
            "hi/Ct/aw/le/fdbi": 77,
            "hi/Ct/aw/le/fdbm": 79,

            "hi/Ct/aw/ri/iti": 80,
            "hi/Ct/aw/ri/cue": 81,
            "hi/Ct/aw/ri/tar": 83,
            "hi/Ct/aw/ri/fdbc": 85,
            "hi/Ct/aw/ri/fdbi": 87,
            "hi/Ct/aw/ri/fdbm": 89,

            "lo/Co/to/le/iti": 90,
            "lo/Co/to/le/cue": 91,
            "lo/Co/to/le/tar": 93,
            "lo/Co/to/le/fdbc": 95,
            "lo/Co/to/le/fdbi": 97,
            "lo/Co/to/le/fdbm": 99,

            "lo/Co/to/ri/iti": 100,
            "lo/Co/to/ri/cue": 101,
            "lo/Co/to/ri/tar": 103,
            "lo/Co/to/ri/fdbc": 105,
            "lo/Co/to/ri/fdbi": 107,
            "lo/Co/to/ri/fdbm": 109,

            "lo/Co/aw/le/iti": 110,
            "lo/Co/aw/le/cue": 111,
            "lo/Co/aw/le/tar": 113,
            "lo/Co/aw/le/fdbc": 115,
            "lo/Co/aw/le/fdbi": 117,
            "lo/Co/aw/le/fdbm": 119,

            "lo/Co/aw/ri/iti": 120,
            "lo/Co/aw/ri/cue": 121,
            "lo/Co/aw/ri/tar": 123,
            "lo/Co/aw/ri/fdbc": 125,
            "lo/Co/aw/ri/fdbi": 127,
            "lo/Co/aw/ri/fdbm": 129,

            "lo/Ct/to/le/iti": 130,
            "lo/Ct/to/le/cue": 131,
            "lo/Ct/to/le/tar": 133,
            "lo/Ct/to/le/fdbc": 135,
            "lo/Ct/to/le/fdbi": 137,
            "lo/Ct/to/le/fdbm": 139,

            "lo/Ct/to/ri/iti": 140,
            "lo/Ct/to/ri/cue": 141,
            "lo/Ct/to/ri/tar": 143,
            "lo/Ct/to/ri/fdbc": 145,
            "lo/Ct/to/ri/fdbi": 147,
            "lo/Ct/to/ri/fdbm": 149,

            "lo/Ct/aw/le/iti": 150,
            "lo/Ct/aw/le/cue": 151,
            "lo/Ct/aw/le/tar": 153,
            "lo/Ct/aw/le/fdbc": 155,
            "lo/Ct/aw/le/fdbi": 157,
            "lo/Ct/aw/le/fdbm": 159,

            "lo/Ct/aw/ri/iti": 160,
            "lo/Ct/aw/ri/cue": 161,
            "lo/Ct/aw/ri/tar": 163,
            "lo/Ct/aw/ri/fdbc": 165,
            "lo/Ct/aw/ri/fdbi": 167,
            "lo/Ct/aw/ri/fdbm": 169,      
        }
    
    event_dict_cue = {
        "hi/Co/to/le/cue": 11,
        "hi/Co/to/ri/cue": 21,
        "hi/Co/aw/le/cue": 31,
        "hi/Co/aw/ri/cue": 41,
        "hi/Ct/to/le/cue": 51,
        "hi/Ct/to/ri/cue": 61,
        "hi/Ct/aw/le/cue": 71,
        "hi/Ct/aw/ri/cue": 81,
        "lo/Co/to/le/cue": 91,
        "lo/Co/to/ri/cue": 101,
        "lo/Co/aw/le/cue": 111,
        "lo/Co/aw/ri/cue": 121,
        "lo/Ct/to/le/cue": 131,
        "lo/Ct/to/ri/cue": 141,
        "lo/Ct/aw/le/cue": 151,
        "lo/Ct/aw/ri/cue": 161,  
        }

    ### find bad epochs based on eye tracking data -----------------------------------------------------------------
    # load trigger mapping
    trig_map = pl.read_csv('eye_tracker_trigger_map.csv')
    def convert_triggers(trig_map, event):
        trig = trig_map.filter(pl.col('event_eyetracker') == event
                            ).select(pl.col('trigger_stimpc')
                                        ).to_numpy()[0, 0]
        return trig

    # convert eyetracker event codes
    eye_events = eye.select(pl.col('event')).to_series().to_numpy()
    # this is very slow, could write better code to make it more efficient. 
    # Also not sure what my plan was if any of the trig values are NaNs. You will need to double-check this in 
    # your own data
    eye_trigs = [convert_triggers(trig_map, i) for i in eye_events[~np.isnan(eye_events)]] 
    eye = eye.with_columns(pl.Series(name="triggers", values=eye_trigs)) 

    # create mne epochs object from raw eye data
    eye_data = np.swapaxes(eye.select(['L_X', 'L_Y', 'R_X', 'R_Y', 'triggers']).to_numpy(), 0, 1)
    eye_times = eye.select(['time_gd'])
    eye_ch_names = ['L_X', 'L_Y', 'R_X', 'R_Y','Status']
    eye_sfreq = 600 
    eye_ch_types = ['eyegaze', 'eyegaze', 'eyegaze', 'eyegaze', 'stim']
    eye_info = mne.create_info(ch_names=eye_ch_names, sfreq=eye_sfreq, ch_types=eye_ch_types)
    raw_eye = mne.io.RawArray(eye_data, eye_info)

    # find eye events
    eye_events = mne.find_events(raw_eye, 
        initial_event=True, 
        consecutive=True, 
        output='onset', 
        shortest_event=1)
    for key, val in event_dict.items():
        ev_mask = eye_events[:, 2] == val
        count = ev_mask.sum()
        print(f'{key}: {count}')

    # create eye epochs
    eye_epochs_cue = mne.Epochs(
        raw=raw_eye,
        events=eye_events,
        event_id=event_dict_cue,
        tmin=0,
        tmax=1,
        baseline=None,
        preload=True, 
        reject=None, 
        flat=None,
        reject_by_annotation=False,
        on_missing='warn')
    #eye_epochs_plh = eye_epochs['plh']
    
    # extract eye data and find missing responses
    eye_data_plh = eye_epochs_cue.get_data(picks='eyegaze')
    missing_eye_data = np.isnan(eye_data_plh)
    # cycle through each epoch and channel to find consecutive (> 2) missing data points; 
    # identify the index of the start and end of these missing periods;
    # then add another 15 samples (25 ms) either side to account for artifacts in the 
    # eye position data. (s_per_sample = 1/raw_eye.info['sfreq'] # seconds per time sample; s_per_sample*15 = 0.025)
    for e, epoch in enumerate(missing_eye_data): # cycle through epochs
        for c, chan in enumerate(epoch): # cycle through channels in each epoch
            onsets = [] # create empy lists to store onsets and offsets of missing data (i.e., blinks)
            offsets = []
            for i, j in enumerate(chan):
                # if not a missing data point, continue    
                if not j: 
                     continue
                else:
                    # if the first data point is missing, it can only be an onset, so only check for subsequent missing data
                    if i == 0: 
                        if chan[i+1]:
                            onsets.append(i)
                    # if the last data point is missing, it can only be an offset, so only check for antecedent missing data       
                    elif i == (len(chan) - 1): 
                        if chan[i-1]:
                             offsets.append(i)
                    # for all data points that aren't the first or last, check for onsets and offsets of blinks
                    else: 
                        if (not chan[i-1]) & (chan[i+1]):
                            onsets.append(i)
                        elif (chan[i-1]) & (not chan[i+1]):
                            offsets.append(i)
                        elif (chan[i-1]) & (chan[i+1]):
                            continue
            # only keep sequences of > 2 consecutive missing data points
            keeps = (np.array(onsets) - np.array(offsets)) < -2 
            # set the onset to 15 samples prior to actual onset
            onsets = np.array(onsets)[keeps]-15 
            # set the offset to 15 samples post the actual offset
            offsets = np.array(offsets)[keeps]+15 
            # if the onset is negative, set it to 0
            if np.any(np.array(onsets) < 0): 
                np.put_along_axis(onsets, np.where(onsets < 0)[0], 0, axis=0)
            # if the offset is greater than n_samples, set it to = n_samples
            if np.any(np.array(offsets) > (len(chan) - 1)): 
                np.put_along_axis(offsets, np.where(offsets > (len(chan) - 1))[0], (len(chan) - 1), axis=0)
            # update missing data array with new values
            for k in range(len(onsets)): 
                missing_eye_data[e, c, onsets[k]:offsets[k]] = True

    # now exclude the missing data
    # first define ROI: 60 pixels = ~2 cm or 1.77 DVA at 65 cm. 
    # first define ROI: 90 pixels = ~3 cm or 2.7 DVA at 65 cm.
    # first define ROI: 102 pixels = ~3.4 cm or 3 DVA at 65 cm
    # NOTE: half the normal resolution for PPixx
    x_min, x_max = 1920/2 - 51, 1920/2 + 51  # half the screen resolution to find centre; ± half of ROI to get square ROI around fixation
    y_min, y_max = 1080/2 - 51, 1080/2 + 51 

    eye_dict = dict()
    for i, _ in enumerate(eye_data_plh):

        # combine missing data for both eyes
        nan_idxs = np.any([missing_eye_data[i, 0], missing_eye_data[i, 2]], axis=0)  
        eye_data = eye_data_plh[i, :, :][:, ~nan_idxs]
        
        # exclude trial if there isn't enough eye data 
        if eye_data.shape[-1] < 0.50*eye_data_plh.shape[-1]:
            exclude_samples = 1
        else: 
            exclude_samples = 0 

        # find deviation trials
        if exclude_samples == 1:
            exclude_dev = 0

        else: 
            # average over left and right eyes and smooth data
            eye_data = np.stack([np.mean([eye_data[0], eye_data[2]], axis=0),
                                np.mean([eye_data[1], eye_data[3]], axis=0)])
            eye_data[1, :] = 1080 - eye_data[1, :] # invert y-axis coordinates because the data is recorded with top as 0 and bottom as 1
            eye_data = savgol_filter(eye_data, window_length=10, polyorder=1)

            # find horizontal fixations and saccades
            grad = np.gradient(eye_data[0]) 
            fix_idxs = (grad > -1) & (grad < 1) # NOTE: I picked 1/-1 as my criterion gradient after a bit of trial and error, but it's based on nothing more than that
            eye_data_fix = np.full(len(eye_data[0]), np.nan)
            eye_data_fix[fix_idxs] = eye_data[0][fix_idxs]
            dev_x = np.any([(eye_data_fix > x_max), (eye_data_fix < x_min)], axis=0)

            # exclude trials that in which the deviation exceeds 50 ms, ignoring first 50 ms
            counts = []
            count = 0
            # run on x coordinate data
            for j, k in enumerate(dev_x):
                if k: # if deviation sample, add 1 count; if last sample, add total to counts list
                    count += 1
                    if j == len(dev_x) - 1:
                        counts.append(count)
                else: 
                    if count > 0: # if there has been deviation samples previously, add total to counts list
                        counts.append(count)
                    count = 0 # reset count total
            
            if np.any(np.array(counts) > 30): # if any deviation periods greater than 50 ms (30 samples), exclude trial
                exclude_dev = 1
            else: 
                exclude_dev = 0

            # plot fixations
            # fig, ax = plt.subplots()
            # ax.hlines(y=x_min, xmin=0, xmax=600, linestyles='--', color='grey')
            # ax.hlines(y=x_min + 51, xmin=0, xmax=600, linestyles='--', color='black')
            # ax.hlines(y=x_max, xmin=0, xmax=600, linestyles='--', color='grey')
            # ax.plot(eye_data[0], label = 'saccade')
            # ax.plot(eye_data_fix, label = 'fixation')
            # plt.title(f'exclude_dev: {exclude_dev}, exclude_samples: {exclude_samples}')
            # plt.legend()

        eye_dict[f'Trial_{i}'] = {
            'exclude_samples': exclude_samples,
            'exclude_dev': exclude_dev,
            #'eye': eye,
            'data_all': eye_data_plh[i],
            'data_eye': eye_data
        }

    # plot eye data
    # for trial in eye_dict.keys():
    #     fig, ax = plt.subplots()
    #     ax.hlines(y=x_min, xmin=0, xmax=600, linestyles='--', color='grey', label='X')
    #     ax.hlines(y=x_max, xmin=0, xmax=600, linestyles='--', color='grey', label='X')
    #     ax.hlines(y=x_min + 51, xmin=0, xmax=600, linestyles='--', color='black')
    #     ax.hlines(y=y_max, xmin=0, xmax=600, linestyles='-.', color='grey')
    #     ax.hlines(y=y_min, xmin=0, xmax=600, linestyles='-.', color='grey', label='Y')
    #     ax.hlines(y=y_min + 51, xmin=0, xmax=600, linestyles='-.', color='black')
    #     ax.plot(eye_dict[trial]['data_eye'][0], color = 'blue', label = 'cleaned')
    #     ax.plot(eye_dict[trial]['data_eye'][1], color = 'blue')
    #     plt.legend()

    # find all bad eye epochs
    bad_eye_epochs = []
    n_missing_data = 0
    n_dev = 0
    for i, trial in enumerate(eye_dict.keys()):
        if eye_dict[trial]['exclude_samples'] == 1:
            n_missing_data += 1
            bad_eye_epochs.append(i)
        if eye_dict[trial]['exclude_dev'] == 1:
            n_dev += 1
            bad_eye_epochs.append(i)
    print(f"# Missing data trials: {n_missing_data}")
    print(f'# Deviation trial: {n_dev}')
    print(f"# Excluded trials: {len(bad_eye_epochs)}")

    # save indices of bad eye epochs
    if not os.path.exists(SAVEPATH + subID):
        os.mkdir(SAVEPATH + subID)  
    with open(SAVEPATH + subID + f"/{subID}_bad_eye_epochs.txt", "w") as output:
        output.write(str(bad_eye_epochs))

