import polars as pl
import mne
import numpy as np 
import scipy as sp 
import os
from pathlib import Path
from eeg_fasterAlgorithm import *
import matplotlib.pyplot as plt
%matplotlib qt

## set data paths
DATAPATH = "C:/Users/cstone/OneDrive - UNSW/Documents/Projects/my_experiments/val_decode_v2/data/sourcedata" 
SAVEPATH = "C:/Users/cstone/OneDrive - UNSW/Documents/Projects/my_experiments/val_decode_v2/data/preprocessed/"
srcDataEEG = sorted(Path(DATAPATH).glob('**/*.bdf'))
# srcDataBeh = sorted(Path(DATAPATH).glob('**/*beh.txt'))
# srcDataFrms = sorted(Path(DATAPATH).glob('**/*frms.txt'))
# srcDataEye= sorted(Path(DATAPATH).glob('**/*eye_block*.csv'))

for path in srcDataEEG[3:9]:

    # extract some BIDS info
    subID, task, modality = path.stem.split('_')

    # load data
    raw = mne.io.read_raw_bdf(
        path,
        preload=True)
    
    # load behavioural data file
    # beh_path =  sorted(Path(DATAPATH).glob(f'**/{subID}*beh.txt')) 
    # beh = pl.read_csv(beh_path[0], separator='\t')
    # eye_path = sorted(Path(DATAPATH).glob(f'**/{subID}*.csv'))
    # eye = pl.concat([pl.read_csv(i)  
    #             for i in sorted(Path(DATAPATH).glob(f'**/{subID}*.csv'))])    
    # eye = pl.read_csv(eye_path[0], separator=',')

    # set montage
    raw.set_montage(
        'biosemi64', 
        on_missing='warn')

    # add EOG channels
    info = mne.create_info( 
        ch_names=['hEOG','vEOG'], 
        sfreq=raw.info['sfreq'], 
        ch_types='eog'
        )
    RH, LH, LV, UV = mne.pick_channels(
        raw.ch_names, 
        ['EXG5', 'EXG6', 'EXG7', 'EXG8']
        )
    hEOG = raw[LH][0] - raw[RH][0]
    vEOG = raw[UV][0] - raw[LV][0]
    newEOG = mne.io.RawArray(
        np.concatenate([hEOG, vEOG]), 
        info=info)
    raw.add_channels(
        add_list=[newEOG], 
        force_update_info=True)
    raw.drop_channels(
        ch_names=['EXG1', 'EXG2', 'EXG3', 'EXG4', 
                'EXG5', 'EXG6', 'EXG7', 'EXG8'])

    # find events
    eeg_events = mne.find_events(raw, 
        initial_event=True, 
        consecutive=True, 
        output='onset', 
        shortest_event=1)
    
    # # modify events to eliminate errors 
    # for i, j in enumerate(eeg_events[:, 2]): 
    #     if j in [150, 190, 250, 290, 350, 390, 450, 490, 550, 590, 650, 690, 750, 790, 850, 890]: # errors
    #         # if (eeg_events[i - 4, 2] in [110, 210, 310, 410, 510, 610, 710, 810]):
    #         #     eeg_events[i - 4, 2] = 999 #error
    #         if (eeg_events[i - 3, 2] in [110, 210, 310, 410, 510, 610, 710, 810]):
    #             eeg_events[i - 3, 2] = 999 #error or  miss

    # # check this worked
    # sum(eeg_events[:, 2] == 999) == len(idx) 

    # create event dictionary
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
    
    # plot events
    # mne.viz.plot_events(eeg_events, 
    #                     event_id=event_dict,
    #                     on_missing='warn',
    #                     sfreq=raw.info['sfreq'])

    # check event counts
    for key, val in event_dict.items():
        ev_mask = eeg_events[:, 2] == val
        count = ev_mask.sum()
        print(f'{key}: {count}')

    # mark breaks in data collection
    break_annots = mne.preprocessing.annotate_break(
        raw=raw,
        events=eeg_events,
        min_break_duration=5,  # consider segments of at least 5 s duration
        t_start_after_previous=2,  # start annotation 3 s after end of previous one
        t_stop_before_next=2,  # stop annotation 3 s before beginning of next one
    ) 
    raw.set_annotations(break_annots)

    # plot raw data
    # raw.plot()

    ## Start actual pre-processing ---------------------------------------------------------

    # filter EEG data
    filt_h = raw.copy().filter(
        l_freq=None, 
        h_freq=40, 
        picks='eeg')

    # identify bad chans
    filt_h.info['bads'] += faster_bad_channels(filt_h, 
                                            use_metrics=['correlation',
                                                         'variance',
                                                         'hurst'])

    # check filtered data to make sure no bad channels were missed and mark bad sections
    #filt_h.plot()

    # handle bad channels
    intrp = filt_h.copy().interpolate_bads()   

    # re-reference to average 
    avRef = intrp.set_eeg_reference()

    # create two copies of data, one for ICA and one for continued preprocessing
    ica_data = avRef.copy().filter(l_freq=1.0, h_freq=None) # use unfiltered data to apply higher high-pass filter
    filt_l = avRef.copy().filter(l_freq=0.1, h_freq=None, picks='eeg')

    # run ICA
    ICA = mne.preprocessing.ICA() 
    ICA.fit(ica_data, decim=8) # speed up processing

    # check ICA components on filtered data
    # ICA.plot_sources(filt_l, show_scrollbars=False)
    # ICA.plot_components()
    # automatically find the ICs that best match the EOG signal
    eog_indices, eog_scores = ICA.find_bads_eog(filt_l)
    print(f'EOG indicies: {eog_indices}')
    ICA.exclude = eog_indices

    # apply ICA 
    ICA.apply(filt_l)
    #filt_l.plot()

    # create eeg epochs
    tmin = -1
    tmax = 2.7
    baseline = (-0.2, 0)
    epochs_cue = mne.Epochs(
        raw=filt_l,
        events=eeg_events,
        event_id=event_dict_cue,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        decim=filt_l.info['sfreq']/256,
        preload=True, 
        reject=None, 
        flat=None,
        reject_by_annotation=False,
        on_missing='warn')
    #epochs_cue = epochs['cue']
    # epochs_trig.selection = np.arange(0, len(epochs))

    # remove bad epochs
    bads_epochs = faster_bad_epochs(epochs_cue)
    # epochs_trig.drop(bads_epochs)

    # # create averaged epochs
    # n_trials = 4 # number of trials to average 
    # epochs_av = epochs_trig.copy()
    # epochs_av.selection = np.arange(0, len(epochs_av)) # renumber epochs to start at 0
    # eeg_events_array = [] # create empty list for new events array
    # epoch_array = [] # create empty list to store averaged epochs
    # for event_type in epochs_av.event_id.keys():
    #     idxs = epochs_av[event_type].selection # find indicies of epochs that belong to the event type
    #     n_epochs = int(np.ceil(len(idxs) / n_trials))
    #     for epoch in range(n_epochs):
    #         eeg_events_array.append(event_dict[event_type]) # add correct number of events to events array
    #     while len(idxs) > n_trials: # loop through epochs to extract as many averages as we can 
    #         this_selection = np.random.choice(idxs, 
    #                                         size=n_trials, 
    #                                         replace=False) # radomly select epochs of the same type to average
    #         av_epoch = epochs_av[this_selection].average(method='mean').get_data() # average epochs
    #         epoch_array.append(av_epoch[np.newaxis, :]) # save to list
    #         bool_array = list(map(lambda x: x not in this_selection, idxs)) # update idxs list to remove the epochs we just averaged together
    #         idxs = idxs[bool_array]
    #     av_epoch = epochs_av[idxs].average(method='mean').get_data() # average remaining epochs together
    #     epoch_array.append(av_epoch[np.newaxis, :])
    # epoch_array = np.concatenate(epoch_array, axis=0)
    # dim1 = np.linspace(0, 
    #                 (np.abs(epochs_av.tmin) + epochs_av.tmax)*1000*len(epoch_array), 
    #                 len(eeg_events_array), 
    #                 endpoint=False, 
    #                 dtype=int)
    # dim2 = np.zeros(len(epoch_array))
    # eeg_events_array = np.stack([dim1, dim2, eeg_events_array], 
    #                             axis=1)
    # info = mne.create_info(epochs_av.info.ch_names[0:64], 
    #                         epochs_av.info['sfreq'], 
    #                         ch_types='eeg')
    # epochs_av = mne.EpochsArray(data=epoch_array, 
    #                             info=info, 
    #                             events=eeg_events_array.astype(int), 
    #                             tmin=epochs_av.tmin, 
    #                             event_id=epochs_av.event_id)

    # save epochs
    # epoch_dict = {
    # 'subID': subID,
    # 'av_trls': str('false'),
    # 'hi_to_le': len(epochs_trig['hi/to/le']),
    # 'hi_to_ri': len(epochs_trig['hi/to/ri']),
    # 'hi_aw_le': len(epochs_trig['hi/aw/le']),
    # 'hi_aw_ri': len(epochs_trig['hi/aw/ri']),
    # 'lo_to_le': len(epochs_trig['lo/to/le']),
    # 'lo_to_ri': len(epochs_trig['lo/to/ri']),
    # 'lo_aw_le': len(epochs_trig['lo/aw/le']),
    # 'lo_aw_ri': len(epochs_trig['lo/aw/ri']),
    # 'n_bd_chns': len(filt_h.info['bads'])}

    epoch_dict = {}
    epoch_dict['subID'] = subID
    for key, val in event_dict_cue.items():
        ev_mask = eeg_events[:, 2] == val
        count = ev_mask.sum()
        epoch_dict[f'{key.replace("/", "_")[0:-4]}'] = int(count) 
    epoch_dict['n_bd_chns'] = len(filt_h.info['bads'])

    df = pl.from_dict(epoch_dict)
    if not os.path.exists(SAVEPATH + f'epochs_cue.txt'):
        df.write_csv(SAVEPATH + f'epochs_cue.txt')
    else:
        df_ld = pl.read_csv(SAVEPATH + f'epochs_cue.txt', schema_overrides={'av_trls':pl.String})
        df_ld.vstack(df, in_place=True)
        df_ld.write_csv(SAVEPATH + f'epochs_cue.txt')
    # save cue epoch files
    if not os.path.exists(SAVEPATH + subID):
        os.mkdir(SAVEPATH + subID)
    FILENAME = f'/{subID}_{task}_{modality}_epochs-cue.fif'
    epochs_cue.save(SAVEPATH + subID + FILENAME)
    # save indices of bad eeg epochs
    with open(SAVEPATH + subID + f"/{subID}_bad_eeg_epochs.txt", "w") as output:
        output.write(str(bads_epochs))


    # # save averaged epochs
    # epoch_dict_av = {
    # 'subID': subID,
    # 'av_trls': str('true'),
    # 'hi_to_le': len(epochs_av['hi/to/le']),
    # 'hi_to_ri': len(epochs_av['hi/to/ri']),
    # 'hi_aw_le': len(epochs_av['hi/aw/le']),
    # 'hi_aw_ri': len(epochs_av['hi/aw/ri']),
    # 'lo_to_le': len(epochs_av['lo/to/le']),
    # 'lo_to_ri': len(epochs_av['lo/to/ri']),
    # 'lo_aw_le': len(epochs_av['lo/aw/le']),
    # 'lo_aw_ri': len(epochs_av['lo/aw/ri']),
    # 'n_bd_chns': len(filt_h.info['bads'])}
    # df = pl.from_dict(epoch_dict_av)
    # if not os.path.exists(SAVEPATH + f'epochs_{trigger}_{segment}_avg.txt'):
    #     df.write_csv(SAVEPATH + f'epochs_{trigger}_{segment}_avg.txt')
    # else:
    #     df_ld = pl.read_csv(SAVEPATH + f'epochs_{trigger}_{segment}_avg.txt', schema_overrides={'av_trls':pl.String})
    #     df_ld.vstack(df, in_place=True)
    #     df_ld.write_csv(SAVEPATH + f'epochs_{trigger}_{segment}_avg.txt')
    # # save cue epoch files
    # if not os.path.exists(SAVEPATH + subID):
    #     os.mkdir(SAVEPATH + subID)
    # FILENAME = f'/{subID}_{task}_{modality}_epochs_{trigger}_{segment}_avg.fif'
    # epochs_av.save(SAVEPATH + subID + FILENAME)
