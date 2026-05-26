from datetime import datetime
from psychopy import core, event, visual, logging, gui, event, monitors, data 
from pathlib import Path
from propixx_functions import *
# import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import os
import itertools
import json
# from pypixxlib import _libdpx as dp
logging.console.setLevel(logging.WARNING)

## ======================================================================
## Conenct to PROPixx
## ======================================================================
dp.DPxOpen()
isReady = dp.DPxIsReady()
if isReady:
    dp.DPxSetPPxDlpSeqPgrm('RGB')
    dp.DPxEnableDoutPixelMode() # enable pixel mode for triggers
    dp.DPxEnablePPxRearProjection() # enable rear projection to reverse display
    dp.DPxWriteRegCache()
else:
    print('Warning! DPx call failed, check connection to hardware')
    core.quit()
## ======================================================================
## Get experiment and monitor information 
## ======================================================================
expInfo = {
    'Monitor Name': 'test',
    'Monitor Refresh Rate': '120',
    'Subject Number': '',
    'Screen Number': '1',
    'Full Screen': True,
    'Demo': True,
    'Demographics': True,
    'Debrief': True,
    #'Instructions': True,
    'Eyetracking': True}

dlg = gui.DlgFromDict(
    expInfo, 
    title = 'Experinent Information')

if not dlg.OK:
    core.quit()

EXPERIMENT = 'valDecode_v2'
DATETIME = datetime.now().strftime("%y%m%d%H%M") 
SUBID = 'sub-' + f'{int(expInfo["Subject Number"])}'.zfill(2)
REFRATE = int(expInfo['Monitor Refresh Rate'])
SCREEN = int(expInfo['Screen Number'])
FULLSCREEN = expInfo['Full Screen']
DEMO = expInfo['Demo']
DEMOGRAPHICS = expInfo['Demographics']
DEBRIEF = expInfo['Debrief']
#INSTRUCTIONS = expInfo['Instructions']
EYETRACKING = expInfo['Eyetracking']

# configure monitor
if expInfo['Monitor Name'] in monitors.getAllMonitors():
    MONITOR = monitors.Monitor(expInfo['Monitor Name'])

else:
    monInfo = {
        'Monitor y (pxl)': '1080',
        'Monitor x (pxl)': '1920',
        'Monitor Width (cm)': '',
        'Monitor Distance (cm)': ''}

    dlg = gui.DlgFromDict(
        monInfo, 
        title = 'Monitor Information')

    if dlg.OK:
        MONITOR = monitors.Monitor(expInfo['Monitor Name'])
        MONITOR.setDistance(int(monInfo['Monitor Distance (cm)']))
        MONITOR.setWidth(float(monInfo['Monitor Width (cm)']))
        MONITOR.setSizePix((int(monInfo['Monitor x (pxl)']), 
                            int(monInfo['Monitor y (pxl)'])))
        MONITOR.saveMon()
    else:
        gui.criticalDlg('Something went wrong. Aborting')
        core.quit()

# collect demographics
if DEMOGRAPHICS:

    demographics = {
        'What is your age? (Leave blank if you would rather not say)': '',
        'How do you describe your sex?': ['Woman/Female', 'Man/Male', 'Prefer not to answer', 'Intersex', 'I use a different term']}

    if expInfo['Demographics']:

        dlg = gui.DlgFromDict(
            demographics, 
            title = 'Demographics and Screening')
        if not dlg.OK:
            core.quit()

    AGE = int(demographics['How old are you?'])
    SEX = demographics['How do you describe your sex?']

## ======================================================================
## Create data paths
## ======================================================================
# DATAPATH = 'C:/Users/cstone/OneDrive - UNSW/Documents/Projects/my_experiments/val_decode/data/'
DATAPATH = '/home/experimenter/Experiments/val_decode_v2/data/'
os.mkdir(DATAPATH + SUBID)
folders = ['/beh/', '/eeg/']
for folder in folders:
    os.mkdir(DATAPATH + SUBID + folder)
FILENAME = f'{SUBID}_task-{EXPERIMENT}_beh.txt'
LOGFILENAME = f'{SUBID}_task-{EXPERIMENT}_log.txt'
FRMSFILENAME = f'{SUBID}_task-{EXPERIMENT}_frms.txt'
FILEPATH = DATAPATH + SUBID + folders[0] + FILENAME
LOGFILEPATH = DATAPATH + SUBID + folders[0] + LOGFILENAME
FRMSFILEPATH = DATAPATH + SUBID + folders[0] + FRMSFILENAME
logging.LogFile(LOGFILEPATH)
## ======================================================================
## Create window, stimuli, and other experiment features
## ======================================================================
# create window
win = visual.Window(
        screen = SCREEN, 
        monitor = MONITOR,
        size = MONITOR.getSizePix(),
        fullscr=FULLSCREEN,
        pos = [0, 0],
        color = [0, 0, 0], # black
        units = 'pix', 
        colorSpace = 'rgb255',
        blendMode = 'avg')
win.refreshThreshold = 1/REFRATE + 0.002

# set mouse to invisible
mouse = event.Mouse(visible=False)

# create clock
clock = core.Clock()

# define stimulus positions
size_params = { # common parameters used to calculate stimulus size in pixels
    'distance': MONITOR.getDistance(), 
    'screen_res': MONITOR.getSizePix(), 
    'screen_width': MONITOR.getWidth()
}
y_pos = dva_to_pix(2, **size_params) # stmuli to appear 2 degrees down from centre
x_pos = dva_to_pix(5, **size_params) # stmuli to appear 5 degrees to left/right of centre
CENTRE = [0, 0]
LEFT = [-x_pos, -y_pos]
RIGHT = [x_pos, -y_pos]

# define position for trigger stimulus
TLC = [-win.size[0]/2, win.size[1]/2] # top left corner

# define positions for calibration
CALIB_POINT_1 = [-(win.size[0]/10)*2.5, (win.size[1]/10)*2.5]
CALIB_POINT_2 = [(win.size[0]/10)*2.5, (win.size[1]/10)*2.5]
CALIB_POINT_3 = [0, 0]
CALIB_POINT_4 = [-(win.size[0]/10)*2.5, -(win.size[1]/10)*2.5]
CALIB_POINT_5 = [(win.size[0]/10)*2.5, -(win.size[1]/10)*2.5]
CALIB_POINTS = [CALIB_POINT_1,
                CALIB_POINT_2,
                CALIB_POINT_3,
                CALIB_POINT_4,
                CALIB_POINT_5]

# define colours
BLUE =   [37, 141, 165] 
ORANGE = [194, 99, 32]
GREEN = [88, 129, 56]
PURPLE = [233, 84, 233]
WHITE = [255, 255, 255]
GREY = [119, 119, 119]

# Aassign high and low value to different colours
colour_assign_lists = [
    [BLUE, ORANGE, GREEN, PURPLE],
    [BLUE, GREEN, ORANGE, PURPLE],
    [BLUE, PURPLE, GREEN, ORANGE],
    [GREEN, PURPLE, BLUE, ORANGE],
    [GREEN, ORANGE, BLUE, PURPLE],
    [ORANGE, PURPLE, BLUE, GREEN]]

sub_col_assign = (int(expInfo["Subject Number"]) -1) % 6
HIGH_1, HIGH_2, LOW_1, LOW_2 = colour_assign_lists[sub_col_assign]

if sub_col_assign == 0:
    sub_col_assign_high = ['BLUE', 'ORANGE']
    sub_col_assign_low = ['GREEN', 'PURPLE']
elif sub_col_assign == 1:
    sub_col_assign_high = ['BLUE', 'GREEN']
    sub_col_assign_low = ['ORANGE', 'PURPLE']
elif sub_col_assign == 2:
    sub_col_assign_high = ['BLUE', 'PURPLE']
    sub_col_assign_low = ['GREEN',  'ORANGE']
elif sub_col_assign == 3:
    sub_col_assign_high = ['GREEN', 'PURPLE']
    sub_col_assign_low = ['BLUE', 'ORANGE']
elif sub_col_assign == 4:
    sub_col_assign_high = ['GREEN', 'ORANGE']
    sub_col_assign_low = ['BLUE', 'PURPLE']
elif sub_col_assign == 5:
    sub_col_assign_high = ['ORANGE', 'PURPLE']
    sub_col_assign_low = ['BLUE', 'GREEN']

# define rules
CON = 'T' # respond toward the stimulus
INCON = 'A' # respond away from the stimulus

# define keys===========================================================================
RESPKEYS = ['c', 'm']
QUITKEYS = ['escape']
CONKEYS = ['space']
EXITKEYS = ['return']

# define number of trials/blocks
NTRIALS = 64
NBLOCKS = 22

# define points boost
POINTS_HIGH = 50
POINTS_LOW = 5

# define function to convert RT into points
min_rt = 0.2
max_rt = 1
def points_function(rt):

    if rt <= min_rt:
        coef = 10
    elif (rt > min_rt) & (rt <= max_rt):
        coef = (1 - (rt - 0.2)*1.25) * 10
    elif rt > max_rt:
        coef = 0

    if coef < 0:
        coef = 0

    return coef

# calculate total possible points
max_total_points = ((NBLOCKS*(NTRIALS/2))*(POINTS_HIGH*points_function(min_rt)) +
                   (NBLOCKS*(NTRIALS/2))*(POINTS_LOW*points_function(min_rt)))

total_reward = 1500 # $15, or 1500 cents
cents_per_point = total_reward / max_total_points

# define timing (multiply all by 4 to account for increase in refrate from PROPixx)
ITI_RANGE = [int(0.5*REFRATE), int(0.7*REFRATE)] 
CUE_DUR = int(0.3*REFRATE)
SOA_DUR = int(0.2*REFRATE)
TAR_DUR = int(1*REFRATE)
FDB_DUR = int(0.4*REFRATE)
FDB_DUR_LNG = int(1.5*REFRATE)

# create stimuli
stim_params = { # common parameters used across stimuli
    'win': win, 
    'units': 'pix', 
    'opacity': 1,
    'contrast': 1,
    'colorSpace': 'rgb255'}
crc_stim_rad = dva_to_pix(dva=1.5, **size_params)
dis_stim = visual.Circle(
    radius = crc_stim_rad,
    edges = 100,
    lineWidth = 0,
    **stim_params)
tar_stim = visual.Circle(
    radius=crc_stim_rad,
    edges=100,
    lineWidth=0,
    color=GREY,
    **stim_params)
txt_stim_height = dva_to_pix(dva=1, **size_params)
txt_stim = visual.TextStim(
    pos=CENTRE,
    color=WHITE,
    height=txt_stim_height,
    wrapWidth=MONITOR.getSizePix()[0]*0.8, # 80% of width of screen
    **stim_params)
fix_stim_height = dva_to_pix(dva=1, **size_params)
fix_stim = visual.TextStim(
    text= "+",
    pos=CENTRE,
    color=WHITE,
    height=fix_stim_height,
    **stim_params)
cue_stim_height = dva_to_pix(dva=1.5, **size_params)
cue_stim = visual.TextStim(
    pos=CENTRE,
    color=WHITE,
    height=cue_stim_height,
    **stim_params)
trig_stim = visual.Line(
    pos=TLC,
    start=TLC,
    end=[TLC[0]+1, TLC[1]],
    interpolate = False,
    **stim_params)

# create stims for eye-tracker calibration
# calib_array = visual.ElementArrayStim(
#     win=win,
#     units='pix',
#     colorSpace = 'rgb255',
#     colors=WHITE,
#     nElements=4,
#     elementTex=None,
#     elementMask='circle',
#     #xys=CALIB_POINTS_QUAD,
#     sizes=crc_stim_rad/6)

calib_stim = visual.Circle(
    #radius = crc_stim_rad/3,
    edges = 100,
    color=WHITE,
    lineWidth = 0,
    **stim_params)

# create oscillation for stimuli
framerate = REFRATE
a = 0.5 # ampltiude
p = 0 # phase
t_calib = np.tile(np.linspace(0, 0.5, framerate, endpoint=False), 300) 
calib_size = a*(np.sin(2*np.pi*0.75*t_calib + 90)) + 0.6

# create experiment handler to save output
experimentDict = {
    'Experiment': EXPERIMENT,
    'Date': DATETIME,
    'Refrate': REFRATE,
    'MonitorSize' : MONITOR.getSizePix(),
    'Subject': SUBID,
    'SubHighValColAssign': sub_col_assign_high,
    'SubLowValColAssign': sub_col_assign_low
    }

exp = data.ExperimentHandler(name=EXPERIMENT,
                             extraInfo=experimentDict)
exp.dataNames = []                     

# write instruction text
instructText = {

'INSTRUCTIONS_1': '''
    Welcome to our experiment! \n\n\n\
    On every trial, you will see a letter followed by a grey circle and a coloured circle presented side by side. \n
    Depending on the letter and the location of the GREY circle, you will need to press either the "C" or the "M" key. \n\n\n
    Press space to continue. 
    ''',

'INSTRUCTIONS_2': '''
    If you see the letter "T", press the response key that is on the SAME side as the GREY circle.\n
    "T" stands for responding "toward" the grey circle.\n
    So, if you see the letter "T", and the grey circle is on the LEFT side of the screen, you would press the "C" key.\n
    But if you see the letter "T", and the grey circle is on the RIGHT side, you would press the "M" key. \n\n\n
    Press space to continue. 
    ''',

'INSTRUCTIONS_3': '''
    If you see the letter "A", press the response key that is on the OPPOSITE side to the GREY circle.\n
    "A" stands for responding "away" from the grey circle. \n
    So, if you see the letter "A", and the grey circle is on the LEFT side of the screen, you would press the "M" key. \n
    But if you see the letter "A", and the grey circle is on the RIGHT side, you would press the "C" key. \n\n\n
    Press space to continue. 
    ''',

'INSTRUCTIONS_4': f'''
    The faster you respond, the more points you can earn! You can earn up to 10 points per trial based on your response speed. \n
    However, if you make a mistake you won't earn any points on that trial, so you need to be accurate too. \n
    The coloured circle will indicate the POINTS BOOST for that trial. \n
    If the coloured circle is {sub_col_assign_high[0]} or {sub_col_assign_high[1]}, your points will be multiplied by {POINTS_HIGH}. \n
    If the coloured circle is {sub_col_assign_low[0]} or {sub_col_assign_low[1]}, your points will be multiplied by {POINTS_LOW}. \n
    These points will be converted into additional reimbursement at the end of the experiment. \n\n\n
    Press space to continue.
    '''
} 
## ======================================================================
## Initialise PROPixx, adjust stimulus parameters, define triggers
## ======================================================================
# # establish connection to hardware
# dp.DPxOpen()
# isReady = dp.DPxIsReady()
# if isReady:
#     dp.DPxSetPPxDlpSeqPgrm('RGB')
#     dp.DPxEnableDoutPixelMode() # enable pixel mode for triggers
#     dp.DPxEnablePPxRearProjection() # enable rear projection to reverse display
#     dp.DPxWriteRegCache()
# else:
#     print('Warning! DPx call failed, check connection to hardware')
#     core.quit()

# define trigger values
'''
Eye tracking:
254 = save data
255 = stop recording

Block number triggers:
200 + block number. e.g., 201 = block 1

Trial structure triggers:
Stim value: H = high, L = low
Colour: Co = Colour one, Ct = Colour two
Rule cue: T = toward (congruent), A = away (incongruent)
Target location: L = left, R = right
HCoTL: 1x
HCoTR: 2x
HCoAL: 3x
HCoAR: 4x
HCtTL: 5x
HCtTR: 6x
HCtAL: 7x
HCtAR: 8x
LCoTL: 9x
LCoTR: 10x
LCoAL: 11x
LCoAR: 12x
LCtTL: 13x
LCtTR: 14x
LCtAL: 15x
LCtAR: 16x

ITI: xx0 # inter-trial interval onset
CUE: xx1 # rule cue onsest
TAR: xx3 # target onset
RES: xx5 # correct response
RES: xx7 # incorrect response
RES: xx9 # missed response
'''

def find_trigger_prefix(trial_info):

    if (trial_info[0] == HIGH_1) & (trial_info[1] == CON) & (trial_info[2] == LEFT):
        pfx = 10
    elif (trial_info[0] == HIGH_1) & (trial_info[1] == CON) & (trial_info[2] == RIGHT):
        pfx = 20
    elif (trial_info[0] == HIGH_1) & (trial_info[1] == INCON) & (trial_info[2] == LEFT):
        pfx = 30
    elif (trial_info[0] == HIGH_1) & (trial_info[1] == INCON) & (trial_info[2] == RIGHT):
        pfx = 40
    elif (trial_info[0] == HIGH_2) & (trial_info[1] == CON) & (trial_info[2] == LEFT):
        pfx = 50
    elif (trial_info[0] == HIGH_2) & (trial_info[1] == CON) & (trial_info[2] == RIGHT):
        pfx = 60
    elif (trial_info[0] == HIGH_2) & (trial_info[1] == INCON) & (trial_info[2] == LEFT):
        pfx = 70
    elif (trial_info[0] == HIGH_2) & (trial_info[1] == INCON) & (trial_info[2] == RIGHT):
        pfx = 80
    elif (trial_info[0] == LOW_1) & (trial_info[1] == CON) & (trial_info[2] == LEFT):
        pfx = 90
    elif (trial_info[0] == LOW_1) & (trial_info[1] == CON) & (trial_info[2] == RIGHT):
        pfx = 100
    elif (trial_info[0] == LOW_1) & (trial_info[1] == INCON) & (trial_info[2] == LEFT):
        pfx = 110
    elif (trial_info[0] == LOW_1) & (trial_info[1] == INCON) & (trial_info[2] == RIGHT):
        pfx = 120
    elif (trial_info[0] == LOW_2) & (trial_info[1] == CON) & (trial_info[2] == LEFT):
        pfx = 130
    elif (trial_info[0] == LOW_2) & (trial_info[1] == CON) & (trial_info[2] == RIGHT):
        pfx = 140
    elif (trial_info[0] == LOW_2) & (trial_info[1] == INCON) & (trial_info[2] == LEFT):
        pfx = 150
    elif (trial_info[0] == LOW_2) & (trial_info[1] == INCON) & (trial_info[2] == RIGHT):
        pfx = 160

    return pfx
## ======================================================================
## Create trial structure
## ======================================================================
trl_stc = np.repeat(
    np.asarray(
        tuple(
            itertools.product(
                [HIGH_1, HIGH_2, LOW_1, LOW_2], # distractor value   
                [CON, INCON], # rule
                [LEFT, RIGHT])), # target location  
            dtype = 'object'), 
    repeats=4, # number of each trial type per block
    axis=0)
## ======================================================================
## Present stimuli
## ======================================================================
## Present instructions -------------------------------------------------------------- START INSTRUCTIONS
for txt in instructText.keys():

    event.clearEvents()
    logging.warning(f'{txt}')
    logging.flush()
    txt_stim.text = instructText[txt]
    instr_idx = 0
    while True:

        # draw stimuli
        txt_stim.draw()
        win.flip()

        # collect user input to exit 
        pressed = event.getKeys(keyList = CONKEYS)
        if pressed: 
            break

        # udpate frame
        instr_idx += 1 

## Start practice trials --------------------------------------------------------------START PRATICE
if DEMO:

    # create list of some stimulus featires to use later
    cols = [HIGH_1, HIGH_2, LOW_1, LOW_2]
    pos = [LEFT, RIGHT]
    cue = [CON, INCON]
    block = 0
    #accDict = {}

    while True: 

        # present instructions 
        txt_stim.text = f'This is a practice block. \n\n Press space to start.' 
        txt_idx = 0
        while True:

            # draw
            txt_stim.draw()
            win.flip()

            # collect user input to exit 
            pressed = event.getKeys(keyList = CONKEYS)
            if pressed:
                break

            # udpate frame
            txt_idx += 1
        
        # start presenting trials ---------------------------------------------------------- TRIAL ONSET
        N_TRIALS_PRAC = 20
        accList = [] # create list to store accuracy data for this demo block
        for trial in range(0, N_TRIALS_PRAC):
        
            # set stimulus properties for this trial 
            dis_stim.fillColor = cols[np.random.choice([0,1,2,3])]
            dis_stim.pos = pos[np.random.choice([0, 1])]
            if np.all(dis_stim.pos == LEFT):
                tar_stim.pos = RIGHT
            elif np.all(dis_stim.pos == RIGHT): 
                tar_stim.pos = LEFT 
            cue_stim.text = cue[np.random.choice([0, 1])]
            ITI_DUR = np.random.randint(ITI_RANGE[0], ITI_RANGE[1])
            
            # reset frame interval counting
            win.frameClock.reset()
            win.frameIntervals = []    
            
            # start stimulus presentation -------------------------------------------------- ITI ONSET
            for frame in range(0, ITI_DUR): 

                if frame == 0:
                    logging.warning('START_ISI')
                    logging.flush()

                # draw fixation
                fix_stim.draw()
                win.flip()
            
            for frame in range(0, CUE_DUR): # ----------------------------------------------- CUE ONSET

                if frame == 0:
                    logging.warning('START_CUE')
                    logging.flush()

                # draw cue
                cue_stim.draw()
                win.flip()

            for frame in range(0, SOA_DUR): # ------------------------------------------------ SOA ONSET

                if frame == 0:
                    logging.warning('START_SOA')
                    logging.flush()
                
                # draw fixation
                fix_stim.draw()
                win.flip()

            # reset things before target display
            event.clearEvents()
            clock.reset()
            for frame in range(0, TAR_DUR): # ------------------------------------------------ TAR ONSET

                if frame == 0:
                    logging.warning('START_TAR')
                    logging.flush()

                # draw stimuli
                fix_stim.draw()
                tar_stim.draw()
                dis_stim.draw()
                win.flip()

                # collect response input
                #end_practice = event.getKeys(keyList = EXITKEYS)
                quitPressed = event.getKeys(keyList = QUITKEYS)
                pressed = event.getKeys(keyList = RESPKEYS, timeStamped = clock)
                if quitPressed: # exit task 
                    dp.DPxClose() 
                    core.quit()
                elif pressed:
                    logging.warning('RESPONSE')
                    logging.flush()
                    response = pressed[0][0]
                    rt = pressed[0][1]
                    break

            # calculate correct response
            if cue_stim.text == CON:
                if np.all(tar_stim.pos == LEFT):
                    cor_response = 'c'
                elif np.all(tar_stim.pos == RIGHT):
                    cor_response = 'm'
            elif cue_stim.text == INCON:
                if np.all(tar_stim.pos == LEFT):
                    cor_response = 'm'
                elif np.all(tar_stim.pos == RIGHT):
                    cor_response = 'c'

            # calculate response accuracy and points if a response is made
            if pressed:  
                if cor_response == response:
                    acc = 1
                elif cor_response != response:
                    acc = 0
                # update points, tiggers, and feedback display 
                if acc == 0:
                    points = 0
                    if rt < min_rt:
                        txt_stim.text = 'Incorrect! \n No points. \n Slow down!'
                    else:
                        txt_stim.text = 'Incorrect! \n No points.'
                elif acc == 1:
                    if (np.all(dis_stim.fillColor == HIGH_1)) or  (np.all(dis_stim.fillColor == HIGH_2)): # high value reward
                        points = np.floor(points_function(rt)*POINTS_HIGH).astype(int)
                    elif (np.all(dis_stim.fillColor == LOW_1)) or  (np.all(dis_stim.fillColor == LOW_2)): # low value reward
                        points = np.floor(points_function(rt)*POINTS_LOW).astype(int)
                    txt_stim.text = f'+ {points} points!'
            # if no response made, assign missing values etc.     
            elif not pressed:
                response = 999
                rt = 999
                acc = np.nan
                points = 0
                txt_stim.text = 'Too slow! No points.'

            if acc != 1:
                fdb_dur = FDB_DUR_LNG
            elif acc == 1:
                fdb_dur = FDB_DUR
            for frame in range(0, fdb_dur): # --------------------------------------------------- FDB ONSET

                if frame == 0:
                    logging.warning('START_FDB')
                    logging.flush()

                # draw stimulus
                txt_stim.draw()
                win.flip()

            # store accuracy data
            accList.append(acc)

        # take a break and save data at the end of the block --------------------------------------------------
        block += 1
        #accDict[f'{block}'] = accList

        txt_stim.text = f'''
        End of Practice Block {block}. \n\n 
        You got {int((np.nansum(np.array(accList))/N_TRIALS_PRAC)*100)}% of trials correct. \n\n
        Press space to repeat, press enter to exit practice.''' 
        event.clearEvents()
        txt_idx = 0
        while True:

            # draw
            txt_stim.draw()
            win.flip()

            # collect user input to exit 
            pressed = event.getKeys(keyList = CONKEYS)
            end_practice = event.getKeys(keyList = EXITKEYS)
            if (pressed) or (end_practice):
                break

            # udpate frame
            txt_idx += 1 

        if end_practice:
            break

## Start eye tracking -----------------------------------------------------------------START EYE
event.clearEvents()
txt_stim.text = 'You will now start the main experiment.\n\nPlease wait for the experimenter to start the EEG recording.'
instr_idx = 0
while True:

    # draw stimuli
    txt_stim.draw()
    win.flip()

    # collect user input to exit 
    pressed = event.getKeys(keyList = EXITKEYS)
    if pressed: 
        break

    # udpate frame
    instr_idx += 1

if EYETRACKING:

    event.clearEvents()
    txt_stim.text = 'Start calibration'
    instr_idx = 0
    while True:

        # draw
        txt_stim.draw()
        win.flip()

        # collect user input to exit 
        begin_calibration = event.getKeys(keyList = CONKEYS)
        if begin_calibration:
            break

    # enter calibration
    while True:

        logging.warning('START_CALIBRATION')
        logging.flush()

        # Cycle through calibration points 
        for point, _ in enumerate(CALIB_POINTS):

            # set positions for current point
            calib_stim.pos = CALIB_POINTS[point]

            # present on screen
            event.clearEvents()
            calib_idx = 0 
            while True:

                # draw stims 
                calib_stim.size = (crc_stim_rad/2)*calib_size[calib_idx]
                calib_stim.draw()

                # flip window
                win.flip()

                #collect user input to exit 
                pressed = event.getKeys(keyList = CONKEYS)
                if pressed:
                    break

                # update idx
                calib_idx += 1
        
        event.clearEvents()
        txt_stim.text = 'Repeat calibration?'
        instr_idx = 0
        while True:

            # draw
            txt_stim.draw()
            win.flip()

            # collect user input to exit 
            continue_calibration = event.getKeys(keyList = CONKEYS)
            end_calibration = event.getKeys(keyList = EXITKEYS)
            if (continue_calibration) or (end_calibration):
                break

            # udpate frame
            instr_idx += 1 

        if end_calibration:
            break

    event.clearEvents()
    txt_stim.text = 'Start validation'
    instr_idx = 0
    while True:

        # draw
        txt_stim.draw()
        win.flip()

        # collect user input to exit 
        begin_validation = event.getKeys(keyList = CONKEYS)
        if begin_validation:
            break

        # udpate frame
        instr_idx += 1

    # enter validation
    while True: 

        logging.warning('START_VALIDATION')
        logging.flush()

        # Cycle through calibration points 
        for point, _ in enumerate(CALIB_POINTS):

            # set positions for current point
            calib_stim.pos = CALIB_POINTS[point]

            # present on screen
            event.clearEvents()
            calib_idx = 0 
            while True:

                # draw stims 
                calib_stim.size = (crc_stim_rad/2)*calib_size[calib_idx]
                calib_stim.draw()

                # flip window
                win.flip()

                #collect user input to exit 
                pressed = event.getKeys(keyList = CONKEYS)
                if pressed:
                    break

                # update idx
                calib_idx += 1
        
        event.clearEvents()
        txt_stim.text = 'Repeat validation?'
        instr_idx = 0
        while True:

            # draw
            txt_stim.draw()
            win.flip()

            # collect user input to exit 
            continue_validation = event.getKeys(keyList = CONKEYS)
            end_validation = event.getKeys(keyList = EXITKEYS)
            if (continue_validation) or (end_validation):
                break

            # udpate frame
            instr_idx += 1 

        if end_validation:
            break

## Start experiment ------------------------------------------------------------------ EXP ONSET
logging.warning('START_EXP')
logging.flush()
framesPerBlock = {}
runningTrialNo = 1
cumulative_points = 0
cumulative_cents = 0
for block in range(1, NBLOCKS + 1): #------------------------------------------------- BLOCK ONSET

    # present instructions 
    txt_stim.text = f'This is Block {block} of 22. \n\n Please keep your eyes on the fixation cross throughout each trial. \n\n Press space to begin.' 
    
    txt_idx = 0
    trg_rgb = dp.DPxTriggerToRGB(200 + block) # convert to RGB
    while True:

        # draw trigger stimulus
        if txt_idx < 2:
            trig_stim.lineColor = trg_rgb
            trig_stim.draw()

        # draw text stimulus
        txt_stim.draw()
        win.flip()

        # collect user input to exit 
        pressed = event.getKeys(keyList = CONKEYS)
        if pressed:
            break

        # udpate frame
        txt_idx += 1 
    
    # randomise trial order at start of each block 
    trl_strc = trl_stc.copy()
    while True:
        np.random.shuffle(trl_strc)
        repeats = 0
        switch = 0
        for i in range(0, len(trl_strc)-1):
            if trl_strc[i][1] == trl_strc[i+1][1]:
                repeats += 1
            elif trl_strc[i][1] != trl_strc[i+1][1]:
                switch += 1
        if np.isin(repeats - switch, [-1, 1]) == True:
            break
        else: 
            continue 
    
    # logging 
    logging.warning(f'START_BLOCK_{block}')
    logging.flush()
    # turn on recording of frame intervals
    win.recordFrameIntervals = True
    framesPerTrial = {}
    points_this_block = 0
    cents_this_block = 0
    # start presenting trials ---------------------------------------------------------- TRIAL ONSET
    for trial in range(0, NTRIALS):
        
        # send trial number trigger ----------------------------------------------------TRG ONSET
        # trg_rgb = dp.DPxTriggerToRGB(trial + 1)
        # # logging
        # logging.warning('TRL_TRG')
        # logging.flush()
        # for frame in range(2): 

        #     # draw stimulus
        #     trig_stim.lineColor = trg_rgb
        #     trig_stim.draw()
        #     win.flip()
        
        # set stimulus properties for this trial 
        dis_stim.fillColor = trl_strc[trial][0]
        tar_stim.pos= trl_strc[trial][2]
        if np.all(tar_stim.pos == LEFT):
            dis_stim.pos = RIGHT
        elif np.all(tar_stim.pos == RIGHT):
            dis_stim.pos = LEFT
        cue_stim.text = trl_strc[trial][1]
        ITI_DUR = np.random.randint(ITI_RANGE[0], ITI_RANGE[1])
        
        # reset frame interval counting
        win.frameClock.reset()
        win.frameIntervals = []

        # find starting trigger value for this trial
        trg_val = find_trigger_prefix(trl_strc[trial]) 
        trg_rgb = dp.DPxTriggerToRGB(trg_val) # convert to RGB

        # housekeeping
        event.clearEvents()
        logging.warning(f'START_TRIAL_{trial}')
        logging.flush()

        # start stimulus presentation -------------------------------------------------- ITI ONSET
        logging.warning('START_ISI')
        logging.flush()
        for frame in range(0, ITI_DUR): 

            # send trigger
            if frame < 2:
                trig_stim.lineColor = trg_rgb
                trig_stim.draw()

            # draw fixation
            fix_stim.draw()
            win.flip()
        
        # update trigger value
        trg_val += 1
        trg_rgb = dp.DPxTriggerToRGB(trg_val)
        logging.warning('START_CUE')
        logging.flush()
        for frame in range(0, CUE_DUR): # ----------------------------------------------- CUE ONSET

            # send trigger
            if frame < 2:
                trig_stim.lineColor = trg_rgb
                trig_stim.draw()

            # draw cue
            cue_stim.draw()
            win.flip()

        logging.warning('START_SOA')
        logging.flush()
        for frame in range(0, SOA_DUR): # ------------------------------------------------ SOA ONSET

            # draw fixation
            fix_stim.draw()
            win.flip()
        
        # reset things before target display
        event.clearEvents()
        clock.reset()
        # update trigger value
        trg_val += 2
        trg_rgb = dp.DPxTriggerToRGB(trg_val)
        logging.warning('START_TAR')
        logging.flush()
        for frame in range(0, TAR_DUR): # ------------------------------------------------ TAR ONSET

            # send trigger 
            if frame < 2:
                trig_stim.lineColor = trg_rgb
                trig_stim.draw()

            # draw stimuli
            fix_stim.draw()
            dis_stim.draw()
            tar_stim.draw()
            win.flip()

            # collect response input
            quitPressed = event.getKeys(keyList = QUITKEYS)
            pressed = event.getKeys(keyList = RESPKEYS, timeStamped = clock)
            if quitPressed: # exit task 
                dp.DPxClose() 
                core.quit()
            elif pressed:
                logging.warning('RESPONSE')
                logging.flush()
                response = pressed[0][0]
                rt = pressed[0][1]
                break

        # calculate correct response
        if trl_strc[trial][1] == CON:
            if trl_strc[trial][2] == LEFT:
                cor_response = 'c'
            elif trl_strc[trial][2] == RIGHT:
                cor_response = 'm'
        elif trl_strc[trial][1] == INCON:
            if trl_strc[trial][2] == LEFT:
                cor_response = 'm'
            elif trl_strc[trial][2] == RIGHT:
                cor_response = 'c'

        # calculate response accuracy and points if a response is made
        if pressed:  
            if cor_response == response:
                acc = 1
            elif cor_response != response:
                acc = 0
            # update points, tiggers, and feedback display 
            if acc == 0:
                points = 0
                trg_val += 4
                if rt < min_rt:
                    txt_stim.text = 'Incorrect! \n No points. \n Slow down!'
                else: 
                    txt_stim.text = 'Incorrect! \n No points.'
            elif acc == 1:
                if (trl_strc[trial][0] == HIGH_1) or (trl_strc[trial][0] == HIGH_2): # high value reward
                    points = np.floor(points_function(rt)*POINTS_HIGH).astype(int)
                elif (trl_strc[trial][0] == LOW_1) or (trl_strc[trial][0] == LOW_2): # low value reward
                    points = np.floor(points_function(rt)*POINTS_LOW).astype(int)
                txt_stim.text = f'+ {points} points!'
                trg_val += 2   
        # if no response made, assign missing values etc.     
        elif not pressed:
            response = 999
            rt = 999
            acc = 999
            points = 0
            trg_val += 6
            txt_stim.text = 'Too slow! No points.'
        
        trg_rgb = dp.DPxTriggerToRGB(trg_val)
        if acc != 1:
            fdb_dur = FDB_DUR_LNG
        elif acc == 1:
            fdb_dur = FDB_DUR
        logging.warning('START_FDB')
        logging.flush()
        for frame in range(0, fdb_dur): # --------------------------------------------------- FDB ONSET

            # send trigger
            if frame < 2:
                trig_stim.lineColor = trg_rgb
                trig_stim.draw()

            # draw stimulus
            txt_stim.draw()
            win.flip()

        ## recover some trial information
        distRGB = trl_strc[trial][0]
        if distRGB in [HIGH_1, HIGH_2]:
            distVal = 'high'
        elif distRGB in [LOW_1, LOW_2]:
            distVal = 'low'
        
        if distRGB == BLUE:
            distColour = 'blue'
        elif distRGB == ORANGE:
            distColour = 'orange'
        elif distRGB == PURPLE:
            distColour = 'purple'
        elif distRGB == GREEN:
            distColour = 'green'  

        if trl_strc[trial][2] == LEFT:
            tarPos = 'left'
            distPos = 'right'
        elif trl_strc[trial][2] == RIGHT:
            tarPos = 'right'
            distPos = 'left'
        
        # calculate reward
        cents_this_trial = points*cents_per_point
        points_this_block += points
        cents_this_block += cents_this_trial
        cumulative_cents += cents_this_trial
        cumulative_points += points

        # update experiment handler data to save
        exp.addData('Trial', trial + 1)
        exp.addData('RunningTrialNo', runningTrialNo)
        exp.addData('Block', block)
        exp.addData('DistractorRGB', distRGB)
        exp.addData('DistractorColour', distColour)
        exp.addData('DistractorValue', distVal)
        exp.addData('DistractorXY', dis_stim.pos)
        exp.addData('DistractorPosition', distPos)
        exp.addData('TargetXY', tar_stim.pos)
        exp.addData('TargetPosition', tarPos)
        exp.addData('ResponseRule', cue_stim.text)
        exp.addData('CorrectResponse', cor_response)
        exp.addData('Response', response)
        exp.addData('RT', rt)
        exp.addData('Accuracy', acc)
        exp.addData('Points', points)
        exp.addData('CumulativePoints', cumulative_points)
        exp.addData('Cents', cents_this_trial)
        exp.addData('CumulativeCents', cumulative_cents)

        exp.nextEntry() # move to next line in data output
        runningTrialNo += 1
        framesPerTrial[f'Trial_{trial + 1}'] = win.frameIntervals

    # take a break and save data at the end of the block --------------------------------------------------
    win.recordFrameIntervals = False
    # save behavioural data
    exp.saveAsWideText(
        fileName = FILEPATH, 
        appendFile=None,
        fileCollisionMethod='overwrite')
    framesPerBlock[f'Block_{block}'] = framesPerTrial
    with open(FRMSFILEPATH, 'w') as file:
        file.write(json.dumps(framesPerBlock)) 
    # trigger save of eye tracking data
    trg_rgb = dp.DPxTriggerToRGB(254)
    for frame in range(2): 

        # draw stimulus
        trig_stim.lineColor = trg_rgb
        trig_stim.draw()
        win.flip()
    # present end of block text
    if block < NBLOCKS:
        txt_stim.text = f'''
        End of Block {block}. Take a break. \n\n 
        You got {points_this_block} points in that block! That's an extra ${np.round(cents_this_block/100, 2)}. \n\n
        You've earnt an extra ${np.round(cumulative_cents/100, 2)} so far. Great job! \n\n
        Press space when you're ready to continue.''' 
    elif block == NBLOCKS:
        txt_stim.text = f'''
        End of Block {block}. You're all done! \n\n 
        You got {cumulative_points} points! That's an extra ${np.round(cumulative_cents/100, 2)}. \n\n
        Press space and contact the experimenter.''' 
    event.clearEvents()
    txt_idx = 0
    while True:

        # draw
        txt_stim.draw()
        win.flip()

        # collect user input to exit 
        pressed = event.getKeys(keyList = CONKEYS)
        if pressed:
            break

        # udpate frame
        txt_idx += 1 
    
# end experiment and shut everything down ----------------------------------------------------
# trigger end of eye tracking
trg_rgb = dp.DPxTriggerToRGB(255)
for frame in range(2): 
    # draw stimulus
    trig_stim.lineColor = trg_rgb
    trig_stim.draw()
    win.flip()

# Run debriefing
if DEBRIEF: 
    debrief = {
        'I acknowledge that I have been appropriately debriefed and shown a copy of the debriefing questions': False,
        'Name': '',
        'Date': ''}

    if expInfo['Debrief']:
        dlg = gui.DlgFromDict(
            debrief, 
            title = 'Acknowledgement of Debriefing')
        if not dlg.OK:
            core.quit()

    # create experiment handler to save output
    dbrf = data.ExperimentHandler(name=EXPERIMENT,
                                  extraInfo=debrief)
    dbrf.dataNames = []
    dbrf.nextEntry() 
    identifier = datetime.now().timestamp()
    identifier = str.split(str(identifier), '.')[0]    
    dbrf.saveAsWideText(
            fileName = DATAPATH + f'AcknowledgementOfDebriefing_{identifier}.txt', 
            appendFile=None,
            fileCollisionMethod='overwrite')

# close everything else down
logging.warning('END_EXPERIMENT')
logging.flush()
event.clearEvents()
dp.DPxSetPPxDlpSeqPgrm('RGB')
dp.DPxDisableDoutPixelMode()
dp.DPxWriteRegCache()
dp.DPxClose()
win.close()
# check data --------------------------------------------------------------------------------
## check dropped frames
# thold = 1/REFRATE + 0.002
thold = win.refreshThreshold
frms_all = []
with open(FRMSFILEPATH) as file:
    frms = json.loads(file.read())
for block in frms.keys():
    for trial in frms[block].keys():
        frms_all.append(frms[block][trial][:])
frms_all = np.concatenate(frms_all)
dropped_frames = sum(frms_all > thold)
dropped_frames_pcnt = np.round(dropped_frames/len(frms_all), 2)
plt.plot(frms_all, marker = 'o', ms = 0.5, ls = '')
plt.hlines(thold, xmin=0, xmax=len(frms_all), color = 'black', ls = '--')
plt.hlines(1/REFRATE, xmin=0, xmax=len(frms_all), color = 'black', )
plt.hlines((1/REFRATE)*2, xmin=0, xmax=len(frms_all), color = 'black')
plt.title(f'Dropped frames: {dropped_frames} of {len(frms_all)} ({dropped_frames_pcnt}%)')
plt.ylabel('Frame duration (s)')
plt.xlabel('Frame')
plt.show()
print(f'Dropped frames: {dropped_frames} of {len(frms_all)} ({dropped_frames_pcnt}%)')

# ## check behavioural performance
# data = pl.read_csv(FILEPATH, sep="\t")
# # check accuracy and response time
# (data.filter(
#     (pl.col('RT') <= 1)
#     ).groupby(
#     ["ResponseRule", "DistractorValue"]
#     ).agg(
#     [pl.col('RT').mean().alias('Mean RT'),
#     pl.col('Accuracy').mean().alias('Mean Acc')],
#     ))
# # check points and reward total
# data.select(['Points', 'Cents']).sum()
# reward = np.round(data[-1, 'CumulativeCents']/100, 2)
# print(f'Reward: ${reward}')

# # quit
# core.quit()


