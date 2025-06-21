import numpy as np
import scipy as sp
import pandas as pd
import imageio
from scipy.signal import butter, filtfilt, hilbert
from skimage.measure import label
from skimage.morphology import closing, opening, erosion, dilation
from scipy.stats import zscore

def derive(ldf,T,ctr='lab3',tip = 'lab1',
           window=0.1, #seconds
           vel_lim=5, #pixels/second
           threshold=0.99,
           grouplabel = 'bite',
           ):
    
    df = ldf.copy()
    vel_ctr = np.sqrt((((ldf.loc[:,(ctr,
                'x')].diff())**2).values)+(((ldf.loc[:,(ctr,
                'y')].diff())**2).values))
    vel_tip = np.sqrt((((ldf.loc[:,(tip,
                'x')].diff())**2).values)+(((ldf.loc[:,(tip,
                'y')].diff())**2).values))
    
    '''vel_tip2 = np.sqrt((((ldf.loc[:,(tip2,
                'x')].diff())**2).values)+(((ldf.loc[:,(tip2,
                'y')].diff())**2).values))'''
    df[grouplabel] = label(moving_average((vel_ctr <= vel_lim) | np.isnan(vel_ctr), window=window, T=T) >= threshold)


    for part in sorted(set(list(list(zip(*ldf.columns))[0])),key=list(list(zip(*ldf.columns)))[0].index):
        df.loc[:,(part,'speed')] = 0
        df.loc[:,(part,'norm_speed')] = 0
    df.loc[:,(slice(None),'x')] = (df.loc[:,(slice(None),'x')].values)-(df.loc[:,(ctr,'x')].values.reshape(-1, 1))
    df.loc[:,(slice(None),'y')] = -((df.loc[:,(slice(None),'y')].values)-(df.loc[:,(ctr,'y')].values.reshape(-1, 1)))
    df.loc[:,(slice(None),'speed')] = np.sqrt((((df.loc[:,(slice(None),
                'x')].diff())**2).values)+(((df.loc[:,(slice(None),
                'y')].diff())**2).values))
    df.loc[:,(slice(None),('norm_speed'))] = np.array(df.loc[:,(slice(None),
                            ('speed'))]/(df.loc[:,(slice(None),('speed'))].max()))
    
    df['time [s]'] = df.index * T / len(df)
    df['tip_speed'] = vel_tip
    #df['tip2_speed'] = vel_tip2
    df['ctr_speed'] = vel_ctr
    
    return df

def calculate_angle(point_a, point_b):
    ang_a = np.arctan2(*point_a[::-1])
    ang_b = np.arctan2(*point_b[::-1])
    return (np.rad2deg((ang_b - ang_a) % (2 * np.pi))-180)%360-180

def angle_part(legDf, part):
    df = legDf.copy()
    a = df.loc[:, ((part+'1'), ('x','y'))].values
    b = df.loc[:, ((part+'2'), ('x','y'))].values
    c = df.loc[:, ((part+'3'), ('x','y'))].values
    ba = a - b
    bc = c - b
    angle = np.zeros(len(ba))
    for i in range(len(ba)):
        angle[i] = calculate_angle(ba[i,:],bc[i,:])
    return np.rad2deg(np.unwrap(np.deg2rad(angle)))

def moving_average(x, window, T):
    w = int(window * len(x) / T)
    y = sp.signal.correlate(x, np.ones(w), mode='same') / w
    return np.where((np.arange(len(x)) < w//2) | (np.arange(len(x)) >= len(x) - w//2), np.nan, y)

def get_leg_time(derDf,T,mode='removal',window=0.05,variable='angle'):
    sign = +1 if mode=='removal' else -1
    if variable=='angle': 
        v = angle_part(derDf,'tar')
        height = 1
    elif variable=='straightness': 
        v = straightness(derDf)
        height = 0.03
    press_times = sp.signal.find_peaks(sign*np.diff(moving_average(v,window=window,T=T)),
        height=height,distance=10)
    return press_times[0]

def get_lag_curve(corrDf, signal1='straightness', signal2='insert_length', time_col='time [s]'):
    corrDf = corrDf.copy()

    # Boolean mask where both signals are valid
    valid_mask = ~((corrDf[signal1].isna()) | (corrDf[signal2].isna()) | (corrDf[signal1]==0) | (corrDf[signal2]==0))

    # Find longest contiguous chunk of valid rows
    valid_groups = (valid_mask != valid_mask.shift()).cumsum()
    group_lengths = valid_mask.groupby(valid_groups).sum()
    longest_valid_group = group_lengths.idxmax()

    longest_chunk = corrDf[(valid_groups == longest_valid_group) & valid_mask]

    # Center signals (mean subtract)
    x = longest_chunk[signal1].to_numpy() - longest_chunk[signal1].mean()
    y = longest_chunk[signal2].to_numpy() - longest_chunk[signal2].mean()

    # Compute full cross-correlation (normalized squared correlation)
    corr = np.correlate(x, y, mode='full')
    r_sq = (corr / (np.linalg.norm(x) * np.linalg.norm(y)))**2

    # Create lag array in time units
    dt = longest_chunk[time_col].diff().median()  # Estimate sampling interval
    lags = np.arange(-len(x) + 1, len(x)) * dt

    return lags, r_sq

def get_needle_time(derDf,T,mode='insertion',window=1/25,prominence=10,distance=2/25):
    window = int(window*np.shape(derDf)[0]/T)
    distance = int(distance*np.shape(derDf)[0]/T)
    sign = -1 if mode=='removal' else 1
    event_times,_ = sp.signal.find_peaks(sign*sp.ndimage.gaussian_filter(-insert_length(derDf), window), prominence = prominence, distance = distance)
    return event_times

def insert_length(derDf, maxWind = 0.5, movement_wind= 1, movement_thresh = 1, bucklingAngleRange = 170, buckling_thresh = 0.65):
    L = (np.sqrt((derDf.loc[:,('lab1','x')]**2).values + (derDf.loc[:,('lab1','y')]**2).values))
    fs = 1/np.median(np.diff(derDf['time [s]'].values))
    l1 = np.sqrt(derDf[('lab2', 'x')].values**2 + derDf[('lab2', 'y')].values**2)
    l2 = np.sqrt((derDf[('lab1', 'x')].values-derDf[('lab2', 'x')].values)**2 + (derDf[('lab1', 'y')].values-derDf[('lab2', 'y')].values)**2)
    R = (l1 + l2)
    M = (pd.Series(R).rolling(int(fs*maxWind), center=True).max().ffill().bfill().values)
    mean_tip_movement = derDf['tip_speed'].rolling(int(fs*movement_wind), center=True).mean().ffill().bfill().values
    iL = (M-L)/M
    iL[~dilation(mean_tip_movement < movement_thresh, np.array([1]*(int(fs*movement_wind))))] = 0  # Set to 0 if tip movement is below threshold implying that the tip is above the skin
    buckling_mask = (np.abs(angle_part(derDf, part ='lab'))>bucklingAngleRange) & (
    np.abs(zscore(pd.Series(np.abs(angle_part(derDf, part ='lab'))).rolling(window=int(fs*movement_wind), 
                center=True).mean().ffill().bfill().values)-zscore(pd.Series(R).rolling(window=int(fs*movement_wind), center=True).mean().ffill().bfill().values))<buckling_thresh)
    iL[dilation(buckling_mask, np.array([1]*(int(fs*movement_wind))))] = 0
    return -iL #returns negative insertion length for visualization purposes

def gut_length(derDf):
    L = (np.sqrt(((derDf.loc[:,('gut3','x')]-derDf.loc[:, ('gut4', 'x')])**2).values + ((derDf.loc[:,('gut3','y')]-derDf.loc[:, ('gut4', 'y')])**2).values))
    return L/L[0] # Normalize to minimum length

def straightness(derDf, part='tar'):
    parts = list(part + np.array(range(1, 4), dtype='str').astype('object'))
    
    # Extract coordinates: A = pt1, B = pt2, C = pt3
    vec = derDf.loc[:, (parts, ('x', 'y'))].values
    x1, y1 = vec[:, 0], vec[:, 1]  # A
    x2, y2 = vec[:, 2], vec[:, 3]  # B
    x3, y3 = vec[:, 4], vec[:, 5]  # C
    
    # Shoelace formula (area of triangle ABC)
    area = 0.5 * np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    
    # Compute |A - B| and |B - C|
    ab = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    bc = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)
    
    # Theoretical max area: (1/2) * |AB| * |BC|
    max_area = 0.5 * ab * bc
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        norm_area = np.true_divide(area, max_area)
        norm_area[np.isnan(norm_area)] = 0  # if max_area=0, set norm_area=0

    # Invert to get straightness
    return 1 - norm_area

def videoExtract(filename, engine='ffmpeg'):
    video = imageio.get_reader(filename, engine)
    frames = []

    # Loop through each frame and append it to the list
    for frame in video:
        frames.append(np.array(frame))

    # Stack all frames along a new dimension to create a 4D array
    video_array = np.stack(frames, axis=0)
    return video_array

def estimate_3d_palp_angle(derDf, r=1.0, segment = 2):
    l1, l2 = np.sqrt(derDf[[('pal1', 'x'), ('pal2', 'x'), ('pal3', 'x')]].diff(axis=1).iloc[:, 1:].values**2 + derDf[[('pal1', 'y'), ('pal2', 'y'), ('pal3', 'y')]].diff(axis=1).iloc[:, 1:].values**2).T
    T = np.sqrt(derDf[[('tar'+str(segment), 'x'), ('tar'+str(segment+1), 'x')]].diff(axis=1).iloc[:,-1].values**2 + derDf[[('tar'+str(segment), 'y'), ('tar'+str(segment+1), 'y')]].diff(axis=1).iloc[:,-1].values**2)
    theta = np.deg2rad(angle_part(derDf,'pal'))
    s1, s2 = np.clip(l1 / (r * T), -1, 1), np.clip(l2 / (r * T), -1, 1)
    c1, c2 = np.sqrt(1 - s1**2), np.sqrt(1 - s2**2)
    cos_theta = np.clip(c1 * c2 + s1 * s2 * np.cos(theta), -1, 1)
    return np.rad2deg(np.arccos(cos_theta))

def estimate_3d_labial_twist(derDf, r = 1.2, segment = 2):
    L1 = derDf[[('lab2', 'x'), ('lab2', 'y')]].values
    L2 = derDf[[('lab1', 'x'), ('lab1', 'y')]].values - derDf[[('lab2', 'x'), ('lab2', 'y')]].values
    l1 = np.sqrt(derDf[('lab2', 'x')].values**2 + derDf[('lab2', 'y')].values**2)
    l2 = np.sqrt((derDf[('lab1', 'x')].values-derDf[('lab2', 'x')].values)**2 + (derDf[('lab1', 'y')].values-derDf[('lab2', 'y')].values)**2)
    T = np.sqrt(derDf[[('tar'+str(segment), 'x'), ('tar'+str(segment+1), 'x')]].diff(axis=1).iloc[:,-1].values**2 + derDf[[('tar'+str(segment), 'y'), ('tar'+str(segment+1), 'y')]].diff(axis=1).iloc[:,-1].values**2)
    L = r*T
    c = 1/(2*L)*np.sqrt(np.abs((L**2 - l1**2 - l2**2)**2 - 4*(l1**2)*(l2**2)))
    d = np.hstack([((L1 + L2)/2), c[:, np.newaxis]])
    dmean = np.nanmean(d, axis=0)                      # (3,)
    phi = np.arccos(np.clip(np.dot(d, dmean) / 
       (np.linalg.norm(d, axis=1) * np.linalg.norm(dmean)), -1.0, 1.0))
    return np.rad2deg(phi)


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs  # Nyquist Frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def deltaPhase(corrDf, bandpass = [5, 20], signal1='lab_angle', signal2='pal_angle', order = 4):
    fs = 1/np.median(np.diff(corrDf['time [s]'].values))
    corrDf = corrDf.copy().interpolate(method='linear', limit_direction='both')
    return np.rad2deg(((np.angle(hilbert(apply_bandpass_filter(corrDf[signal1], bandpass[0], bandpass[1], fs=fs, order = order)))-np.angle(
        hilbert(apply_bandpass_filter(corrDf[signal2], bandpass[0], bandpass[1], fs=fs, order = order))))-np.pi)%(2*np.pi)-np.pi
        )
    
def straightness_motion(corrDf, filts = [20, 20, 251], threshes = [-0.0001, 0.0005], dir = 'removal', phase = 'late', threshold_netchange = 0.01,
                        phase_time = 0.8 #second
                        ):
    
    corrDf = corrDf.copy()
    if dir == 'insertion': 
        removal = closing((sp.ndimage.gaussian_filter(np.diff(corrDf['insert_length'].interpolate(), append=0), filts[0]) < threshes[0]) & (corrDf['insert_length']<0.), np.array([1]*filts[2]))
    elif dir=='removal': 
        removal = closing((sp.ndimage.gaussian_filter1d(np.diff(corrDf[
            'insert_length'].interpolate(), prepend=0), filts[1])>threshes[1]) & (corrDf['insert_length']<0.), np.array([1]*filts[2]))
    rem = removal.copy()

    # Remove segments where the net change in insert length is below a threshold
    for l in np.unique(label(removal)):
        if l==0:
            continue
        else:
            if (np.abs(corrDf.loc[label(removal)==l, 'insert_length'].values[-1] - corrDf.loc[label(removal)==l, 
                                            'insert_length'].values[0])) < threshold_netchange:
                rem[label(removal)==l] = 0

    # Crop segments based on an "early" or "late" phase
    if phase is not None:
        T = int(phase_time * len(corrDf)/corrDf['time [s]'].max())
        labels = label(rem)  # Label once
        for l in np.unique(labels):
            if l == 0:
                continue  # 0 is background
            inds = np.where(labels == l)[0]
            if phase == 'late':
                if len(inds) > T:
                    rem[inds[:len(inds)-T]] = False
            elif phase == 'early':
                if T < len(inds):
                    rem[inds[T:]] = False
    return rem


def zoom_str_mot(corrDf, zoomInterval = 0.95, event='removal', which=1):
    
    removal = label(corrDf[event])==which
    T = int(zoomInterval * 1/np.median(np.diff(corrDf['time [s]'].values)))
    if np.sum(removal) > T:
        rem = removal.copy()
        rem[:np.where(removal)[0][0] + int((np.where(removal)[0][-1] - np.where(removal)[0][0])/2 - T/2)] = 0
        rem[np.where(removal)[0][0] + int((np.where(removal)[0][-1] - np.where(removal)[0][0])/2 + T/2):] = 0
        return rem.copy()
    else:
        return np.zeros(len(corrDf), dtype=bool)


def computePerBite(derDf, fun, grouplabel = 'bite', background=0, **kwargs):
    # Preallocate output array
    array = np.full(len(derDf), np.nan)

    # Loop over groups and fill in values directly
    for _, group_indices in derDf.groupby(grouplabel).groups.items():
        values = fun(derDf.loc[group_indices].reset_index(drop=True), **kwargs)
        array[group_indices] = values  # assign per row
    
    # Fill background values
    array[derDf[grouplabel] == background] = np.nan
    return array


def computeForAllBites(derDf, fun,  grouplabel = 'bite', background=0, **kwargs):
    
    array = fun(derDf, **kwargs)
    # Fill background values
    array[derDf[grouplabel] == background] = np.nan
    return array