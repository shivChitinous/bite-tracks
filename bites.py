import numpy as np
import scipy as sp
import pandas as pd
from scipy import signal
from scipy.fft import fftshift
import imageio

def derive(ldf,T,t_bite_start,t_bite_end,ctr='lab3'):
    df = ldf.copy()
    #slice out the bite
    df = df.loc[round(len(df)*t_bite_start/T):round(len(df)*t_bite_end/T)-1,:].copy()
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

def moving_average(x, 
                   window, #in seconds
                   T
                  ):
    w = int(window*np.shape(x)[0]/T)
    smoothed = sp.signal.correlate(x,np.ones(w),mode='same')/w
    return smoothed[int(w/2):-int(w/2)]

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

def get_lag_curve(corrDf, signal1= 'straightness', signal2 = 'insert_length'):
    x = corrDf[signal1] - corrDf[signal1].mean()
    y = corrDf[signal2] - corrDf[signal2].mean()

    # Compute full cross-correlation
    corr = np.correlate(x, y, mode='full')
    r_sq = (corr/(np.linalg.norm(x) * np.linalg.norm(y)))**2

    # Create lag array
    lags = np.arange(-len(x) + 1, len(x))*corrDf['time [s]'].max()/len(corrDf)

    return lags, r_sq

def get_needle_time(derDf,T,mode='insertion',window=1/25,prominence=10,distance=2/25):
    window = int(window*np.shape(derDf)[0]/T)
    distance = int(distance*np.shape(derDf)[0]/T)
    sign = -1 if mode=='removal' else 1
    event_times,_ = sp.signal.find_peaks(sign*sp.ndimage.gaussian_filter(-insert_length(derDf), window), prominence = prominence, distance = distance)
    return event_times

def insert_length(derDf):
    L = needle_length(derDf)
    iL = (np.nanmax(L)-L)/np.nanmax(L)
    return iL

def needle_length(derDf, likelihood_threshold=0.1):
    L = (np.sqrt((derDf.loc[:,('lab1','x')]**2).values + (derDf.loc[:,('lab1','y')]**2).values))
    L[derDf[('lab1', 'likelihood')].values < likelihood_threshold] = np.nan
    return L

def gut_length(derDf):
    L = (np.sqrt(((derDf.loc[:,('gut3','x')]-derDf.loc[:, ('gut4', 'x')])**2).values + ((derDf.loc[:,('gut3','y')]-derDf.loc[:, ('gut4', 'y')])**2).values))
    return (L-np.nanmin(L))/(np.nanmax(L)-np.nanmin(L))  # Normalize to [0, 1]

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

def estimate_3d_palp_angle(derDf, r=1.0, segment = 1):
    L1, L2 = np.sqrt(derDf[[('pal1', 'x'), ('pal2', 'x'), ('pal3', 'x')]].diff(axis=1).iloc[:, 1:].values**2 + derDf[[('pal1', 'y'), ('pal2', 'y'), ('pal3', 'y')]].diff(axis=1).iloc[:, 1:].values**2).T
    L3 = np.sqrt(derDf[[('tar'+str(segment), 'x'), ('tar'+str(segment+1), 'x')]].diff(axis=1).iloc[:,-1].values**2 + derDf[[('tar'+str(segment), 'y'), ('tar'+str(segment+1), 'y')]].diff(axis=1).iloc[:,-1].values**2)
    Theta = np.deg2rad(angle_part(derDf,'pal'))
    s1, s2 = np.clip(L1 / (r * L3), -1, 1), np.clip(L2 / (r * L3), -1, 1)
    c1, c2 = np.sqrt(1 - s1**2), np.sqrt(1 - s2**2)
    cos_theta = np.clip(c1 * c2 + s1 * s2 * np.cos(Theta), -1, 1)
    return np.rad2deg(np.arccos(cos_theta))

def estimte_3d_labial_twist(derDf, segment = 1, lab = 2):
    L3 = np.sqrt(derDf[[('tar'+str(segment), 'x'), ('tar'+str(segment+1), 'x')]].diff(axis=1).iloc[:,-1].values**2 + derDf[[('tar'+str(segment), 'y'), ('tar'+str(segment+1), 'y')]].diff(axis=1).iloc[:,-1].values**2)
    L2 = np.sqrt(derDf[[('lab'+str(lab), 'x'), ('lab'+str(lab+1), 'x')]].diff(axis=1).iloc[:, 1].values**2 + derDf[[('lab'+str(lab), 'y'), ('lab'+str(lab+1), 'y')]].diff(axis=1).iloc[:, 1].values**2)/L3
    L2 = (L2 - np.mean(L2))/(np.max(L2) - np.min(L2)) #center and scale
    return np.rad2deg(np.arcsin(L2))

def butterworth_filter_subtract_hilbert_phase(signal, cutoff, fs, order=5):
    from scipy.signal import butter, filtfilt, hilbert
    import numpy as np

    # Normalize input
    signal = (signal - np.nanmin(signal)) / (np.nanmax(signal) - np.nanmin(signal))

    # Low-pass filter
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low')
    filtered = filtfilt(b, a, signal)

    # Detrend (optional)
    detrended = signal - filtered

    # Normalize again (optional — may skip)
    # detrended = (detrended - np.nanmin(detrended)) / (np.nanmax(detrended) - np.nanmin(detrended))

    # Hilbert phase
    analytic = hilbert(detrended)
    return np.angle(analytic)

def deltaPhase(corrDf, signal1 = 'pal_angle', signal2 = 'lab_angle', filt_win = 0.01, cutoffHz = 7.5, removal= False, filt_removal = 1, mov_percentile = 50, dir = 'removal'):
    """
    Calculate the phase difference between two signals.
    """
    fs = len(corrDf)/np.max(corrDf['time [s]'])  # Sampling frequency (Hz)
    filt_win = round(filt_win * fs) | 1 # Convert window size to samples
    if removal:
        removal = straightness_motion(corrDf, filt_removal=filt_removal, mov_percentile=mov_percentile, dir=dir)
    else:
        removal = np.ones(len(corrDf), dtype=bool)
    signal1vals = corrDf[signal1].values[removal]
    signal2vals = corrDf[signal2].values[removal]
    dPh = np.rad2deg(sp.signal.medfilt(np.unwrap((butterworth_filter_subtract_hilbert_phase(signal1vals, cutoffHz, 
                            fs)-butterworth_filter_subtract_hilbert_phase(signal2vals, cutoffHz
                                                                           , fs))), filt_win)%(2*np.pi))
    return dPh

def straightness_motion(corrDf, filt_removal=1, mov_percentile=50, dir = 'removal'):
    from skimage.morphology import opening
    fs = len(corrDf)/np.max(corrDf['time [s]'])
    filt_removal = round(filt_removal * fs) | 1
    if dir == 'insertion': removal = np.diff(sp.signal.medfilt(corrDf['insert_length'], filt_removal), prepend = 0)<np.percentile(np.diff(sp.signal.medfilt(corrDf['insert_length'], filt_removal), prepend=0), mov_percentile)
    elif dir=='removal': removal = np.diff(sp.signal.medfilt(corrDf['insert_length'], filt_removal), prepend = 0)>np.percentile(np.diff(sp.signal.medfilt(corrDf['insert_length'], filt_removal), prepend=0), mov_percentile)
    return opening(removal, np.array([1]*filt_removal))

def zoom_str_mot(corrDf, zoomInterval = 0.5, which=1, filt_removal=1, mov_percentile=50, dir = 'removal'):
    from skimage.measure import label
    removal = label(straightness_motion(corrDf, filt_removal = filt_removal, mov_percentile=mov_percentile, dir=dir))==which
    T = int(zoomInterval * len(corrDf)/corrDf['time [s]'].max())
    if np.sum(removal) > T:
        rem = removal.copy()
        rem[:np.where(removal)[0][0] + int((np.where(removal)[0][-1] - np.where(removal)[0][0])/2 - T/2)] = 0
        rem[np.where(removal)[0][0] + int((np.where(removal)[0][-1] - np.where(removal)[0][0])/2 + T/2):] = 0
        return rem.copy()
    else:
        return np.zeros(len(corrDf), dtype=bool)
