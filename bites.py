import numpy as np
import scipy as sp
import pandas as pd
from scipy import signal
from scipy.fft import fftshift

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
    return np.rad2deg((ang_b - ang_a) % (2 * np.pi))

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
    return angle

def moving_average(x, 
                   window, #in seconds
                   T
                  ):
    w = int(window*np.shape(x)[0]/T)
    smoothed = sp.signal.correlate(x,np.ones(w),mode='same')/w
    return smoothed[int(w/2):-int(w/2)]

def get_leg_time(derDf,T,mode='removal',window=0.05):
    sign = +1 if mode=='removal' else -1
    press_times = sp.signal.find_peaks(sign*np.diff(moving_average(angle_part(derDf,'tar'),window=window,T=T)),
        height=1,distance=10)
    return press_times[0]