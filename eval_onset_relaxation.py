import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

n_channels = 64
R = C = int(np.sqrt(n_channels))
#pred_ = np.fromfile('predicted_signals_afno_validation_30to30.bin')
pred_ = np.fromfile('predicted_signals_fno_validation_30to30.bin')
targ_ = np.fromfile('real_signals_fno_validation_30to30.bin')

len_  = int(len(pred_) / n_channels)
nx = ny = int(np.sqrt(n_channels))
print(len_)
pred_ = np.reshape(pred_, [len_, R, C])
targ_ = np.reshape(targ_, [len_, R, C])
print("number of ts: ", len_)

sep_  = np.fromfile("validation_events.bin").astype('int')
n_events = len(sep_)-1
print("number of events: ", n_events)

start_events = 0
total_events = n_events
pred_ts   = 30
obsv_ts   = 30
start_ts  = sep_[start_events] - (pred_ts+obsv_ts)*start_events

corr_summary_fno = np.zeros(total_events)
for i in range(total_events):
    interval = sep_[start_events+1]-sep_[start_events] - pred_ts - obsv_ts
    print(interval, pred_.shape, start_ts, start_events)
    corr_summary_fno[i] = pearsonr(pred_[start_ts:start_ts+interval, :].flatten(), targ_[start_ts:start_ts+interval, :].flatten())[0]
    if (i%60==0):
        plt.plot(pred_[start_ts:start_ts+interval, 5,5], 'b')
        plt.plot(targ_[start_ts:start_ts+interval, 5,5], 'g')
        ts = sep_[start_events]+obsv_ts+pred_ts
        plt.show()
    start_events += 1
    start_ts += interval
plt.hist(corr_summary_fno,bins=20, alpha=0.5, label='FNO')
plt.legend()
plt.savefig('AFNO_30_30.png')

