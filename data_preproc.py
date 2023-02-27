from sensorfabric.athena import athena
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import numpy as np
from math import floor, ceil
from torch import nn
import torch.nn.functional as F
import datetime
import boto3
import time
import s3fs
import scipy
from sklearn.metrics import auc
from scipy import signal
from collections import defaultdict
import h5py
import os

import warnings
warnings.filterwarnings("ignore")

#the next line is for running this in JupyterNote book, else need to use 'export' in the python env
#%env AWS_PROFILE=uaprofile

class Dataset:
  """
  connnect to AWS databse to get necessary data
  """
    def __init__(self, database):
      """
      database: string, please use elise
      """
        self.client = boto3.client("athena")
        self.database = database
    
    def query_to_df(self, Qeury):
      """
      Input:
        Qeury: string, SQL query
      return:
        Pandas.DataFrame or None if query failed.
      """
        res = self.client.start_query_execution(QueryString=Qeury, QueryExecutionContext={'Database':self.database})
        query_id = res['QueryExecutionId']
        while True:
            state = self.client.get_query_execution(QueryExecutionId = query_id)['QueryExecution']['Status']['State']
            if state in {"RUNNING", "QUEUED"}:
                time.sleep(1)
            else:
                print(state)
                break
        if state == "SUCCEEDED":
            path = self.client.get_query_execution(QueryExecutionId = query_id)['QueryExecution']['ResultConfiguration']['OutputLocation']
            df = pd.read_csv(path)
            return df
        else:
            print("Qeury failed")
            return None
    
    def get_pids(self):
        pid_df = self.query_to_df("SELECT redcap_pid.pid FROM redcap_pid INNER JOIN temperature_pid ON temperature_pid.pid=redcap_pid.pid")
        return np.array(pid_df['pid'])

def cal_hrv_freq(rr_intervals):
  """
  Calculate Frequency domain HRV features
  Input: 
    rr_intervals: array of integers, including rr intervals
  Output:
    LF: float, Low frequency power
    HF: float, High frequency power
    lf_to_hf: float, LF to HF ratio
  """
    time = []
    t = 0
    for rr in rr_intervals:
        t += rr/1000
        time.append(t)

    nout = 10000
    w = np.linspace(0.001, np.pi, nout)
    m = np.mean(rr_intervals)
    for i in range(len(rr_intervals)):
        rr_intervals[i] -= m
        
    pgram = scipy.signal.lombscargle(time, rr_intervals, w, normalize=True)
    w = w/2/np.pi

    LF_x, LF_y = w[np.logical_and(w>= 0.04, w < 0.15)], pgram[np.logical_and(w>= 0.04, w < 0.15)]
    HF_x, HF_y = w[np.logical_and(w>= 0.15, w < 0.4)], pgram[np.logical_and(w>= 0.15, w < 0.4)]

    LF, HF = auc(LF_x, LF_y), auc(HF_x, HF_y)
    try:
        lf_to_hf = LF/HF
    except:
        lf_to_hf = 0
    return LF, HF, lf_to_hf

def cal_hrv(rr_intervals):
    """
    Calculate HRV based on rr_intervals
    Input: 
      rr_intervals: array of integers, including rr intervals
    Output:
      SDNN: float, std of normal RR intevals
      RMSSD: float, root mean square of successive RR intervals
      NN50: int, count of intervals that difference is larger than 50ms compared to previous intervals
      PNN50: float, percent of NN50 in all NN intervals
      LF: float, Low frequency power
      HF: float, High frequency power
      lf_to_hf: float, LF to HF ratio    
      HR: float, heart rate
    """
    SDNN = np.std(rr_intervals)
    HR = 60*1000/np.mean(rr_intervals)
    diffs = []
    NN50 = 0
    for i in range(1, len(rr_intervals)):
        dif = abs(rr_intervals[i] - rr_intervals[i-1])
        if dif >= 50: NN50 += 1
        diffs.append(dif**2)
    RMSSD = np.sqrt(np.mean(diffs))

    LF, HF, lf_to_hf = cal_hrv_freq(rr_intervals)
    if len(rr_intervals) <= 1: PNN50 = 0
    else: PNN50 = NN50/(len(rr_intervals) - 1) * 100
    return SDNN, RMSSD, NN50, PNN50, LF, HF, lf_to_hf, HR

def filter_rr_interval(rr_intervals, low = 400, high = 2000):
    #Normal rr_intervals = [400, 2000] ms
    filtered_rr_intervals = []
    for i in range(len(rr_intervals)):
        if rr_intervals[i] < low or rr_intervals[i] > high: continue
        else: filtered_rr_intervals.append(rr_intervals[i])
    return filtered_rr_intervals

#perform interpolation for the missing data
def interpolate_missing_data(data_dict):
    def helper(array):
        return np.isnan(array), lambda z: z.nonzero()[0]
    for key in ["temperature", "SDNN", "RMSSD", "NN50", "PNN50", "LF", "HF", "lf_to_hf", "HR", "Labor"]:
        data_dict[key] = np.array(data_dict[key])
        nans, val = helper(data_dict[key])
        data_dict[key][nans] = np.interp(val(nans), val(~nans), data_dict[key][~nans])

if __name__ == "__main__":
    Ds = Dataset('elise')
    pids = Ds.get_pids()
    
    #get pids that has been processed
    processed_data = [int(file.split("_")[-1].split(".")[0]) for file in os.listdir("/xdisk/aoli1/jiayanh/BioBaby")]
    
    for i, pid in enumerate(pids):
        if pid in processed_data:
            print("pid = %d has been processed" %(pid))
            continue
        data_dict = {"temperature":[], "SDNN":[], "RMSSD":[], "NN50":[], "PNN50":[], "LF":[], "HF":[], "lf_to_hf":[], "HR":[], "Labor":[]}
        print("-"*10, "processing %d of %d subject" %(i+1, len(pids)), "-"*10)
        #get temperature data
        df_temp = Ds.query_to_df("SELECT * FROM temperature WHERE pid = {}".format(pid))
        #get ibi data
        df_ibi = Ds.query_to_df("SELECT * FROM ibi WHERE pid = {}".format(pid))
        #get labor date
        labor_date = Ds.query_to_df("SELECT * FROM labordate WHERE pid = {}".format(pid))['labordate']
        labor_date = datetime.datetime.strptime(str(labor_date[0]), "%Y-%m-%d")
        #get temperature and ibi data overlap time
        temp_start, temp_end = int(np.min(df_temp['unixtimestamp'])), int(np.max(df_temp['unixtimestamp']))
        ibi_start, ibi_end = int(np.min(df_ibi['unixtimestamp'])), int(np.max(df_ibi['unixtimestamp']))
        start, end = max(temp_start,ibi_start), min(temp_end, ibi_end)
        #filter the data
        df_temp_overlap = df_temp[(df_temp["unixtimestamp"]>=start) & (df_temp["unixtimestamp"]<=end)]
        df_ibi_overlap = df_ibi[(df_ibi["unixtimestamp"]>=start) & (df_ibi["unixtimestamp"]<=end)]
        #Align the starting point from the second day, each day will have a sampling rate of 5 min, so 288 data point per day
        #e.g., '2021-11-26 19:20:02'
        ibi_start_time = datetime.datetime.fromtimestamp(int(np.min(df_ibi_overlap['unixtimestamp'])))
        #if starting time is not 00:00, truncate data to next midnight
        if int(ibi_start_time.hour) + int(ibi_start_time.minute) > 1:
            new_start_date = ibi_start_time +  datetime.timedelta(days=1)
            new_start_date = datetime.datetime.combine(new_start_date.date(), datetime.time(0))
            new_start_unixtimestamp = int(datetime.datetime.timestamp(new_start_date))
            df_temp_overlap = df_temp_overlap[(df_temp_overlap["unixtimestamp"]>=new_start_unixtimestamp)]
            df_ibi_overlap = df_ibi_overlap[(df_ibi_overlap["unixtimestamp"]>=new_start_unixtimestamp)]
        
        #use a window size of 5 mins
        start_stamp = new_start_unixtimestamp
        end_stamp = start_stamp + 300
        
        while end_stamp < min(int(np.max(df_temp_overlap['unixtimestamp'])), int(np.max(df_ibi_overlap['unixtimestamp']))):
            df_temp_slice = df_temp_overlap[(df_temp_overlap["unixtimestamp"]>=start_stamp) & (df_temp_overlap["unixtimestamp"]<end_stamp)]
            df_ibi_slice = df_ibi_overlap[(df_ibi_overlap["unixtimestamp"]>=start_stamp) & (df_ibi_overlap["unixtimestamp"]<end_stamp)]

            if df_temp_slice.empty:
                data_dict['temperature'].append(np.nan)
            else:
                data_dict['temperature'].append(np.mean(df_temp_slice['skintemp']))

            if df_ibi_slice.empty:
                data_dict['SDNN'].append(np.nan)
                data_dict['RMSSD'].append(np.nan)
                data_dict['NN50'].append(np.nan)
                data_dict['PNN50'].append(np.nan)
                data_dict['LF'].append(np.nan)
                data_dict['HF'].append(np.nan)
                data_dict['lf_to_hf'].append(np.nan)
                data_dict['HR'].append(np.nan)
            else:
                try:
                #in case of multiple measurement at the same data point, average the ibi with same time stamp first
                    rr_intervals = filter_rr_interval(np.array(df_ibi_slice['ibi']))
                    SDNN, RMSSD, NN50, PNN50, LF, HF, lf_to_hf, HR = cal_hrv(rr_intervals)
                    data_dict['SDNN'].append(SDNN)
                    data_dict['RMSSD'].append(RMSSD)
                    data_dict['NN50'].append(NN50)
                    data_dict['PNN50'].append(PNN50)
                    data_dict['LF'].append(LF)
                    data_dict['HF'].append(HF)
                    data_dict['lf_to_hf'].append(lf_to_hf)
                    data_dict['HR'].append(HR)
                except:
                    data_dict['SDNN'].append(np.nan)
                    data_dict['RMSSD'].append(np.nan)
                    data_dict['NN50'].append(np.nan)
                    data_dict['PNN50'].append(np.nan)
                    data_dict['LF'].append(np.nan)
                    data_dict['HF'].append(np.nan)
                    data_dict['lf_to_hf'].append(np.nan)
                    data_dict['HR'].append(np.nan)
            #move to next 5 mins window
            start_stamp = end_stamp
            end_stamp = start_stamp + 300
            #calculate date info
            current_date = datetime.datetime.fromtimestamp(start_stamp)
            labor_days_minus_cnt = labor_date - current_date
            labor_days_minus_cnt = labor_days_minus_cnt.days
            
            if not data_dict["Labor"] or data_dict["Labor"][-1] != labor_days_minus_cnt: 
                data_dict["Labor"].append(labor_days_minus_cnt)
                
        interpolate_missing_data(data_dict)
        
        #save to h5py
        h = h5py.File('/xdisk/aoli1/jiayanh/BioBaby/data_dict_{}.hdf5'.format(pid), "w")
        dict_group = h.create_group('data_dict')
        for k, v in data_dict.items():
            dict_group[k] = v
        h.close()
