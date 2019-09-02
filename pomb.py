####################################################
# 
# Author: Vishnu J. Seesahai
# Version: 1.0
# 10/2/18
#
# Script models a scoring system for optimizing a 
# Network for low latency, high bandwidth and low packet loss
#
###################################################

from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from pykalman import KalmanFilter
from pykalman import UnscentedKalmanFilter
import math
from datetime import datetime, timedelta
import sys
sys.path.append('./speedtest-cli/')
import speedtest


speedtester = speedtest.Speedtest()
IPV6_TARGET = '2001:4860:4860::8888'
IPV4_TARGET = 'www.google.com' #localhost' 
USE_IPV4 = True
NETWORK_INTERFACE = "en0"


# Thresholds
LATENCY_THRESHOLD = 40 # well below the 103 ms expected latency in network, and at the top 10% of latency
BANDWIDTH_THRESHOLD = 7 # 56 Mbps = 7 MBps = node upload rate at the 50th percentile
DROPS_THRESHOLD = .1 # .1% tolerance

# Kalman
AUTO_REGR = 3

BLOCK_SIZE = 10 # expanding block size from 1MB to 10MB is symetric to increasing block time by a factor of 10, ie 1 minute block times
BW_50 = BANDWIDTH_THRESHOLD # Bandwidth at the 50th percentile, MBps
T_HOP = float(BLOCK_SIZE / BW_50) # Time for one hop
NUM_OF_NODES = 9000
NUM_OF_PEERS = 8 
SEQ_PROP_FACTOR = math.log(NUM_OF_NODES, 2)  # A sequential propagation factor
T_50 = SEQ_PROP_FACTOR * T_HOP # Time to reach majority of network at 50th percentile bandwidth (BW_50)
MIN_POMB_REQUIRED = .9

# Outputs
POMB_SCORES_FILE = 'POMB_scores.csv'
POMB_RATES_FILE = 'POMB_rates.csv'
METRICS_FILE1 = 'metrics_file1.csv'
METRICS_FILE2 = 'metrics_file2.csv'

# Inits
sm_latency_array = []
cov_latency_array = []
sm_drops_array = []
cov_drops_array = []
sm_bandwidth_array = []
cov_bandwidth_array = []
data_array = []
avg_drops = []
avg_latency = []
avg_bandwidth = []
pomb_scores = []

# Construct a Kalman filter	
kf = KalmanFilter(transition_matrices = [1], 
	observation_matrices = [1],
	initial_state_mean = 0,
	initial_state_covariance = 1,
	observation_covariance=1,
	transition_covariance=.01)	#.01

def totimestamp(dt, epoch=datetime(1970,1,1)):
	td = dt - epoch
	# return td.total_seconds()
	return (td.microseconds + (td.seconds + td.days * 86400) * 10**6) / 10**6 
    
def run_kalman(data, metric_type, sm_array, cov_array):
	global kf 
	
	data_array = list((data[metric_type].values)) #.flatten())
	print('----------------')
	print(metric_type, ':\n',data_array)
	print('----------------')


	if (len(sm_array) == 0 or len(cov_array) == 0):		
		kf = kf.em(data_array, n_iter=5)
		state_means, covariances = kf.filter(data_array)
		#state_means, covariances = kf.smooth(data_array)
		state_means = pd.Series(state_means.flatten())#.flatten()
		covariances = pd.Series(covariances.flatten())#.flatten()
		sm_array.extend(list(state_means.values))
		cov_array.extend(list(covariances.values))
	else:
		for i in data:
			next_mean, next_covariance = kf.filter_update(
			    sm_array[-1], cov_array[-1], data[i]
			)
	
			sm_array.append(next_mean[0][0])
			cov_array.append(next_covariance[0][0])
			data_array.append(data[i])
		
		print(len(sm_array), len(data_array))
	
	if len(sm_array) == (len(data_array) + 1):
		kalman_mse = run_mse(sm_array[:-2], data_array)
		print(metric_type, 'MSE:', kalman_mse)
		print('NEXT PREDICTION:', sm_array[-1], 'WITH PREDICTED COVARIANCE:', cov_array[-1])	
	else:
		kalman_mse = run_mse(sm_array, data_array)
		print(metric_type, 'DROPS MSE:', kalman_mse)	
				
	return sm_array, cov_array, data_array
	
		
def run_mse(x_hat, x):
	mse = mean_squared_error(x_hat, x)
	#print('MODEL:', x_hat, 'ACTUAL:', x)		
	return mse 

def run_binarize(data, threshold, metric_type):

	data_array = list((data[metric_type].values))
	binarized = []
	print('-------------------------------------------------------------------')
	for i, elem in enumerate(data_array):
		binary_val = 0
		
		if (not metric_type == 'Bandwidth' and (data_array[i] <= threshold)) or (metric_type == 'Bandwidth' and data_array[i] >= threshold):
	 		binarized.append(1)
	 		binary_val = 1
		else:
			binarized.append(0)
			
		print('METRIC TYPE:', metric_type, '| ELEMENT:', elem, '| THRESHOLD:', threshold, '| BINARY VALUE:', binary_val)	
			
	return binarized
	
def append_file(filepath, df):

	
	if not os.path.isfile(filepath):
		df.to_csv(filepath, header=False, index=False)
	else:
		df.to_csv(filepath, mode='a', header=False, index=False)				
		

def EMA_calc2(n):
	ema = x
	ema_variance = 0 
	
	df_rates = pd.read_csv('./POMB_rates.csv', header=None)
	periods = len(df)
	ema = df_rates.ewm(span=n, min_periods=periods).mean()
	print('PERIODS:', periods, '| EMA NEW:', ema)

	return ema
	
def EMA_calc(x, n):
	ema = x
	ema_variance = 0 
	
	if os.path.isfile(POMB_SCORES_FILE):
		print('-------------------------------------------------------------------')
		# Get EMA for last time-step
		with open(POMB_SCORES_FILE) as f:
			lines = f.readlines()
			periods = len(lines)
			ema = float(lines[-1].split(',')[1])
			print('PERIODS:', periods, '| EMA LAST:', ema)
		
			if periods >= AUTO_REGR:	
				# Calc current EMA
				alpha = 2 / (n + 1);
				delta = x - ema;
				ema = ema + alpha * delta;
				ema_variance = (1 - alpha) * (ema_variance + (alpha * (delta ** 2)));
				print('EMA NEW:', ema, '| DELTA:', delta)	
		print('-------------------------------------------------------------------')
	
	return ema, ema_variance
'''
# Curls every 10 seconds - DEPRECATED
def curl_test(interval, df2):
	start_time = time.clock()
	current_time = 0
	bandwidth = 0 
	kb_to_mb_conversion = .001
	output = []

	while current_time < interval:
		
		sys_cmd = 'wget http://speedtest.tele2.net/1MB.zip -O ->/dev/null'
		p = Popen(sys_cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
		
		for line in p.stdout:
			line = line.rstrip()
			output.append(line)
		
		# parse 	
		bandwidth = float(str(output[-2]).split(' ')[2].replace("(", ""))
		units = str(output[-2]).split(' ')[3].replace(")", "")
		
		# conversion, MBPS
		if units == 'KB/s':
			bandwidth = bandwidth * kb_to_mb_conversion
			
		df2 = df2.append({'Bandwidth': bandwidth}, ignore_index=True)
		
		# Sleep 10 secs
		time.sleep(10)
		
	return df2
'''	

# Tests Upload, every 10 seconds 
def bandwidth_test(interval, df2):
	start_time = time.clock()
	current_time = 0
	bandwidth = 0 
	kb_to_mb_conversion = .001
	output = []

	speedtester.get_best_server()
	
	print('-------------------------------------------------------------------')
	while current_time < interval:
		
		bandwidth = float(speedtester.upload()) / 8000000 # converted from Mbps to MBps			
		df2 = df2.append({'Bandwidth': bandwidth}, ignore_index=True)
		
		print('CURRENT UPLOAD BANDWIDTH:', str(bandwidth) + ' MBps')
		# Sleep 10 secs
		time.sleep(10)
		current_time = time.clock() - start_time
	print('-------------------------------------------------------------------')	
	return df2
			

def ping_test(interval, df):
	start_time = time.clock()
	current_time = 0
	metrics = []
	latency = 0
	packet_loss = 0
	
	print('-------------------------------------------------------------------')	
	while current_time < interval:
		if USE_IPV4:
			sys_command = "ping -c 1 " + IPV4_TARGET
		else:
			sys_command = "ping6 -I "+NETWORK_INTERFACE+"-c 1 " + IPV6_TARGET	
		output = str(os.popen(sys_command).read()).split('\n\n')[1]
		output = output.split('=')
		#print('OUTPUT LENGTH:',len(output), output)
		output_metrics = output[1].split('/')
		latency_min = float(output_metrics[0])
		latency_avg = float(output_metrics[1])
		latency_max = float(output_metrics[2])
		#jitter = float(output_metrics[3])
		drops = float(output[0].split(',')[-1].split(' ')[1].split('%')[0])
		current_time = time.clock() - start_time
	
		# Parse response and place it in df
		latency = latency_avg
		packet_loss = drops
		df = df.append({'Latency':latency, 'Drops':packet_loss}, ignore_index=True)
		print('CURRENT LATENCY:', latency, '| CURRENT PACKET LOSS:', packet_loss)
	print('-------------------------------------------------------------------')		
	return df
	
if __name__ == '__main__':

	
	
	while True:
		
		# Refresh DS
		header = ['Latency', 'Drops']
		header2 = ['Bandwidth']
		header3 = ['Timestamp', 'POMB Score']
		header4 = ['Timestamp', 'POMB Rate']
		
		df = pd.DataFrame(columns=header)
		df2 = pd.DataFrame(columns=header2)
		
		# Get ping test metrics (drops, and latency) for 1 min interval
		df = ping_test(1, df)
		#print('PING FRAME\n', df)
		
		# Get ping test metrics (drops, and latency) for 1 min interval
		df2 = bandwidth_test(1, df2)
		#print('BW FRAME\n', df2)
		
		# Append CSV
		append_file(METRICS_FILE1, df)
		append_file(METRICS_FILE2, df2)
	 		
	 	# --- POMB CALCULATIONS ---#
		# Binarize Raw Data - ie. as a series bernoulli trials	
		latency_binary_data = run_binarize(df, LATENCY_THRESHOLD, 'Latency')
		drops_binary_data = run_binarize(df, DROPS_THRESHOLD, 'Drops')
		bandwidth_binary_data = run_binarize(df2, BANDWIDTH_THRESHOLD, 'Bandwidth')
		#print(latency_binary_data, drops_binary_data, bandwidth_binary_data, len(latency_binary_data), len(drops_binary_data), len(bandwidth_binary_data))
		
		# Compute Expected values
		latency_mean = np.mean(latency_binary_data)
		drops_mean = np.mean(drops_binary_data)
		bandwidth_mean = np.mean(bandwidth_binary_data)
		
		# Compute rate, if enough lags exist
		POMB_rate = (latency_mean + drops_mean + bandwidth_mean) / 3
		
		print('-------------------------------------------------------------------')
		print('BINARY LATENCY MEAN:', latency_mean, '| BINARY DROPS MEAN:', drops_mean, '| BINARY BANDWIDTH MEAN:', bandwidth_mean, '| POMB RATE', POMB_rate)
		
		#df4 = pd.DataFrame(columns=header4)
		#df4 = df3.append({'Timestamp': now, 'POMB Rate':POMB_rate}, ignore_index=True)	
		#append_file(POMB_RATES_FILE, df4)
		#POMB_score = EMA_calc2(AUTO_REGR)
		POMB_score, POMB_variance = EMA_calc(POMB_rate, AUTO_REGR)	
		
		if not POMB_score == 0: 
			# Update POMB score list
			now = totimestamp(datetime.utcnow())
			df3 = pd.DataFrame(columns=header3)
			df3 = df3.append({'Timestamp': now, 'POMB Score':POMB_score}, ignore_index=True)	
			append_file(POMB_SCORES_FILE, df3)
			
			print('-------------------------------------------------------------------')
			print('CURRENT POMB SCORE:', str(POMB_score * 100) + '%', '| MINIMUM REQUIRED POMB: ', str(MIN_POMB_REQUIRED *100)+ '%')
			
			if POMB_score < MIN_POMB_REQUIRED:
				pomb_scores.append(0) 
				print('POMB SCORE MINIMUM REQUIREMENT NOT MET OVER LAST PERIOD')
			else:
				pomb_scores.append(1)
				print('POMB SCORE MINIMUM REQUIREMENT MET')
			
	
			# --- T_50 FOR BANDWIDTH CHECK --- #
			mean_bandwidth = df2["Bandwidth"].mean() # MBps
			t_hop = float(10 / mean_bandwidth) # Time for one hop
			t_50 = SEQ_PROP_FACTOR * t_hop # Time t
			print('AVERAGE BLOCK PROPAGATION TIME FOR CURRENT BANDWIDTH:', str(t_50) +' secs', '| OPTIMAL BLOCK PROPAGATION TIME REQUIRED:', str(T_50) +' secs')# NOTE: assuming normality
			
			if t_50 > T_50:
				avg_bandwidth.append(0)
				print('BLOCK PROPAGATION TIME IS SUBOPTIMAL')
			else:
				avg_bandwidth.append(1)
				print('BLOCK PROPAGATION TIME IS OPTIMAL')
		
			# --- LATENCY CHECK --- #
			mean_latency = df["Latency"].mean()
			print('AVERAGE LATENCY:', str(mean_latency) + ' ms', '| MAX ALLOWABLE LATENCY:', str(LATENCY_THRESHOLD) + ' ms' )
			print('AVERAGE LATENCY EXCEEDS ALLOWABLE LATENCY' if mean_latency > LATENCY_THRESHOLD else 'AVERAGE LATENCY SATISFIES ALLOWABLE LATENCY')
			
			if mean_latency > LATENCY_THRESHOLD: 
				avg_latency.append(0) 
			else:
				avg_latency.append(1)
						
			# --- DROPS CHECK --- #
			mean_drops = df["Drops"].mean()
			print('AVERAGE DROPS:', str(mean_drops) + ' %', '| MAX ALLOWABLE DROPS:', str(DROPS_THRESHOLD) + ' %' )
			
			if mean_drops > DROPS_THRESHOLD:
				avg_drops.append(0)
				print('AVERAGE DROPS EXCEEDS ALLOWABLE DROPS') 
			else:
				avg_drops.append(1)
				print('AVERAGE DROPS SATISFIES ALLOWABLE DROPS')
			
			# --- MSE --- #
			#print('POMB SATISFIED:',pomb_scores, '| AVG LATENCY SATISFIED:', avg_latency, '| AVG BW SATISFIED:', avg_bandwidth)
			bandwidth_mse = run_mse(pomb_scores, avg_bandwidth)
			latency_mse = run_mse(pomb_scores, avg_latency)
			drops_mse = run_mse(pomb_scores, avg_drops)
			print('BANDWIDTH MSE:', bandwidth_mse, '| LATENCY MSE', latency_mse, '| DROPS MSE:', drops_mse)
			print('-------------------------------------------------------------------')

		'''
		# --- PREDICTION CALCULATIONS --- #
		# Latency Prediction
		sm_latency_array, cov_latency_array, data_array = run_kalman(df, header[0], sm_latency_array, cov_latency_array)
			
		plt.plot(sm_latency_array, color='red')
		plt.plot(data_array, color='blue')
		
		# Drops Prediction
		sm_drops_array, cov_drops_array, data_array = run_kalman(df, header[1], sm_drops_array, cov_drops_array)
		
		# Bandwidth Prediction
		sm_bandwidth_array, cov_bandwidth_array, data_array = run_kalman(df2, header2[0], sm_bandwidth_array, cov_bandwidth_array)
				
		plt.plot(sm_drops_array, color='red')
		plt.plot(data_array, color='blue')
		plt.show()
		
		'''