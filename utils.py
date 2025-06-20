# utils.py (정리본)

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from natsort import natsorted
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.spatial import distance
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import ops

# TensorFlow 설정
tf.disable_v2_behavior()
ops.reset_default_graph()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# ─────────────────────────────
# ✅ 현재 사용 중인 핵심 함수들
# ─────────────────────────────

def PositiveFFT(Input_rawdata, Sampling_frequency):
    Fs = Sampling_frequency
    T = 1 / Fs
    L = len(Input_rawdata)
    t = np.arange(0, L) * T
    Y = np.fft.fft(Input_rawdata)
    N = int(len(Y)/2 + 1)
    Y = 2 * np.abs(Y[1:N]) / L
    freq = np.linspace(0, Fs/2, N, endpoint=True)
    return Y, freq

def pro_rec_plot(s, steps):
    N = s.size
    S = np.repeat(s[None, :], N, axis=0)
    Z = np.floor(np.abs(S - S.T) / 0.01)
    Z[Z > steps] = steps
    return np.flip(Z, axis=0)

def FindRPM(data, sr):
    FFT = PositiveFFT(data, sr)
    index = list(FFT[0]).index(max(list(FFT[0])))
    return FFT[1][1:][index]

def load_single_csv(csv_path, slicing_length):
    aa = pd.read_csv(csv_path, encoding='cp949')
    L = [str(i) for i in range(np.shape(aa)[1] - 3)]
    II = ['Timestamp', 'Phase', 'Sampling_Rate'] + L
    aa.columns = II
    aa = aa.iloc[:, 0:slicing_length]
    return aa

def visualize(conv_grad):
    weights = np.mean(conv_grad, axis=(0, 1))  # alpha_k
    return weights

def Trained_Model(trained_model_path, data, label, cost, gb_grad, mode, X, Y):
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(trained_model_path)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt.model_checkpoint_path)
        _, gb_grad_value = sess.run([cost, gb_grad], feed_dict={X: data, Y: label})
        return gb_grad_value

# ─────────────────────────────
# ❌ 현재 사용되지 않는 함수들 (보존용)
# ─────────────────────────────
# def flatten(xss):
#     return [x for xs in xss for x in xs]

# def MAF(interval, window_size):
#     window = np.ones(int(window_size)) / float(window_size)
#     return np.convolve(interval, window, 'same')

# def moving_average_filtering(interval, window_size):
#     window = np.ones(int(window_size)) / float(window_size)
#     return np.convolve(interval, window, 'same')

# def CSVDataLoader(data_path, slicing_length):
#     data_folder_list = natsorted(os.listdir(data_path))
#     if '.ipynb_checkpoints' in data_folder_list:
#         data_folder_list.remove('.ipynb_checkpoints')
#     data = []
#     for temp in tqdm(data_folder_list):
#         path = os.path.join(data_path, temp)
#         aa = pd.read_csv(path, encoding='cp949')
#         L = [str(i) for i in range(np.shape(aa)[1] - 3)]
#         aa.columns = ['Timestamp', 'Phase', 'Sampling_Rate'] + L
#         data.append(aa)
#     raw_data = pd.concat(data, ignore_index=True)
#     raw_data = raw_data.iloc[:, 0:slicing_length]
#     return raw_data
