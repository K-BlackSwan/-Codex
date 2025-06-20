# model_tester.py (업데이트됨)

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.spatial import distance
from natsort import natsorted
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import ops

from utils import visualize, Trained_Model, FindRPM, pro_rec_plot, load_single_csv

tf.disable_v2_behavior()

resize_size = (200, 200)
mode = "cpu"
class_number = 2


def test_model():
    # ... (기존 test_model 함수 동일 - 생략)
    pass


def predict_one_point():
    hz = os.environ.get("CURRENT_HZ_LABEL", "50Hz")
    base_dir = os.getcwd()
    today = datetime.now().strftime("%Y-%m-%d")

    raw_dir = os.path.join(base_dir, "Predict_maintenance", "Realtime", "Raw_data", f"{today}_{hz}")
    model_dir = os.path.join(base_dir, "Predict_maintenance", "Realtime", "models", f"model_{hz}")

    # 최신 CSV 파일 찾기 (R상)
    r_files = [f for f in os.listdir(raw_dir) if f.endswith("_R.csv")]
    if not r_files:
        print("❌ R상 CSV 없음")
        return None

    latest_file = sorted(r_files)[-1]
    csv_path = os.path.join(raw_dir, latest_file)
    df = load_single_csv(csv_path, slicing_length=16000)
    r_data = df[df.iloc[:, 1].astype(str).str.contains("R상")].iloc[:, 4:].to_numpy(dtype='float64').flatten()

    freq = FindRPM(r_data, 7680)
    if freq < 1:
        print("❌ 유효 주파수 아님")
        return None

    samples_per_2cycle = int((7680 / freq) * 2)
    if len(r_data) < samples_per_2cycle:
        print("❌ 데이터 길이 부족")
        return None

    # RP 변환
    sliced = r_data[:samples_per_2cycle]
    RP = pro_rec_plot(sliced, 100)
    RP = Image.fromarray(RP).resize(resize_size, Image.ANTIALIAS)
    RP = np.array(RP).reshape([-1, resize_size[0], resize_size[1], 1])

    label = np.array([[1.0, 0.0]])  # normal 가정

    # 모델 로딩
    tf.reset_default_graph()
    with tf.device("/device:GPU:0" if mode == 'gpu' else "/device:CPU:0"):
        X = tf.placeholder(tf.float32, [None, resize_size[0], resize_size[1], 1])
        Y = tf.placeholder(tf.float32, [None, class_number])

        conv1 = tf.layers.conv2d(X, 4, [5, 5], padding='same', use_bias=False)
        relu1 = tf.nn.relu(conv1)
        conv2 = tf.layers.conv2d(relu1, 12, [3, 3], padding='same', use_bias=False)
        relu2 = tf.nn.relu(conv2)
        gap = tf.reduce_mean(relu2, [1, 2], keep_dims=True)
        flat = tf.reshape(gap, [-1, int(np.prod(gap.shape[1:]))])
        fc1 = tf.layers.dense(flat, class_number)
        fc2 = tf.layers.dense(fc1, class_number)
        logits = fc2

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
        gb_grad = tf.gradients(cost, conv2)[0]
        gb_grad = tf.div(gb_grad, tf.sqrt(tf.reduce_mean(tf.square(gb_grad))) + 1e-5)

    grad_val = Trained_Model(os.path.join(model_dir, "CNN_model"), RP, label, cost, gb_grad, mode, X, Y)
    weight = visualize(grad_val[0])

    mean = np.load(os.path.join(model_dir, "MD_model", "mean.npy"))
    inv = np.load(os.path.join(model_dir, "MD_model", "inv.npy"))

    md = distance.mahalanobis(weight, mean, inv)
    return float(md)


if __name__ == "__main__":
    print("📍 실시간 1포인트 MD 결과:", predict_one_point())
