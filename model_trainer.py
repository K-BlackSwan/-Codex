# model_trainer.py

import os
import numpy as np
from datetime import datetime, timedelta
from PIL import Image
import cv2
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import tensorflow.compat.v1 as tf
from utils import visualize, Trained_Model

tf.disable_v2_behavior()

# ▶ config
resize_size = (200, 200)
data_interval = 5
learningRate = 0.001
train_batch_size = 100
training_epochs = 1000
mode = "gpu"

# ▶ 날짜 범위 설정
def get_date_range(start_date, end_date):
    date_list = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    while current <= end:
        date_list.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return date_list

def train_model():
    base_dir = os.getcwd()
    hz_label = os.environ.get("CURRENT_HZ_LABEL", "50Hz")
    start_date = os.environ.get("TRAIN_START_DATE", datetime.now().strftime("%Y-%m-%d"))
    end_date = os.environ.get("TRAIN_END_DATE", start_date)

    dates = get_date_range(start_date, end_date)
    folders = ["0_original", "1_scaled"]

    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(label_encoder.fit_transform(folders).reshape(-1, 1))

    train_data, train_label = [], []
    for date in dates:
        for idx, folder in enumerate(folders):
            img_dir = os.path.join(base_dir, "Predict_maintenance", "Realtime", "RP_data", f"{date}_{hz_label}", folder)
            if not os.path.exists(img_dir):
                continue
            for img_name in os.listdir(img_dir):
                if img_name.lower().endswith(('.png', '.jpg')):
                    try:
                        img = Image.open(os.path.join(img_dir, img_name)).convert("L")
                        img = cv2.resize(np.array(img), resize_size, cv2.INTER_AREA)
                        train_data.append([img])
                        train_label.append([onehot_encoded[idx]])
                    except:
                        continue

    train_data = np.array(train_data)
    train_label = np.array(train_label)

    num, _, col, row = train_data.shape
    _, _, class_number = train_label.shape
    train_data = np.reshape(train_data, (-1, row, col, 1))[::data_interval]
    train_label = np.reshape(train_label, (-1, class_number))[::data_interval]

    tf.reset_default_graph()
    device = "/device:GPU:0" if mode == "gpu" else "/device:CPU:0"

    with tf.device(device):
        X = tf.placeholder(tf.float32, [None, resize_size[0], resize_size[1], 1])
        Y = tf.placeholder(tf.float32, [None, class_number])

        conv1 = tf.layers.conv2d(X, 4, [5, 5], padding='same', use_bias=False)
        relu1 = tf.nn.relu(conv1)
        conv2 = tf.layers.conv2d(relu1, 12, [3, 3], padding='same', use_bias=False)
        relu2 = tf.nn.relu(conv2)
        gap = tf.reduce_mean(relu2, [1, 2], keep_dims=True)
        flat = tf.reshape(gap, [-1, np.prod(gap.shape[1:])])
        fc1 = tf.layers.dense(flat, class_number)
        fc2 = tf.layers.dense(fc1, class_number)
        logits = fc2

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
        optimizer = tf.train.AdamOptimizer(learningRate).minimize(cost)
        prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
        gb_grad = tf.gradients(cost, conv2)[0]
        gb_grad = tf.div(gb_grad, tf.sqrt(tf.reduce_mean(tf.square(gb_grad))) + 1e-5)

    model_dir = os.path.join(base_dir, "Predict_maintenance", "Realtime", "models", f"model_{hz_label}")
    os.makedirs(os.path.join(model_dir, "CNN_model"), exist_ok=True)
    os.makedirs(os.path.join(model_dir, "MD_model"), exist_ok=True)

    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    train_total_batch = max(1, len(train_data) // train_batch_size)
    for epoch in range(training_epochs):
        acc, loss = 0, 0
        for i in range(train_total_batch):
            start = i * train_batch_size
            end = (i + 1) * train_batch_size
            x_batch = train_data[start:end]
            y_batch = train_label[start:end]

            sess.run(optimizer, feed_dict={X: x_batch, Y: y_batch})
            batch_loss, batch_acc = sess.run([cost, accuracy], feed_dict={X: x_batch, Y: y_batch})
            acc += batch_acc
            loss += batch_loss

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Cost: {loss/train_total_batch:.6f}, Accuracy: {acc/train_total_batch:.6f}")
        if loss/train_total_batch < 0.0003:
            print("Early Stopping.")
            break

    saver.save(sess, os.path.join(model_dir, "CNN_model", "LeNet.ckpt"))
    print("✅ CNN 모델 저장 완료")

    normal_data = train_data[:len(train_data)//2]
    normal_label = train_label[:len(train_label)//2]

    gb_grad_value = Trained_Model(os.path.join(model_dir, "CNN_model"), normal_data, normal_label, cost, gb_grad, mode, X, Y)
    weights = [visualize(g) for g in gb_grad_value]
    MD_data = np.array(weights)
    mean = np.mean(MD_data, axis=0)
    inv_cov = np.linalg.pinv(np.cov(MD_data.T))

    np.save(os.path.join(model_dir, "MD_model", "mean.npy"), mean)
    np.save(os.path.join(model_dir, "MD_model", "inv.npy"), inv_cov)

    print("✅ MD 모델 저장 완료")

    # 후처리 테스트
    subprocess.run([sys.executable, "Test_TF1.py"], check=True)

if __name__ == "__main__":
    train_model()
