# rp_generator.py

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from natsort import natsorted
import subprocess
import sys

from utils import FindRPM, pro_rec_plot, load_single_csv

# ─────────────────────────────────────────────
def generate_rp_from_raw(
    raw_dir,
    rp_base_dir,
    test_dir,
    slicing_length=16000,
    sampling_rate=7680,
    rp_threshold=100,
    scale_factor=0.7,
    max_test=50
):
    os.makedirs(rp_base_dir, exist_ok=True)
    original_dir = os.path.join(rp_base_dir, "0_original")
    scaled_dir = os.path.join(rp_base_dir, "1_scaled")
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(scaled_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    file_list = natsorted(os.listdir(raw_dir))
    normal_saved = scaled_saved = test_saved_normal = test_saved_scaled = 0

    for fname in file_list:
        if not fname.endswith("_R.csv"):
            continue

        full_path = os.path.join(raw_dir, fname)
        try:
            df = load_single_csv(full_path, slicing_length)
            r_rows = df[df.iloc[:, 1].astype(str).str.contains("R상")]
            input_data = r_rows.iloc[:, 4:].to_numpy(dtype='float64').flatten()

            freq = FindRPM(input_data, sampling_rate)
            if freq < 1:
                continue

            samples_per_2cycle = int((sampling_rate / freq) * 2)
            stride = samples_per_2cycle
            total_len = len(input_data)

            for start_idx in range(0, total_len - samples_per_2cycle + 1, stride):
                sliced_data = input_data[start_idx:start_idx + samples_per_2cycle]

                RP = pro_rec_plot(sliced_data, rp_threshold)
                plt.imsave(os.path.join(original_dir, f"0_rp_{normal_saved + 1}.png"), RP)
                normal_saved += 1

                if test_saved_normal < max_test:
                    plt.imsave(os.path.join(test_dir, f"test_normal_{test_saved_normal + 1}.png"), RP)
                    test_saved_normal += 1

                RP_scaled = pro_rec_plot(sliced_data * scale_factor, rp_threshold)
                plt.imsave(os.path.join(scaled_dir, f"1_rp_scaled_{scaled_saved + 1}.png"), RP_scaled)
                scaled_saved += 1

                if test_saved_scaled < max_test:
                    plt.imsave(os.path.join(test_dir, f"test_scaled_{test_saved_scaled + 1}.png"), RP_scaled)
                    test_saved_scaled += 1

        except Exception as e:
            print(f"⚠️ 오류 발생: {fname} → {e}")

    print(f"✅ 원본 RP: {normal_saved}, 스케일 RP: {scaled_saved}")
    print(f"✅ 테스트 저장 - 원본: {test_saved_normal}, 스케일: {test_saved_scaled}")

# ─────────────────────────────────────────────
def run_rp_and_train(hz_label=None, use_manual_date=False):
    base_dir = os.getcwd()
    today = datetime.now().strftime("%Y-%m-%d")
    hz = hz_label or os.environ.get("CURRENT_HZ_LABEL", "50Hz")

    raw_dir = os.path.join(base_dir, "Predict_maintenance", "Realtime", "Raw_data", f"{today}_{hz}")
    rp_base_dir = os.path.join(base_dir, "Predict_maintenance", "Realtime", "RP_data", f"{today}_{hz}")
    test_dir = os.path.join(base_dir, "Predict_maintenance", "Realtime", "Test_data", f"2_test_{today}_{hz}")

    generate_rp_from_raw(raw_dir, rp_base_dir, test_dir)

    # 날짜 설정 (환경변수)
    if use_manual_date:
        train_start, train_end = "2025-05-07", "2025-05-07"
    else:
        train_start = train_end = today

    os.environ["TRAIN_START_DATE"] = train_start
    os.environ["TRAIN_END_DATE"] = train_end
    os.environ["CURRENT_HZ_LABEL"] = hz

    # CNN 학습 실행
    train_script = os.path.join(base_dir, "Train_by_date_full.py")
    if os.path.exists(train_script):
        print("📦 학습 실행 중...")
        try:
            subprocess.run([sys.executable, train_script], check=True)
            print("✅ 학습 완료")
        except subprocess.CalledProcessError as e:
            print(f"❌ 학습 오류: {e}")
    else:
        print("⚠️ 학습 스크립트가 없습니다.")

# ─────────────────────────────────────────────
if __name__ == "__main__":
    run_rp_and_train()
