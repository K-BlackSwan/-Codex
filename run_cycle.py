# run_cycle.py

import os
import time
import subprocess
from datetime import datetime
import pandas as pd
import numpy as np
import data_collector  # 내부에 run_modbus_client 포함

# ─────────────────────────────────────────────
# 주파수 추정 함수 (Zero-crossing 기반)
def find_rpm_from_csv(csv_path, sampling_rate=7680):
    df = pd.read_csv(csv_path, header=None)
    data = df.iloc[1:, 1].astype(float).to_numpy()
    data = data - np.mean(data)

    zero_crossings = np.where(np.diff(np.sign(data)))[0]
    if len(zero_crossings) < 2:
        return 0

    intervals = np.diff(zero_crossings) / sampling_rate
    average_period = np.mean(intervals) * 2  # full sine wave 기준
    rpm = 60 / average_period
    return rpm

# ─────────────────────────────────────────────
# 전체 수집 루프
def run_cycle():
    print("\n📋 [AI 예지보전 자동 측정 루프 시작]\n")

    # 사용자 입력
    daq_ip = input("📡 DAQ 장비 IP 주소 (기본: 192.168.30.110): ") or "192.168.30.110"
    recording_minutes = int(input("⏱️  측정 시간 (분): "))
    waiting_minutes = int(input("🕒  대기 시간 (분): "))
    total_minutes = int(input("⏲️  전체 실행 시간 (분): "))

    recording_seconds = recording_minutes * 60
    waiting_seconds = waiting_minutes * 60
    total_seconds = total_minutes * 60

    # 환경설정
    current_dir = os.getcwd()
    raw_data_dir = os.path.join(current_dir, "Raw_data")
    os.makedirs(raw_data_dir, exist_ok=True)

    # config.json에 IP 주소 반영
    config_path = "./config.json"
    if os.path.exists(config_path):
        import json
        with open(config_path, "r") as f:
            config = json.load(f)
        config["daq_ip"] = daq_ip
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    # ───────────── Step 1: 샘플 수집 후 주파수 추정 ─────────────
    print("\n🎯 샘플 수집 및 주파수 추정 중...")

    sample_path = os.path.join(raw_data_dir, "sample.csv")
    data_collector.run_modbus_client(10)  # 10초 샘플 측정
    os.rename(sorted(os.listdir(raw_data_dir))[-1], sample_path)  # 가장 최근 파일을 샘플로

    try:
        rpm = find_rpm_from_csv(sample_path)
        hz_label = f"{int(round(rpm / 60))}Hz"
        print(f"📡 추정 주파수: {hz_label}")
    except Exception as e:
        print(f"⚠️ 주파수 추정 실패: {e}")
        hz_label = "UnknownHz"

    try:
        os.remove(sample_path)
    except:
        pass

    os.environ["CURRENT_HZ_LABEL"] = hz_label

    # ───────────── Step 2: 반복 수집 루프 ─────────────
    today = datetime.now().strftime("%Y-%m-%d")
    today_folder = os.path.join(raw_data_dir, f"{today}_{hz_label}")
    os.makedirs(today_folder, exist_ok=True)

    print(f"\n🚀 측정 시작! 총 {total_minutes}분 동안 실행됩니다.\n")

    start_time = time.time()
    cycle_count = 1

    while (time.time() - start_time) < total_seconds:
        print(f"\n[{cycle_count}번째 사이클] 데이터 수집 중...")
        data_collector.run_modbus_client(recording_seconds)
        print(f"[{cycle_count}번째 사이클] 수집 완료 ✅ → {waiting_minutes}분 대기...")
        time.sleep(waiting_seconds)
        cycle_count += 1

    print("\n🎉 모든 데이터 수집 완료! RP 변환을 시작합니다...")

    # ───────────── Step 3: RP 이미지 자동 변환 ─────────────
    rp_script_path = os.path.join(current_dir, "generate_rp_ncycle_dirs_folderprefix.py")
    if os.path.exists(rp_script_path):
        subprocess.run(["python3", rp_script_path])
    else:
        print("⚠️ RP 변환 스크립트가 없습니다. 스킵합니다.")

# ─────────────────────────────────────────────
if __name__ == "__main__":
    run_cycle()
