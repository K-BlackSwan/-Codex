# pipeline_controller.py (with realtime prediction option)

import json
import os
import time
from datetime import datetime

from run_cycle import run_cycle
from rp_generator import run_rp_and_train
from model_trainer import train_model
from model_tester import test_model, predict_one_point


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def apply_environment_variables(config):
    os.environ["CURRENT_HZ_LABEL"] = config.get("hz_label", "50Hz")
    today = datetime.now().strftime("%Y-%m-%d")
    os.environ["TRAIN_START_DATE"] = today
    os.environ["TRAIN_END_DATE"] = today


def start_pipeline(config_path="config.json"):
    print("🔧 [1] 설정 로딩 중...")
    config = load_config(config_path)

    apply_environment_variables(config)

    print("\n📡 [2] 데이터 수집 시작...")
    run_cycle(
        daq_ip=config["daq_ip"],
        recording_seconds=config["recording_seconds"],
        waiting_seconds=config["waiting_seconds"],
        total_minutes=config["total_run_minutes"]
    )

    print("\n🌀 [3] RP 이미지 생성 + 학습 시작...")
    run_rp_and_train(
        hz_label=config["hz_label"],
        slicing_length=config["slicing_length"],
        sampling_rate=config["sampling_rate"],
        rp_threshold=config["rp_threshold"],
        scale_factor=config["scale_factor"],
        max_test=config["max_test_images"]
    )

    print("\n📚 [4] CNN + MD 모델 학습 시작...")
    train_model(
        learning_rate=config["cnn_learning_rate"],
        batch_size=config["cnn_batch_size"],
        epochs=config["cnn_epochs"],
        mode=config["mode"]
    )

    if config.get("auto_start_prediction", True):
        print("\n🔎 [5] 테스트 모델 실행...")
        test_model()

    if config.get("predict_one_loop", False):
        print("\n🔁 [6] 실시간 단일 포인트 예측...")
        result = predict_one_point()
        print(f"✅ 예측 결과(MD): {result}")

    print("\n✅ 전체 자동화 파이프라인 완료!")


if __name__ == "__main__":
    start_pipeline("config.json")
