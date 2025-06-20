# data_collector.py

import modbus_tk.modbus_tcp as modbus_tcp
import modbus_tk.defines as cst
import time
import logging
import os
import json
import struct
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# import csvfile_ver2 as csvfile  # 기존 이름 그대로 유지

# ─────────────────────────────────────────────
# 로깅 및 환경 초기화
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, 'modbus_client.log')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(log_file, encoding='utf-8')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# ─────────────────────────────────────────────
# 설정 불러오기
def load_config():
    default_config = {"daq_ip": "192.168.30.110", "daq_port": 502}
    config_path = "./config.json"
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ config.json 로딩 실패: {e}")
    with open(config_path, "w") as f:
        json.dump(default_config, f)
    return default_config

# ─────────────────────────────────────────────
# 전류 보정 계산 함수
def Calcuator(current_max, slope, slopebase, measuredvalue, intercept):
    ratio = 1
    if current_max >= 6000:
        ratio = 16
    elif current_max >= 600:
        ratio = 8
    elif current_max >= 6:
        ratio = 4
    return ((((slope * ratio / slopebase) * measuredvalue) / 10000 + intercept) / 10)

# ─────────────────────────────────────────────
# 데이터 수신 함수
def fetch_raw_data(client, switch):
    rawData = []
    float_list = []

    for address in range(2099, 17459, 64):
        time.sleep(0.01)
        registers = client.execute(1, cst.READ_HOLDING_REGISTERS, address, 64)
        rawData.extend(registers)

    # 센서 설정값 추출
    if switch == 0:
        slope, intercept = client.execute(1, cst.READ_HOLDING_REGISTERS, 0, 2)
        current_max = client.execute(1, cst.READ_HOLDING_REGISTERS, 451, 1)[0] / 100
    elif switch == 1:
        slope, intercept = client.execute(1, cst.READ_HOLDING_REGISTERS, 3, 2)
        current_max = client.execute(1, cst.READ_HOLDING_REGISTERS, 452, 1)[0] / 100
    else:
        slope, intercept = client.execute(1, cst.READ_HOLDING_REGISTERS, 6, 2)
        current_max = client.execute(1, cst.READ_HOLDING_REGISTERS, 453, 1)[0] / 100

    for i in range(0, len(rawData), 2):
        upper_bytes = struct.pack('>H', rawData[i])
        lower_bytes = struct.pack('>H', rawData[i + 1])
        float_value = lower_bytes + upper_bytes
        raw_int = struct.unpack('>l', float_value)[0]
        current = Calcuator(current_max, slope, 10000, raw_int, intercept)
        float_list.append(current)

    csvfile.dbsave(switch, float_list)
    print(f"✅ R/S/T 상 {switch}번 데이터 저장 완료")

# ─────────────────────────────────────────────
# 메인 측정 함수
def run_modbus_client(recording_seconds=60):
    # config = load_config()
    # host = config.get("daq_ip", "192.168.30.110")
    # port = config.get("daq_port", 502)
    #
    # client = modbus_tcp.TcpMaster(host=host, port=port, timeout_in_sec=10)
    #
    # try:
    #     client.open()
    #     print(f"🔌 Modbus 연결됨 ({host}:{port})")
    #
    #     switch = 0
    #     sendflag = 0
    #     start_time = time.time()
    #
    #     while (time.time() - start_time) < recording_seconds:
    #         time.sleep(1)
    #
    #         registers = client.execute(1, cst.WRITE_SINGLE_REGISTER, 718, output_value=7805)
    #         if not registers:
    #             print("⚠️ 시작 신호 실패")
    #             continue
    #
    #         for _ in range(3):  # R/S/T 상
    #             phase_code = {0: 3, 1: 2, 2: 1}[switch]
    #             client.execute(1, cst.WRITE_SINGLE_REGISTER, 1501, output_value=phase_code)
    #             client.execute(1, cst.WRITE_SINGLE_REGISTER, 1500, output_value=1)
    #
    #             time.sleep(2)
    #             fetch_raw_data(client, switch)
    #             switch = (switch + 1) % 3
    #
    # except Exception as e:
    #     logging.error(f"Modbus 오류: {e}")
    #     print(f"❌ Modbus 오류: {e}")
    # finally:
    #     try:
    #         client.close()
    #         print("🔌 Modbus 연결 종료")
    #     except:
    #         pass
    def run_modbus_client(recording_seconds=60):
        # [1] 측정 대신 더미 데이터 생성
        print("⚠️ [시뮬레이션 모드] 실제 장비 연결 없이 더미 데이터 생성 중...")

        raw_dir = "./Predict_maintenance/Realtime/Raw_data"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_dir = os.path.join(raw_dir, timestamp)
        os.makedirs(save_dir, exist_ok=True)

        # [2] R/S/T 상 더미 CSV 생성
        for i, phase in enumerate(['R', 'S', 'T']):
            data = np.sin(np.linspace(0, 10 * np.pi, 16000)) * (1 + 0.1 * i)  # 약간 변형
            times = np.linspace(0, 16000 / 7680, 16000)
            dummy_csv = np.vstack([times, data]).T
            np.savetxt(os.path.join(save_dir, f"ch{i + 1}.csv"), dummy_csv, delimiter=",", header="time,current",
                       comments='')

        print(f"✅ [시뮬레이션] {save_dir} 에 더미 전류파형 저장 완료")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    run_modbus_client(60)
