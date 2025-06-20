from run_cycle import run_cycle
from rp_generator import run_rp_and_train
from model_trainer import train_model
from model_tester import test_model

def main():
    run_cycle()
    run_rp_and_train()
    train_model()
    test_model()

if __name__ == "__main__":
    main()
