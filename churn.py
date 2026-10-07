# Main script: load real Telco churn data, then preprocess and train.
import subprocess
import sys

if __name__ == "__main__":
    subprocess.run([sys.executable, "-m", "src.load_data"], check=True)
    subprocess.run([sys.executable, "-m", "src.train_model"], check=True)
