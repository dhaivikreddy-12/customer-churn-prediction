"""Main script: preprocess, train, evaluate."""
import subprocess
import sys

if __name__ == "__main__":
    subprocess.run([sys.executable, "src/preprocess.py"], check=True)
    subprocess.run([sys.executable, "src/train_model.py"], check=True)
