import subprocess
import sys
import os

def main():
    csv_path = os.path.join("outputs", "submission.csv")
    validator_path = os.path.join("..", "India_runs_data_and_ai_challenge", "validate_submission.py")
    res = subprocess.run([sys.executable, validator_path, csv_path])
    sys.exit(res.returncode)

if __name__ == "__main__":
    main()
