"""Run every experiment end-to-end and save all figures into figures/."""
import subprocess
import shutil
import os

os.makedirs("figures", exist_ok=True)

subprocess.run(["python3", "experiments_1_2_3.py"], check=True)
subprocess.run(["python3", "experiment_4.py"], check=True)

for f in ["experiment_1.png", "experiment_2.png", "experiment_3.png", "experiment_4.png"]:
    if os.path.exists(f):
        shutil.move(f, os.path.join("figures", f))

print("All figures saved to figures/")
