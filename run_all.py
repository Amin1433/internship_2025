import subprocess

commands = [
    "python semi_supervised.py --proportion 0.1",
    "python semi_supervised.py --proportion 0.01",
    "python semi_supervised.py --proportion 0.05",
    "python weakly.py --proportion 0.1",
    "python weakly.py --proportion 0.01",
    "python weakly.py --proportion 0.05",
]

for cmd in commands:
    print(f"[INFO] Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"[ERROR] Command failed: {cmd}")
        break
    else:
        print(f"[INFO] Finished: {cmd}\n")
