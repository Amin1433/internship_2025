import subprocess

commands = [
    "python semi_supervised.py --proportion 0.01",
    "python semi_supervised.py --proportion 0.05",
    "python semi_supervised.py --proportion 0.1"
]

for cmd in commands:
    print(f"[INFO] Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"[ERROR] Command failed: {cmd}")
        break
    else:
        print(f"[INFO] Finished: {cmd}\n")
