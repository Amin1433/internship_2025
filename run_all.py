import subprocess

commands = [
    # "python supervised.py --proportion 0.1",
    "python semi_supervised.py --proportion 0.1",
    "python -m modules.self_supervised.downstream --proportion 0.1",
    "python self_supervised.py --proportion 0.1 --freeze-encoder"

]

for cmd in commands:
    print(f"[INFO] Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"[ERROR] Command failed: {cmd}")
        break
    else:
        print(f"[INFO] Finished: {cmd}\n")
