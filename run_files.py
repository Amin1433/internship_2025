import subprocess

commands = [
    "python eval_model.py semi_supervised_1%_phase_2",
    "python eval_model.py semi_supervised_1%_phase_3",
    "python eval_model.py semi_supervised_1%_phase_4",
    "python eval_model.py semi_supervised_1%_phase_5",
    "python eval_model.py semi_supervised_1%_phase_6",
    "python eval_model.py semi_supervised_1%_phase_7",
    "python eval_model.py semi_supervised_1%_phase_8",
    # "python eval_model.py semi_supervised_10%_phase_9",
    # "python eval_model.py semi_supervised_10%_phase_10",
    # # "python supervised.py --proportion 0.1",
    # # "python semi_supervised.py --proportion 0.1",
    # "python -m modules.self_supervised.upstream --proportion 0.1",
    # "python self_supervised.py --proportion 0.1 --freeze-encoder"

]

for cmd in commands:
    print(f"[INFO] Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"[ERROR] Command failed: {cmd}")
        break
    else:
        print(f"[INFO] Finished: {cmd}\n")
