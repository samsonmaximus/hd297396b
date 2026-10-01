"""Remaining work, ordered so that the paper's pending numbers resolve first."""
import subprocess, time, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
env = dict(os.environ, ADOPT="see_serval_in_gauss_1p1c_power",
           ADOPT0P="see_serval_in_gauss_0p_power")
STEPS = [
    (["python3", "-u", "ecc_v1.py"], "logs/ecc_v1.log"),
    (["python3", "-u", "verify_v1.py"], "logs/verify.log"),
    (["python3", "-u", "injrec_gp.py"], "logs/injrec.log"),
    (["python3", "-u", "adversarial.py"], "logs/adversarial.log"),
]
with open("logs/queue.log", "a") as q:
    for cmd, log in STEPS:
        q.write(f"START {cmd[2]} {time.ctime()}\n"); q.flush()
        with open(log, "a") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        q.write(f"DONE {cmd[2]} {time.ctime()}\n"); q.flush()
    q.write(f"ALL DONE {time.ctime()}\n")
