"""One command to connect to the Tufts cluster and check that MuJoCo works there.

    python3 setup.py              # everything
    python3 setup.py --skip-gpu   # no GPU job
    python3 setup.py --reinstall  # rebuild ~/mujoco-env on the cluster

Run it from this folder, on your laptop (off campus: connect the VPN first). It prints
PASS/FAIL for each step and writes everything to logs/setup-<time>.log. If a step fails,
send that log file to whoever is helping you.
"""
import argparse
import socket
import sys

from hpc import HERE, Cluster, StepFailed

FILES = ["02_check_cluster.sh", "03_setup_env.sh", "04_test_mujoco.py", "05_job.sbatch", "06_gpu_job.sbatch",
         "08_demo.py", "08_demo.sbatch"]
results = []


def record(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}: {name} {detail}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-gpu", action="store_true")
    ap.add_argument("--reinstall", action="store_true", help="delete and rebuild ~/mujoco-env on the cluster")
    ap.add_argument("--cpu-partition", default="batch")
    ap.add_argument("--gpu-partition", default="gpu")
    args = ap.parse_args()

    c = Cluster("setup")
    print(f"log file: {c.logpath}\n")
    try:
        print("1. Network")
        try:
            socket.getaddrinfo(c.host, 22)
            socket.create_connection((c.host, 22), timeout=5).close()
            record("can reach " + c.host, True)
        except OSError as e:
            record("can reach " + c.host, False, f"({e}) off campus? connect the Tufts VPN")
            c.say("  continuing anyway; the next step is the real test")

        print("2. Log in")
        c.connect()
        _, out = c.ssh("echo $(whoami) on $(hostname)", show=False)
        record("ssh login", True, out.strip())

        print("3. Copy files to the cluster")
        c.push([HERE / f for f in FILES if (HERE / f).exists()], c.dir)
        record("copy files", True, f"-> {c.dir}")

        print("4. Survey the cluster (slurm, partitions, python, disk)")
        _, out = c.ssh(f"cd {c.dir} && bash 02_check_cluster.sh", show=True)
        record("slurm available", "MISSING: sbatch" not in out)
        record("pip can reach the internet", "FAIL: cannot reach pypi" not in out)

        print("5. MuJoCo environment (~/mujoco-env)")
        have_env = False
        if not args.reinstall:
            rc, _ = c.ssh("~/mujoco-env/bin/python -c 'import mujoco, imageio_ffmpeg'", check=False, show=False)
            have_env = rc == 0
        if have_env:
            record("environment already installed", True, "(use --reinstall to rebuild)")
        else:
            c.ssh(f"cd {c.dir} && bash 03_setup_env.sh")
            record("environment installed", True)

        print("6. Quick test on the login node")
        _, out = c.ssh(f"cd {c.dir} && source ~/mujoco-env/bin/activate && python 04_test_mujoco.py", check=False)
        record("mujoco physics on login node", "PASS physics" in out)

        print("7. CPU job through Slurm")
        run_job(c, "05_job.sbatch", "mujoco-test", args.cpu_partition, gpu=False)

        if args.skip_gpu:
            print("8. GPU job: skipped")
        else:
            print("8. GPU job through Slurm (can wait in the queue)")
            run_job(c, "06_gpu_job.sbatch", "mujoco-gpu", args.gpu_partition, gpu=True)
    except StepFailed as e:
        record("stopped", False, str(e))
    except KeyboardInterrupt:
        c.say("interrupted")
    finally:
        bad = [r for r in results if not r[1]]
        print("\n==== Summary ====")
        for name, ok, detail in results:
            print(f"{'PASS' if ok else 'FAIL'}  {name}")
        print("\nAll good. Next: python3 runIt.py --gpu 08_demo.py" if not bad else "\nSomething failed.")
        print(f"Full log: {c.logpath}")
    sys.exit(1 if [r for r in results if not r[1]] else 0)


def run_job(c, sbatch_file, prefix, partition, gpu):
    jobid = c.submit(c.dir, sbatch_file, f"--partition={partition}")
    res = c.wait(jobid)
    text = c.read_remote(f"{c.dir}/{prefix}-{jobid}.out")
    print("    " + text.replace("\n", "\n    ").rstrip())
    record("job completed", res["state"] == "COMPLETED", f"({res['state']} on {res['node']})")
    record("physics on compute node", "PASS physics" in text)
    if gpu:
        record("GPU allocated", "GPU 0:" in text)
        record("EGL rendering", "PASS render via egl" in text, "(optional)" if "PASS render" not in text else "")


if __name__ == "__main__":
    main()
