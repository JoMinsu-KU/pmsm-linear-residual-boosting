"""Supervise the standalone study pipeline with a hard invocation time limit."""
from pathlib import Path
import argparse,json,os,subprocess,sys

ROOT=Path(__file__).resolve().parents[1]
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[name]="1"
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
sys.path.insert(0,str(ROOT/"src"))

def main():
    # Imports are local and pinned; do not import historical orchestration.
    from pmsm_repro.pipeline import parser
    parsed=parser().parse_args()
    if not 0<parsed.max_hours<=120:raise ValueError("Finite positive --max-hours <=120 required")
    if os.environ.get("PMSM_REPRO_CHILD")=="1":
        from pmsm_repro.pipeline import main as run
        return run()
    env=os.environ.copy();env["PMSM_REPRO_CHILD"]="1"
    process=subprocess.Popen([sys.executable,"-u","-X","utf8","-B",str(Path(__file__).resolve()),*sys.argv[1:]],env=env)
    try:
        return process.wait(timeout=parsed.max_hours*3600)
    except (subprocess.TimeoutExpired,KeyboardInterrupt):
        process.terminate()
        try:process.wait(timeout=15)
        except subprocess.TimeoutExpired:process.kill();process.wait()
        print("Invocation stopped; verified completed stages remain resumable.",file=sys.stderr)
        return 124
    finally:
        lock=parsed.workdir.resolve()/".active.lock"
        if process.poll() is not None and lock.is_file():
            try:
                if json.loads(lock.read_text())["pid"]==process.pid:lock.unlink()
            except (OSError,ValueError,KeyError):pass

if __name__=="__main__":raise SystemExit(main())
