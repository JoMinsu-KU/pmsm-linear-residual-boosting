"""Run the repository's focused numerical and pipeline checks."""
from pathlib import Path
import os,sys,unittest

ROOT=Path(__file__).resolve().parents[1]
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[name]="1"
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/"src"))
if __name__=="__main__":
    suite=unittest.defaultTestLoader.discover(str(ROOT/"tests"))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
