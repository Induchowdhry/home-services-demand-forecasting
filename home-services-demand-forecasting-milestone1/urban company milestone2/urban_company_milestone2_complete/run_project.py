"""Run the full Milestone 2 workflow from the project root."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for script in ["src/data_pipeline.py", "src/train_baseline.py"]:
    print(f"\n>>> Running {script}")
    subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, check=True)
print("\nMilestone 2 pipeline completed.")
