"""Compare the instrumented engine with the original, committed implementation."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kmc_test import DislocationKMC

REFERENCE = "a7e4325e55da73c5c8cfe8f198cfd65c9121bead"
source = subprocess.check_output(["git", "show", f"{REFERENCE}:kmc_test.py"], cwd=ROOT, text=True)
namespace = {"__name__": "legacy_baseline"}
exec(compile(source, "legacy_kmc_test.py", "exec"), namespace)
cases = []
for temperature, stress in [(300, 0), (300, 10), (300, -10), (300, 100), (300, 200), (600, 10)]:
    np.random.seed(454)
    old = namespace["DislocationKMC"](temp_k=temperature, stress_mpa=stress, num_segments=24)
    expected = old.run_simulation(2000)
    new = DislocationKMC(temp_k=temperature, stress_mpa=stress, num_segments=24, seed=454)
    actual = new.run_simulation(2000, record_events=True)
    assert expected == actual
    np.testing.assert_array_equal(old.positions, new.positions)
    np.testing.assert_array_equal(old.time_history, new.time_history)
    np.testing.assert_array_equal(old.mean_position_history, new.mean_position_history)
    cases.append({"temperature_k": temperature, "stress_mpa": stress, "passed": True})
report = {"reference_commit": REFERENCE, "reference_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
          "criterion": "Exact equality of all times, all mean positions, final segment positions, and metrics",
          "num_segments": 24, "events_per_run": 2000, "seed": 454, "cases": cases, "passed": True}
destination = ROOT/"outputs"/"v0_baseline"/"legacy_regression.json"
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(report, indent=2)+"\n")
print(f"Original-code comparison passed: {len(cases)} cases, exact equality throughout")
