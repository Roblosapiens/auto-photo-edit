"""Verify the prediction pipeline runs fully offline.

Usage: python tests/test_offline.py <request.json>

Runs src/predict.py twice on the same request (with cropping and straightening, so CLIP, YOLO pose and
GeoCalib all load):
  1. audit mode   - the libraries' normal behaviour; every network attempt is recorded, nothing blocked
  2. offline mode - the default: offline switches + network guard; any attempt would be blocked
Passes if the offline run succeeds with zero network attempts and produces the same settings as the
audit run.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def run(request, out, extra_env):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("HF_", "YOLO_", "AUTOPHOTOEDIT_", "TRANSFORMERS_"))}
    env.update(extra_env)
    p = subprocess.run([PY, str(ROOT / "src" / "predict.py"), request, out], env=env, capture_output=True, text=True)
    attempts = [l for l in p.stderr.splitlines() if l.startswith("[offline]")]
    return p.returncode, attempts, p.stdout.strip().splitlines()[-1:] or p.stderr.strip().splitlines()[-3:]


def main(request):
    tmp = Path(tempfile.mkdtemp())
    rc_a, att_a, msg_a = run(request, str(tmp / "audit.tsv"), {"AUTOPHOTOEDIT_NETWORK_AUDIT": "1"})
    print(f"audit run (normal library behaviour): exit {rc_a}, {len(att_a)} network attempts")
    for a in sorted(set(att_a)):
        print("   ", a)
    rc_o, att_o, msg_o = run(request, str(tmp / "offline.tsv"), {})
    print(f"offline run (default): exit {rc_o}, {len(att_o)} network attempts | {msg_o}")
    for a in att_o:
        print("   ", a)
    same = rc_a == 0 and rc_o == 0 and (tmp / "audit.tsv").read_text() == (tmp / "offline.tsv").read_text()
    print(f"identical settings from both runs: {same}")
    ok = rc_o == 0 and not att_o and same
    print("OFFLINE TEST PASSED" if ok else "OFFLINE TEST FAILED")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1]) else 1)
