"""One-command validation: launches Flask, runs pytest, then 50-worker HTTP load test."""
import os, subprocess, sys, time, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def main():
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    server = subprocess.Popen([sys.executable, 'app.py'], cwd=ROOT, env=env,
                              stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    try:
        ready = False
        for _ in range(60):
            if server.poll() is not None:
                raise RuntimeError('Flask server exited before becoming ready. Check dependencies and app.py.')
            try:
                with urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=2) as response:
                    if response.status == 200: ready = True; break
            except Exception:
                time.sleep(1)
        if not ready: raise RuntimeError('API did not become ready within 60 seconds.')
        print('\n[1/2] Running integration tests...')
        subprocess.run([sys.executable, '-m', 'pytest', '-q'], cwd=ROOT, env=env, check=True)
        print('\n[2/2] Running 50-concurrent-request load test (200 total requests)...')
        subprocess.run([sys.executable, 'scripts/load_test.py', '--concurrency', '50', '--requests', '200'], cwd=ROOT, env=env, check=True)
        print('\nVALIDATION COMPLETE. Attach reports/load_test_results.md and reports/load_test_results.json to the milestone.')
    finally:
        server.terminate()
        try: server.wait(timeout=10)
        except subprocess.TimeoutExpired: server.kill(); server.wait()

if __name__ == '__main__':
    main()
