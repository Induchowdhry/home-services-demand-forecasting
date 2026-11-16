"""HTTP load test: 50 concurrent workers / 200 requests by default.
Run the Flask API first, then execute this script from any working directory.
"""
import argparse, concurrent.futures, json, os, platform, statistics, time
from datetime import datetime, timezone
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]

def one_request(url, payload, timeout):
    start = time.perf_counter()
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        return {"ok": response.status_code == 200, "status_code": response.status_code,
                "latency_ms": (time.perf_counter() - start) * 1000}
    except Exception as exc:
        return {"ok": False, "status_code": 0,
                "latency_ms": (time.perf_counter() - start) * 1000, "error": str(exc)}

def percentile(values, p):
    return round(sorted(values)[min(len(values)-1, max(0, int((p/100)*(len(values)-1))))], 2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:5000/forecast')
    parser.add_argument('--requests', type=int, default=200)
    parser.add_argument('--concurrency', type=int, default=50)
    parser.add_argument('--timeout', type=float, default=60)
    parser.add_argument('--city', default='Bangalore')
    parser.add_argument('--service', default='AC Repair')
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1:
        parser.error('--requests and --concurrency must be positive')
    payload = {'city': args.city, 'service_type': args.service}
    started_at = datetime.now(timezone.utc).isoformat()
    wall_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        results = list(executor.map(lambda _: one_request(args.url, payload, args.timeout), range(args.requests)))
    elapsed = time.perf_counter() - wall_start
    latencies = [r['latency_ms'] for r in results]
    successes = sum(r['ok'] for r in results)
    failures = args.requests - successes
    report = {
        'test_type': 'HTTP POST /forecast load test', 'started_at_utc': started_at,
        'target_concurrency_workers': args.concurrency, 'total_requests': args.requests,
        'successful_requests': successes, 'failed_requests': failures,
        'error_rate_percent': round(failures / args.requests * 100, 2),
        'total_elapsed_seconds': round(elapsed, 3),
        'throughput_requests_per_second': round(args.requests / elapsed, 2),
        'successful_throughput_requests_per_second': round(successes / elapsed, 2),
        'latency_ms': {'mean': round(statistics.mean(latencies), 2), 'p50': percentile(latencies, 50),
                       'p95': percentile(latencies, 95), 'p99': percentile(latencies, 99),
                       'min': round(min(latencies), 2), 'max': round(max(latencies), 2)},
        'environment': {'python': platform.python_version(), 'platform': platform.platform()},
        'request': payload, 'endpoint': args.url
    }
    out = ROOT / 'reports' / 'load_test_results.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    md = ROOT / 'reports' / 'load_test_results.md'
    md.write_text('# Observed HTTP load-test results\n\n' +
        f"- Started (UTC): {started_at}\n- Endpoint: `{args.url}`\n- Request: `{json.dumps(payload)}`\n" +
        f"- Concurrent workers: **{args.concurrency}**\n- Total requests: **{args.requests}**\n" +
        f"- Successful / failed: **{successes} / {failures}**\n- Error rate: **{report['error_rate_percent']}%**\n" +
        f"- Total elapsed: **{report['total_elapsed_seconds']} s**\n" +
        f"- Throughput (all requests): **{report['throughput_requests_per_second']} requests/s**\n" +
        f"- Successful throughput: **{report['successful_throughput_requests_per_second']} requests/s**\n\n" +
        '| Latency statistic | Observed value (ms) |\n|---|---:|\n' +
        ''.join(f"| {k.upper()} | {v} |\n" for k,v in report['latency_ms'].items()) +
        '\nThese values are measured by this run on the recorded environment; results vary by hardware and runtime.\n')
    print(json.dumps(report, indent=2))
    print(f'\nSaved: {out}\nSaved: {md}')
    if failures:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
