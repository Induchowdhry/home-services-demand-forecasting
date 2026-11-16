# Observed HTTP load-test results

- Started (UTC): 2026-11-16T13:52:04.529443+00:00
- Endpoint: `http://127.0.0.1:5000/forecast`
- Request: `{"city": "Bangalore", "service_type": "AC Repair"}`
- Concurrent workers: **50**
- Total requests: **200**
- Successful / failed: **200 / 0**
- Error rate: **0.0%**
- Total elapsed: **6.646 s**
- Throughput (all requests): **30.09 requests/s**
- Successful throughput: **30.09 requests/s**

| Latency statistic | Observed value (ms) |
|---|---:|
| MEAN | 1553.05 |
| P50 | 1648.24 |
| P95 | 1947.23 |
| P99 | 2062.23 |
| MIN | 579.48 |
| MAX | 2081.12 |

These values are measured by this run on the recorded environment; results vary by hardware and runtime.
