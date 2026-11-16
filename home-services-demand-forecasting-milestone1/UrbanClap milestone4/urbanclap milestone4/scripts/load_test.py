"""Concurrent API load test. Start app first, then run python scripts/load_test.py."""
import argparse, concurrent.futures, json, statistics, time
import requests

def one_request(url,payload,timeout):
    start=time.perf_counter()
    try:
        r=requests.post(url,json=payload,timeout=timeout)
        return {'ok':r.status_code==200,'status_code':r.status_code,'latency_ms':(time.perf_counter()-start)*1000}
    except Exception as e:
        return {'ok':False,'status_code':0,'latency_ms':(time.perf_counter()-start)*1000,'error':str(e)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:5000/forecast');p.add_argument('--requests',type=int,default=200);p.add_argument('--concurrency',type=int,default=50);p.add_argument('--timeout',type=float,default=30);a=p.parse_args()
    payload={'city':'Bangalore','service_type':'AC Repair'}; start=time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        results=list(pool.map(lambda _:one_request(a.url,payload,a.timeout),range(a.requests)))
    elapsed=time.perf_counter()-start; lat=[x['latency_ms'] for x in results]; good=sum(x['ok'] for x in results)
    report={'target_concurrency':a.concurrency,'total_requests':a.requests,'successful_requests':good,'failed_requests':a.requests-good,'error_rate_percent':round((a.requests-good)/a.requests*100,2),'total_elapsed_seconds':round(elapsed,3),'throughput_requests_per_second':round(a.requests/elapsed,2),'latency_ms':{'mean':round(statistics.mean(lat),2),'p50':round(statistics.median(lat),2),'p95':round(sorted(lat)[int(.95*(len(lat)-1))],2),'max':round(max(lat),2)}}
    print(json.dumps(report,indent=2))
    with open('reports/load_test_results.json','w') as f: json.dump(report,f,indent=2)
    if good != a.requests: raise SystemExit(1)
if __name__=='__main__':main()
