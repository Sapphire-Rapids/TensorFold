"""Generated-input CUDA cancel/retry and cold-prefill receipts; never reads user conversations."""
import argparse
import concurrent.futures
import hashlib
import http.client
import json
import pathlib
import re
import socket
import sys
import time
import urllib.parse
import urllib.request

p = argparse.ArgumentParser()
p.add_argument('--base', default='http://127.0.0.1:8910')
p.add_argument('--model', default='cuda-resume-check')
p.add_argument('--source', required=True)
p.add_argument('--out', required=True)
p.add_argument('--mode', choices=['baseline-cache', 'candidate-cache', 'baseline-off', 'candidate-off'], required=True)
p.add_argument('--cancel-seconds', type=float, default=1.0)
p.add_argument('--cancel-count', type=int, choices=[1, 2], default=2)
p.add_argument('--progress-reference')
p.add_argument('--words', type=int, default=5000)
p.add_argument('--payload')
p.add_argument('--reference')
p.add_argument('--skip-cold', action='store_true')
p.add_argument('--legacy-token-hash', action='store_true')
p.add_argument('--allow-complete-retry', action='store_true')
args = p.parse_args()
sys.path.insert(0, str(pathlib.Path(args.source) / 'tools'))
from bench_evidence import Evidence, token_equal
import bench_concurrent as bench
import bench_cache

out = pathlib.Path(args.out)
out.mkdir(parents=True, exist_ok=True)
base = args.base.rstrip('/')
where = urllib.parse.urlsplit(base)
hash_field = 'token_sha256'

def same_tokens(left, right):
    return token_equal(left, right) is True


def progress_advanced(current, earlier, cancel_count):
    assert current['resumed_cached_tokens'] > 0, 'retry recomputed the entire prompt'
    if earlier is not None:
        assert earlier['request_sha256'] == current['request_sha256'], 'progress reference uses a different request'
        assert current['resumed_cached_tokens'] > earlier['resumed_cached_tokens'], 'retry retained no additional progress'
        return True
    assert cancel_count == 1, 'repeated cancellation needs a baseline or single-cancellation reference'
    return False


def post(route, body):
    request = urllib.request.Request(base + route, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)

def running():
    with urllib.request.urlopen(base + '/metrics', timeout=5) as response:
        text = response.read().decode()
    values = {}
    for name in ['requests_running', 'requests_waiting', 'prompt_tokens_cached_total']:
        found = re.search(r'^tensorfold:' + name + r' (\S+)$', text, re.M)
        if found:
            values[name] = float(found[1])
    assert 'requests_running' in values, 'running metric is unavailable'
    return values

def idle(limit=600):
    until = time.monotonic() + limit
    while True:
        values = running()
        if values['requests_running'] == 0 and values.get('requests_waiting', 0) == 0:
            return values
        if time.monotonic() >= until:
            raise TimeoutError('server did not become idle after the generated request')
        time.sleep(0.1)

def streamed(body, label, timeout=1800, cancel=False, may_finish=False):
    if cancel:
        idle()
    began = time.monotonic()
    conn = http.client.HTTPConnection(where.hostname, where.port, timeout=timeout)
    response = None
    evidence = Evidence()
    usage, runtime = {}, {}
    result = {'label': label, 'complete': False, 'client_cancelled': False}
    pieces = 0
    try:
        conn.request('POST', '/v1/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
        if cancel:
            until = time.monotonic() + 15
            while running()['requests_running'] == 0:
                if time.monotonic() >= until:
                    raise RuntimeError('generated request was never observed running')
                time.sleep(0.01)
            result['observed_running_s'] = time.monotonic() - began
            conn.sock.settimeout(timeout)
        response = conn.getresponse()
        result['http_status'] = response.status
        if response.status != 200:
            raise RuntimeError(f'HTTP {response.status}: {response.read().decode()}')
        for raw in response:
            line = raw.decode().strip()
            if line == 'data: [DONE]':
                result['complete'] = True
                break
            if not line.startswith('data:'):
                continue
            value = json.loads(line[5:])
            if value.get('error'):
                raise RuntimeError(value['error'])
            usage = value.get('usage') or usage
            runtime = value.get('tensorfold') or runtime
            for choice in value.get('choices', []):
                evidence.add(choice)
                delta = choice.get('delta') or {}
                text = choice.get('text') or delta.get('content') or delta.get('reasoning_content') or ''
                if text:
                    pieces += 1
                    if 'ttft_s' not in result:
                        result['ttft_s'] = time.monotonic() - began
    except (TimeoutError, socket.timeout):
        if not cancel:
            raise
        result['client_cancelled'] = True
    finally:
        if response:
            response.close()
        if conn.sock:
            try:
                conn.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        conn.close()
    result['elapsed_s'] = time.monotonic() - began
    result['pieces'] = pieces
    result.update(evidence.finish(usage, runtime, result['complete']))
    if result['complete'] and args.legacy_token_hash:
        returned = runtime.get('token_ids')
        assert isinstance(returned, list) and returned and all(type(x) is int and 0 <= x <= 0xffffffff for x in returned)
        assert len(returned) == result['tokens'], 'returned token count differs from usage'
        digest = hashlib.sha256(','.join(map(str, returned)).encode()).hexdigest()
        assert result['token_sha'] == digest[:12], 'returned token ids disagree with the server fingerprint'
        result['token_ids'] = returned
        result['token_sha256'] = digest
    if cancel:
        result['after'] = idle()
        if result['complete']:
            assert may_finish and result[hash_field], 'prompt finished before the cancellation window'
        else:
            assert result['client_cancelled'], 'request did not finish or cancel'
            assert pieces == 0, 'cancellation happened during decode rather than prefill'
    else:
        assert result['complete'] and result[hash_field] and result['tokens'] > 0, result
    return result

if args.payload:
    body = json.loads(pathlib.Path(args.payload).read_text())
    body['model'] = args.model
else:
    text = ('This is a generated notebook, not real measurements. Each record has a fixed identifier and value.\n'
            + '\n'.join(f'Record {i:06d}: sample {(i * 337 + 101) % 899 + 100}, batch {i % 97}.' for i in range(args.words))
            + '\nReturn only the sample values for records '
            + ', '.join(f'{i:06d}' for i in [0, args.words // 3, 2 * args.words // 3, args.words - 1])
            + ', in that order, separated by commas.')
    body = {'model': args.model, 'messages': [{'role': 'user', 'content': text}],
            'temperature': 0, 'seed': 1234, 'max_tokens': 32, 'ignore_eos': True,
            'stream': True, 'stream_options': {'include_usage': True},
            'chat_template_kwargs': {'enable_thinking': False}}
if args.legacy_token_hash:
    body['return_token_ids'] = True
encoded = json.dumps(body).encode()
(out / 'synthetic-request.json').write_bytes(encoded)
tokenized = post('/tokenize', body)
ids = tokenized['tokens']
(out / 'prompt-ids.json').write_text(json.dumps(ids) + '\n')
report = {'synthetic_only': True, 'mode': args.mode, 'prompt_tokens': len(ids),
          'request_sha256': hashlib.sha256(encoded).hexdigest(),
          'fingerprint': 'sha256-token-ids', 'runs': []}

def save(row):
    report['runs'].append(row)
    (out / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(row), flush=True)

idle()
if args.mode.endswith('-off'):
    for n in ([] if args.skip_cold else [2048, 8192, 32768, 65536]):
        assert len(ids) >= n, 'fixture is too short for the cold-prefill cells'
        item = {'name': f'cold-{n}', 'kind': 'completion', 'prompt': ids[:n]}
        row = bench_cache.receipt(bench.stream(base, args.model, item, 32, 0, 1234), f'cold-{n}')
        assert row.get('complete') and row.get('cached_tokens') == 0 and row.get(hash_field), row
        save(row)
    fresh = streamed(body, 'fresh-long')
    assert fresh['cached_tokens'] == 0, fresh
    save(fresh)
else:
    resumed = None
    for i in range(args.cancel_count):
        row = streamed(body, f'retry-window-{i + 1}', args.cancel_seconds, True, args.allow_complete_retry and i > 0)
        save(row)
        if row['complete']:
            resumed = row
            break
    if resumed is None:
        resumed = streamed(body, 'resumed')
        save(resumed)
    report['resumed_cached_tokens'] = resumed['cached_tokens']
    if args.mode == 'candidate-cache':
        earlier = json.loads(pathlib.Path(args.progress_reference).read_text()) if args.progress_reference else None
        report['progress_advanced'] = progress_advanced(report, earlier, args.cancel_count)
    plain_body = dict(body, draft=False)
    plain = streamed(plain_body, 'plain-retry')
    save(plain)
    assert same_tokens(resumed, plain), 'drafted and plain retry tokens differ'
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(streamed, body, f'concurrent-{i}') for i in range(2)]
        for future in futures:
            row = future.result()
            save(row)
            assert same_tokens(resumed, row), 'concurrent retry differs from the solo reply'
if args.reference:
    prior = json.loads(pathlib.Path(args.reference).read_text())
    fresh = next(row for row in prior['runs'] if row.get('label') == 'fresh-long')
    for row in report['runs']:
        if row.get('label') and row.get('complete'):
            assert same_tokens(fresh, row), 'resumed or candidate tokens differ from the cache-off reference'
        if row.get('phase', '').startswith('cold-'):
            other = next(x for x in prior['runs'] if x.get('phase') == row['phase'])
            assert same_tokens(other, row), 'candidate cold tokens differ from baseline'
report['qualified'] = True
(out / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'qualified': True, 'mode': args.mode}), flush=True)
