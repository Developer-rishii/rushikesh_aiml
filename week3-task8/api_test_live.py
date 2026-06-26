import requests, json

BASE = 'http://127.0.0.1:8001'

ok_example = None
ok_ids = None
warn_example = None
warn_ids = None

# Search for one OK and one LOW_FIT_WARNING
for cid in range(1, 50):
    for jid in range(1, 20):
        r = requests.post(f'{BASE}/guardrail-check', json={'candidate_id': cid, 'job_id': jid})
        d = r.json()
        if d.get('fit_status') == 'OK' and ok_example is None:
            ok_example = d
            ok_ids = (cid, jid)
        if d.get('fit_status') == 'LOW_FIT_WARNING' and warn_example is None:
            warn_example = d
            warn_ids = (cid, jid)
        if ok_example and warn_example:
            break
    if ok_example and warn_example:
        break

print('=' * 65)
print(f'  VERIFIED OK EXAMPLE: candidate_id={ok_ids[0]}, job_id={ok_ids[1]}')
print('=' * 65)
print(json.dumps(ok_example, indent=2))

print()
print('=' * 65)
print(f'  VERIFIED LOW_FIT_WARNING: candidate_id={warn_ids[0]}, job_id={warn_ids[1]}')
print('=' * 65)
print(json.dumps(warn_example, indent=2))

# Failure cases
print()
print('=' * 65)
print('  FAILURE CASE 1: Unknown candidate_id')
print('=' * 65)
r = requests.post(f'{BASE}/guardrail-check', json={'candidate_id': 9999, 'job_id': 1})
print(f'Status code: {r.status_code}')
print(json.dumps(r.json(), indent=2))

print()
print('=' * 65)
print('  FAILURE CASE 2: Unknown job_id')
print('=' * 65)
r = requests.post(f'{BASE}/guardrail-check', json={'candidate_id': 1, 'job_id': 9999})
print(f'Status code: {r.status_code}')
print(json.dumps(r.json(), indent=2))

print()
print('=' * 65)
print('  FAILURE CASE 3: Malformed request body')
print('=' * 65)
r = requests.post(f'{BASE}/guardrail-check', json={'candidate_id': 'abc'})
print(f'Status code: {r.status_code}')
print(json.dumps(r.json(), indent=2))
