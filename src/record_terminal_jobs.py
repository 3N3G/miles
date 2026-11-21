"""Read scheduler terminal states for this registry's jobs without mutating jobs."""
import datetime,json,subprocess
from pathlib import Path

path=Path('jobs.jsonl');rows=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
ids=sorted({str(r['job_id']) for r in rows if r.get('job_id') is not None and str(r['job_id']).isdigit()})
known={str(r['job_id']) for r in rows if r.get('event')=='terminal_observation'}
cmd='sacct --allocations --parsable2 --noheader --jobs='+','.join(ids)+' --format=JobIDRaw,State,Elapsed,ExitCode'
r=subprocess.run(['ssh','amd',cmd],text=True,capture_output=True,check=True)
terminal={'COMPLETED','CANCELLED','FAILED','TIMEOUT','OUT_OF_MEMORY','PREEMPTED','NODE_FAIL','BOOT_FAIL','DEADLINE'}
current=[]
for line in r.stdout.splitlines():
    parts=line.split('|')
    if len(parts)<4:continue
    job,state,elapsed,code=parts[:4];status=state.split()[0];current.append({'id':job,'cluster':'hpcfund','status':status,'elapsed':elapsed,'exit_code':code})
    if job not in known and status in terminal:
        with path.open('a') as f:f.write(json.dumps({'event':'terminal_observation','at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cluster':'hpcfund','job_id':job,'state':state,'elapsed':elapsed,'exit_code':code})+'\n')
Path('evidence/scheduler_observation.json').write_text(json.dumps({'observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'jobs':current},indent=2)+'\n')
print(json.dumps(current))
