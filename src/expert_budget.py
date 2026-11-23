"""Account every physical GPU in this approved program, using live Slurm receipts."""

import datetime
import json
import subprocess
from pathlib import Path


def main():
    path=Path('evidence/expert_program_budget.json')
    data=json.loads(path.read_text());jobs=data['jobs']
    ids=','.join(j['job_id'] for j in jobs)
    result=subprocess.run(['ssh','amd','sacct -n -X -j '+ids+' --format=JobIDRaw,State,ElapsedRaw -P'],
                          text=True,capture_output=True,check=True)
    observed={}
    for line in result.stdout.splitlines():
        fields=line.strip().split('|')
        if len(fields)>=3 and fields[0] in ids.split(','):observed[fields[0]]=fields[1:3]
    used=data['prior_pilot_seconds_counted_conservatively'];reserved=0
    for job in jobs:
        if job['job_id'] not in observed:raise RuntimeError('Missing scheduler record: '+job['job_id'])
        state,elapsed=observed[job['job_id']];elapsed=int(elapsed)
        job.update(status=state,observed_allocation_seconds=elapsed)
        if state.split()[0] in ['PENDING','RUNNING','COMPLETING','CONFIGURING','SUSPENDED','REQUEUED']:
            job['reserved_gpu_seconds']=job['maximum_allocation_seconds']*job['gpus']
            reserved+=job['reserved_gpu_seconds']
        else:
            job['actual_allocation_seconds']=elapsed
            job['actual_gpu_seconds']=elapsed*job['gpus']
            job['reserved_gpu_seconds']=0;used+=job['actual_gpu_seconds']
    reserved+=sum(x['maximum_gpu_seconds'] for x in data.get('pending_reservations',[]))
    data.update(used_gpu_seconds=used,reserved_gpu_seconds=reserved,
                unreserved_gpu_seconds=data['limit_gpu_seconds']-used-reserved,
                remaining_gpu_seconds=data['limit_gpu_seconds']-used,
                observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    assert data['unreserved_gpu_seconds']>=0,data
    path.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data,indent=2))


if __name__=='__main__':main()
