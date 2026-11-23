"""Run independent real inference workers on every GPU of an exclusive AMD node."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=8)
    p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    assert a.command and a.workers==8
    a.out.mkdir(parents=True,exist_ok=False)
    processes=[];logs=[];started=time.time()
    for rank in range(a.workers):
        log=(a.out/f'worker{rank}.log').open('w');logs.append(log)
        env=dict(os.environ,ROCR_VISIBLE_DEVICES=str(rank),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        env.pop('HIP_VISIBLE_DEVICES',None);env.pop('CUDA_VISIBLE_DEVICES',None)
        cmd=[*a.command,'--out',str(a.out),'--rank',str(rank),'--world-size',str(a.workers)]
        processes.append(subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT))
    codes=[proc.wait() for proc in processes]
    for log in logs:log.close()
    (a.out/'parallel_manifest.json').write_text(json.dumps({'workers':a.workers,'returncodes':codes,
           'program_wall_seconds':time.time()-started,'all_allocated_gpus_count_toward_budget':True},indent=2)+'\n')
    print(json.dumps({'returncodes':codes,'wall_seconds':time.time()-started}),flush=True)
    sys.exit(1 if any(codes) else 0)


if __name__=='__main__':main()
