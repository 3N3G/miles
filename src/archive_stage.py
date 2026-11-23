"""Archive a completed stage on its host and record a private immutable link."""
import argparse,datetime,json,shlex,subprocess
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('stages',nargs='+');p.add_argument('--already-uploaded',action='store_true');a=p.parse_args()
    root='/work1/aviralkumar/geney/inkling-interp-data'
    for stage in a.stages:
        if not stage.replace('-','').replace('_','').isalnum():raise ValueError(stage)
        if not a.already_uploaded:
            cmd='HF_XET_CACHE='+shlex.quote(root+'/xet-cache')+' push-run inkling-interp '+shlex.quote(root+'/runs/'+stage)+' > '+shlex.quote(root+'/archive-staging/upload-'+stage+'.log')+' 2>&1'
            subprocess.run(['ssh','amd',cmd],check=True)
        code='''import json
from huggingface_hub import HfApi
api=HfApi();repo="3N3G/internalization-runs"
info=api.repo_info(repo,repo_type="dataset",timeout=20)
assert info.private
stage=STAGE
files=list(api.list_repo_tree(repo,path_in_repo="inkling-interp/"+stage,repo_type="dataset",revision=info.sha,recursive=True))
assert len(files)>=3
print(json.dumps({"stage":stage,"repo":repo,"revision":info.sha,"private":info.private,"files":len(files),"url":"https://huggingface.co/datasets/"+repo+"/tree/"+info.sha+"/inkling-interp/"+stage}))
'''.replace('STAGE',repr(stage))
        r=subprocess.run(['ssh','amd','/work1/aviralkumar/geney/hf_cache/cli/venv/bin/python -'],input=code,text=True,capture_output=True,check=True)
        record=json.loads(r.stdout);record['verified_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        path=Path('evidence/artifact_archives.json');old=json.loads(path.read_text())
        if not any(x['stage']==stage and x['revision']==record['revision'] for x in old):old.append(record);path.write_text(json.dumps(old,indent=2)+'\n')
        registry=Path('jobs.jsonl')
        rows=[json.loads(line) for line in registry.read_text().splitlines()]
        submissions=[r for r in rows if r.get('run_name')==stage and r.get('submission_returncode')==0 and r.get('job_id')]
        if submissions and not any(r.get('artifact_url')==record['url'] for r in rows):
            receipt={'at':record['verified_at'],'kind':'artifact_archive','cluster':'hpcfund',
                     'job_id':submissions[-1]['job_id'],'run_name':stage,
                     'artifact_url':record['url'],'artifact_revision':record['revision']}
            with registry.open('a') as f:f.write(json.dumps(receipt)+'\n')
        print(json.dumps(record),flush=True)
if __name__=='__main__':main()
