import io,json,struct,sys,tarfile
p='/work1/aviralkumar/geney/inkling-interp-data/hf/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/ed40e7da625f594fc9c4990f5f66be24873c22d7/model-00001-of-00001.safetensors'
with open(p,'rb') as f:
 n=struct.unpack('<Q',f.read(8))[0]; h=json.loads(f.read(n)); base=8+n
 keys=['model.llm.layers.11.mlp.experts.'+suffix for suffix in ['w2_weight','w2_weight_scale','w2_weight_amax','w2_weight_outer_scale_denominator']]
 metadata={'model_revision':'ed40e7da625f594fc9c4990f5f66be24873c22d7','source_file':p,'data_start':base,'tensors':{key:h[key] for key in keys}}
 with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as archive:
  raw=json.dumps(metadata,indent=2).encode(); item=tarfile.TarInfo('metadata.json'); item.size=len(raw);archive.addfile(item,io.BytesIO(raw))
  for key in keys:
   a,b=h[key]['data_offsets'];f.seek(base+a); item=tarfile.TarInfo(key.rsplit('.',1)[1]+'.raw');item.size=b-a; archive.addfile(item,f)
