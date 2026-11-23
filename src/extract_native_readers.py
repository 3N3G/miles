"""Stream a few pinned block-reader vectors and the layer12 convolution."""

import io,json,struct,sys,tarfile
path='/work1/aviralkumar/geney/inkling-interp-data/hf/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/ed40e7da625f594fc9c4990f5f66be24873c22d7/model-00001-of-00001.safetensors'
keys=['model.llm.attn_res.attn_queries.11','model.llm.attn_res.attn_queries.17',
      'model.llm.attn_res.mlp_queries.12','model.llm.attn_res.mlp_queries.18',
      'model.llm.attn_res.output_query','model.llm.layers.11.mlp_sconv.weight']
with open(path,'rb') as f:
    n=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(n));base=8+n
    with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as t:
        meta=json.dumps({'model_revision':'ed40e7da625f594fc9c4990f5f66be24873c22d7','tensors':{k:h[k] for k in keys}},indent=2).encode()
        item=tarfile.TarInfo('metadata.json');item.size=len(meta);t.addfile(item,io.BytesIO(meta))
        for k in keys:
            a,b=h[k]['data_offsets'];f.seek(base+a);item=tarfile.TarInfo(k+'.raw');item.size=b-a;t.addfile(item,f)
