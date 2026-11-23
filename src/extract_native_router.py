"""Stream only the pinned physical layer-12 router tensors. No credentials."""

import io
import json
import struct
import sys
import tarfile

path='/work1/aviralkumar/geney/inkling-interp-data/hf/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/ed40e7da625f594fc9c4990f5f66be24873c22d7/model-00001-of-00001.safetensors'
with open(path,'rb') as stream:
    length=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(length));base=8+length
    keys=['model.llm.layers.11.mlp.gate.'+suffix for suffix in ['weight','bias']]
    with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as archive:
        meta=json.dumps({'model_revision':'ed40e7da625f594fc9c4990f5f66be24873c22d7',
                         'source_file':path,'tensors':{key:header[key] for key in keys}},indent=2).encode()
        item=tarfile.TarInfo('metadata.json');item.size=len(meta);archive.addfile(item,io.BytesIO(meta))
        for key in keys:
            start,end=header[key]['data_offsets'];stream.seek(base+start)
            item=tarfile.TarInfo(key.rsplit('.',1)[-1]+'.raw');item.size=end-start;archive.addfile(item,stream)
