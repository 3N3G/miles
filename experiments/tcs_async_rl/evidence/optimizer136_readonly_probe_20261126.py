"""Read a uniform coordinate sample of saved Adam second moments. Standard library only."""
import bisect
import collections
import hashlib
import io
import json
import math
import pickle
import random
import struct
import zipfile
from pathlib import Path

class Bag:
    def __init__(self, *args):
        self.args = args
    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        elif type(self).__name__ == 'TensorProperties':
            self.__dict__.update(zip(['dtype', 'layout', 'requires_grad', 'memory_format', 'pin_memory'], state))
        else:
            self.state = state

classes = {}
class MetadataReader(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch' and name == 'Size':
            return tuple
        if module == 'torch' and name in ['float32', 'bfloat16', 'float16', 'int64', 'int32', 'uint8']:
            return name
        if module == 'torch.serialization' and name == '_get_layout':
            return str
        if module == 'torch.distributed.checkpoint.metadata' and name == '_MEM_FORMAT_ENCODING':
            return int
        if module in ['torch.distributed.checkpoint.metadata', 'torch.distributed.checkpoint.filesystem'] and name in ['Metadata', 'TensorStorageMetadata', 'TensorProperties', 'ChunkStorageMetadata', 'BytesStorageMetadata', 'MetadataIndex', '_StorageInfo', 'StorageMeta']:
            if name not in classes:
                classes[name] = type(name, (Bag,), {})
            return classes[name]
        raise pickle.UnpicklingError((module, name))

class Segment:
    def __init__(self, file, base, length):
        self.file, self.base, self.length, self.position = file, base, length, 0
    def tell(self):
        return self.position
    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.length + offset
        if position < 0:
            raise ValueError('negative seek')
        self.position = position
        return position
    def seekable(self):
        return True
    def read(self, count=-1):
        count = max(0, self.length - self.position) if count < 0 else min(count, max(0, self.length - self.position))
        self.file.seek(self.base + self.position)
        data = self.file.read(count)
        self.position += len(data)
        return data

class TensorReader(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch._utils' and name == '_rebuild_tensor_v2':
            return lambda storage, offset, size, stride, *args: {'storage': storage, 'offset': offset, 'size': size, 'stride': stride}
        if module == 'torch' and name == 'FloatStorage':
            return 'FloatStorage'
        if module == 'collections' and name == 'OrderedDict':
            return collections.OrderedDict
        raise pickle.UnpicklingError((module, name))
    def persistent_load(self, value):
        if not (value[0] == 'storage' and value[1] == 'FloatStorage'):
            raise pickle.UnpicklingError('unexpected storage')
        return value

def raw_offset(segment, info):
    assert info.compress_type == zipfile.ZIP_STORED
    segment.seek(info.header_offset)
    header = segment.read(30)
    assert header[:4] == b'PK\x03\x04'
    name_len, extra_len = struct.unpack_from('<HH', header, 26)
    return info.header_offset + 30 + name_len + extra_len

def validate_reader():
    values = struct.pack('<4f', 0.0, 1e-20, 0.25, 2.0)
    # A standalone segment uses zip offsets relative to its beginning.
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, 'w', compression=zipfile.ZIP_STORED) as z:
        z.writestr('archive/data/0', values)
    stream = io.BytesIO(b'prefix' + payload.getvalue() + b'suffix')
    segment = Segment(stream, 6, len(payload.getvalue()))
    with zipfile.ZipFile(segment) as z:
        info = z.getinfo('archive/data/0')
        offset = raw_offset(segment, info)
        segment.seek(offset)
        assert segment.read(info.file_size) == z.read(info) == values
    return True

def probe(directory, count=4096, seed=20261126):
    assert validate_reader()
    metadata_bytes = (directory / '.metadata').read_bytes()
    metadata = MetadataReader(io.BytesIO(metadata_bytes)).load()
    chunks = []
    for key, location in metadata.storage_data.items():
        if not key.fqn.endswith('.exp_avg_sq'):
            continue
        tensor = metadata.state_dict_metadata[key.fqn]
        assert tensor.properties.dtype == 'float32'
        matches = [c for c in tensor.chunks if tuple(c.offsets) == tuple(key.offset)]
        assert len(matches) == 1
        numel = math.prod(matches[0].sizes)
        chunks.append({'key': key.fqn, 'file': location.relative_path, 'offset': location.offset, 'length': location.length, 'numel': numel, 'logical_offsets': tuple(key.offset)})
    chunks.sort(key=lambda c: (c['key'], c['logical_offsets'], c['file'], c['offset']))
    cumulative = []
    for chunk in chunks:
        cumulative.append((cumulative[-1] if cumulative else 0) + chunk['numel'])
    rng = random.Random(seed)
    requests = collections.defaultdict(list)
    for sample_index in range(count):
        coordinate = rng.randrange(cumulative[-1])
        chunk_index = bisect.bisect_right(cumulative, coordinate)
        requests[chunk_index].append((sample_index, coordinate - (cumulative[chunk_index-1] if chunk_index else 0)))
    samples = [None] * count
    checked_records = 0
    small_record_verified = False
    files = {}
    try:
        for chunk_index, positions in sorted(requests.items()):
            chunk = chunks[chunk_index]
            if chunk['file'] not in files:
                files[chunk['file']] = (directory / chunk['file']).open('rb')
            file = files[chunk['file']]
            segment = Segment(file, chunk['offset'], chunk['length'])
            with zipfile.ZipFile(segment) as archive:
                tensor = TensorReader(io.BytesIO(archive.read('archive/data.pkl'))).load()
                assert tuple(tensor['size']) == (chunk['numel'],) and tuple(tensor['stride']) == (1,)
                assert tensor['storage'][2] == '0'
                info = archive.getinfo('archive/data/0')
                assert tensor['offset'] >= 0
                assert (tensor['offset'] + chunk['numel']) * 4 <= info.file_size
                start = raw_offset(segment, info) + 4 * tensor['offset']
                if chunk['length'] < 20000:
                    segment.seek(start)
                    direct = segment.read(4 * chunk['numel'])
                    assert direct == archive.read(info)[4*tensor['offset']:4*(tensor['offset']+chunk['numel'])]
                    small_record_verified = True
                for sample_index, position in positions:
                    segment.seek(start + position * 4)
                    value, = struct.unpack('<f', segment.read(4))
                    assert math.isfinite(value) and value >= 0
                    samples[sample_index] = value
                checked_records += 1
    finally:
        for file in files.values():
            file.close()
    smallest = min(chunks, key=lambda c: c['length'])
    assert smallest['length'] < 20000
    with (directory / smallest['file']).open('rb') as file:
        segment = Segment(file, smallest['offset'], smallest['length'])
        with zipfile.ZipFile(segment) as archive:
            info = archive.getinfo('archive/data/0')
            offset = raw_offset(segment, info)
            segment.seek(offset)
            assert segment.read(info.file_size) == archive.read(info)
            small_record_verified = True
    assert all(value is not None for value in samples)
    return {'metadata_sha256': hashlib.sha256(metadata_bytes).hexdigest(), 'seed': seed, 'coordinate_samples': count, 'population_stored_coordinates': cumulative[-1], 'population_chunks': len(chunks), 'records_read': checked_records, 'segment_fixture_passed': True, 'small_record_full_read_crosscheck': small_record_verified, 'second_moments': samples}

if __name__ == '__main__':
    print(json.dumps(probe(Path('/work1/aviralkumar/geney/tcs-async-rl/runs/easy-medium-236-v2/checkpoints/iter_0000136'))))
