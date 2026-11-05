"""Project the pinned PyTorch recorder into reviewed collective metadata.

Never return process-group names, thread names, stack text, tensor values or
arbitrary recorder fields. This module writes no diagnostic file.
"""
import json
import math

OPS = {
    'nccl:all_reduce': 'all_reduce', 'nccl:broadcast': 'broadcast',
    'nccl:reduce': 'reduce', 'nccl:all_gather': 'all_gather',
    'nccl:_all_gather_base': 'all_gather', 'nccl:all_gather_into_tensor': 'all_gather',
    'nccl:reduce_scatter': 'reduce_scatter', 'nccl:_reduce_scatter_base': 'reduce_scatter',
    'nccl:all_to_all': 'all_to_all', 'nccl:all_to_all_single': 'all_to_all',
    'nccl:send': 'send', 'nccl:recv': 'recv', 'nccl:barrier': 'barrier',
}
STATES = {'scheduled', 'started', 'completed'}
WIDTHS = {'Float': 4, 'Double': 8, 'Half': 2, 'BFloat16': 2, 'Long': 8,
          'Int': 4, 'Short': 2, 'Char': 1, 'Byte': 1, 'Bool': 1,
          'ComplexFloat': 8, 'ComplexDouble': 16}


def integer(value):
    return type(value) is int and 0 <= value < 2**63


def project_trace(raw):
    if type(raw) not in (bytes, str) or len(raw) > 1048576:
        raise ValueError('invalid recorder payload')
    document = json.loads(raw)
    if type(document) is not dict or type(document.get('entries')) is not list:
        raise ValueError('invalid recorder document')
    result = []
    entries = document['entries']
    rejected = max(0, len(entries) - 32)
    for entry in entries[-32:]:
        if type(entry) is not dict:
            rejected += 1
            continue
        name, state = entry.get('profiling_name'), entry.get('state')
        if type(name) is not str or name not in OPS or type(state) is not str or state not in STATES:
            rejected += 1
            continue
        sequence = entry.get('p2p_seq_id') if entry.get('is_p2p') is True else entry.get('collective_seq_id')
        sizes, dtypes = entry.get('input_sizes'), entry.get('input_dtypes')
        if not integer(sequence) or type(sizes) is not list or len(sizes) > 64:
            rejected += 1
            continue
        counts = []
        for shape in sizes:
            if type(shape) is not list or len(shape) > 16 or any(not integer(n) for n in shape):
                break
            count = math.prod(shape)
            if not integer(count):
                break
            counts.append(count)
        if len(counts) != len(sizes) or not integer(sum(counts)):
            rejected += 1
            continue
        value = {'op': OPS[name], 'state': state, 'sequence': sequence, 'count': sum(counts)}
        if integer(entry.get('pg_id')):
            value['group_id'] = entry['pg_id']
        if type(dtypes) is list and len(dtypes) == len(counts) and all(type(t) is str and t in WIDTHS for t in dtypes):
            size = sum(n * WIDTHS[t] for n, t in zip(counts, dtypes))
            if integer(size):
                value['bytes'] = size
        for source, target in (('time_created_ns','created_ns'),
                               ('time_discovered_started_ns','started_ns'),
                               ('time_discovered_completed_ns','completed_ns'),
                               ('timeout_ms','timeout_ms')):
            number = entry.get(source)
            if integer(number) and (number > 0 or target == 'timeout_ms'):
                value[target] = number
        result.append(value)
    return result, rejected
