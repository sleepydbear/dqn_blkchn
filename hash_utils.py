import hashlib as hl
import json

# __all__ = ['hash_string_256', 'hash_block']

def hash_string_256(string):

    return hl.sha256(string).hexdigest()


def hash_block(block):

    hashable_block = block.__dict__.copy()
    hashable_block['interactions'] = [tx.to_ordered_dict() for tx in hashable_block['interactions']]
    return hash_string_256(json.dumps(hashable_block, sort_keys=True).encode())