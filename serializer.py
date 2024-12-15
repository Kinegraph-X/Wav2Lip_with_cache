import json
import numpy as np


def serialize_chunk(shape,  chunk_start, chunk_data):
    # Example metadata
    metadata = {
        "shape": shape, # [32, 270, 480, 3]  # Original shape of the array
        "chunk_start": chunk_start,
    }

    chunk_data = chunk_data.tobytes()

    # Serialize metadata
    metadata_json = json.dumps(metadata).encode('utf-8')
    metadata_length = len(metadata_json)

    # Combine metadata length, metadata, and binary data
    return metadata_length.to_bytes(4, 'big') + metadata_json + chunk_data