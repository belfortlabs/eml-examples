"""The end client: encrypt MNIST test images, have the deploy classify them,
and decrypt the scores.

    uv run client.py [COUNT]

The secret key derives from a seed this script keeps in a local file per
token, so running it again on the same token reuses the key pair.
"""

import hashlib
import os
import sys
from pathlib import Path

from belfort_ml_client import Client
from mnist import test_images

count = int(sys.argv[1]) if len(sys.argv) > 1 else 10
token = os.environ["BELFORT_CLIENT_TOKEN"]
seed_file = Path(f".secret-seed-{hashlib.sha256(token.encode()).hexdigest()[:12]}")
seed = bytes.fromhex(seed_file.read_text()) if seed_file.exists() else None
images, labels = test_images()

with Client(token, secret_seed=seed) as client:
    client.prepare()
    if seed is None:
        seed_file.touch(mode=0o600)
        seed_file.write_text(client.secret_seed.hex())
    correct = 0
    for image, label in zip(images[:count], labels[:count]):
        answer = client.infer([image.tolist()])
        scores = answer["values"]
        predicted = scores.index(max(scores))
        correct += predicted == label
        print(f"digit {label}: predicted {predicted} in {answer['latency_ms']} ms")
    print(f"{correct} of {count} correct")
