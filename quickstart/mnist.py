"""The model and the test images, both downloaded from Hugging Face."""

import io

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file
from torch import nn


class MLP(nn.Module):
    """dacorvo/mnist-mlp: a 28x28 image, flattened, to ten digit scores."""

    def __init__(self) -> None:
        super().__init__()
        self.input_layer = nn.Linear(784, 256)
        self.mid_layer = nn.Linear(256, 256)
        self.output_layer = nn.Linear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.input_layer(x))
        x = torch.relu(self.mid_layer(x))
        return self.output_layer(x)


def load_model() -> MLP:
    weights = hf_hub_download(
        "dacorvo/mnist-mlp",
        "model.safetensors",
        revision="7ed1e96d182395fd7380ea8495b9eb2364ce9dc1",
    )
    net = MLP()
    net.load_state_dict(load_file(weights))
    return net.eval()


def test_images() -> tuple[torch.Tensor, list[int]]:
    """The 10,000 MNIST test images, normalized and flattened as the model expects."""
    path = hf_hub_download(
        "ylecun/mnist",
        "mnist/test-00000-of-00001.parquet",
        repo_type="dataset",
        revision="77f3279092a1c1579b2250db8eafed0ad422088c",
    )
    table = pq.read_table(path)
    pngs = table["image"].to_pylist()
    pixels = np.stack([np.array(Image.open(io.BytesIO(png["bytes"]))) for png in pngs])
    pixels = pixels.astype(np.float32) / 255
    images = torch.from_numpy((pixels - 0.1307) / 0.3081).flatten(1)
    return images, table["label"].to_pylist()
