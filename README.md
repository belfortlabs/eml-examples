# Belfort ML examples

Examples for [Belfort](https://eml.belfortlabs.com), which runs PyTorch models
on encrypted data. The [documentation](https://eml.belfortlabs.com/docs) explains
the steps they take.

| Directory | What it is |
|---|---|
| `quickstart` | A pretrained MNIST classifier on encrypted images, as a uv project: compile, serve, and run an end client. Start here; its README has the steps. |
| `mlp` | A three-layer MLP classifying MNIST digits |
| `lola` | The LoLA CNN on MNIST: one convolution block, then two linear layers |
| `helrm` | Test data only (dense and sparse features, labels), no model yet |

`mlp` and `lola` each hold `<name>.py` (the `nn.Module`), the trained model
(`<name>.pth`), and `test_data.npz` with 2000 test inputs under `X` and their
labels under `y`.

Each `vX.Y.Z` tag holds the examples as first published with that Belfort release, pinned to it.
