# Belfort ML quickstart: MNIST on encrypted images

This example runs a pretrained MNIST classifier on encrypted images: the end
client encrypts each image, a GPU runs the model on the ciphertext, and the end
client decrypts the ten class scores. It holds the files of the
[Quickstart](https://eml.belfortlabs.com/docs/getting-started/quickstart) as a
uv project.

## What you need

- macOS 14 or newer on Apple Silicon, or Linux (x86-64 or ARM) on a recent
  distribution
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- A C++20 compiler
- A Belfort API key, from the console's
  [API keys page](https://eml.belfortlabs.com/api-keys). A new account has
  enough credits for this example.

## 1. Get the example

```bash
git clone https://github.com/belfortlabs/eml-examples.git
cd eml-examples/quickstart
uv sync
```

## 2. Compile

```bash
export BELFORT_API_KEY=blf_live_...          # your API key
uv run provider.py compile
```

`provider.py` compiles the pretrained model from `mnist.py`, calibrated on the
last 64 MNIST test images, and uploads it. Running it again reuses the compiled
model.

## 3. Serve

In the same terminal:

```bash
uv run provider.py serve
```

It prints `BELFORT_CLIENT_TOKEN=...`, starts a GPU deploy that serves that
token's keys, and waits for Enter. The deploy stops when you press Enter, or
after 30 minutes; you are charged for the time it ran, rounded up to 10
minutes, at most 3 credits.

## 4. Run the end client

In a second terminal, in the same directory:

```bash
export BELFORT_CLIENT_TOKEN=...              # the token from the first terminal
uv run client.py                              # or: uv run client.py 20
```

`client.py` encrypts the first test images, has the deploy classify them, and
prints one line per image, such as `digit 7: predicted 7 in 287 ms`, then the
number correct. The first run takes about 1.5 minutes to build the encryption
code, generate keys and upload the evaluation keys. It keeps the secret key's
seed in a `.secret-seed-*` file, so running it again on the same token reuses
the key pair and takes about 20 seconds.

Press Enter in the first terminal to stop the deploy.

## Files

| File | What |
|---|---|
| `mnist.py` | Downloads the classifier (`dacorvo/mnist-mlp`) and the MNIST test images from Hugging Face |
| `provider.py` | The model provider: `compile` and `serve` |
| `client.py` | The end client: encrypts test images and checks the predictions |

## Using another gateway

`provider.py` talks to Belfort's production gateway. To use another one, such
as a local development stack, set `BELFORT_BASE_URL` for it, for example
`export BELFORT_BASE_URL=http://localhost:8000/api/v1`. The end client needs
nothing: the token names the gateway it was minted on.

## Next

The [documentation](https://eml.belfortlabs.com/docs) covers your own models,
keys, deploys and credits, and troubleshooting.
