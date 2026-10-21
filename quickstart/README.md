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
last 512 MNIST test images, and uploads it. Running it again reuses the compiled
model.

## 3. Serve

In the same terminal:

```bash
uv run provider.py serve
```

The provider mints a client token, writes it to `.client-token`, prints it as an
`export BELFORT_CLIENT_TOKEN=...` line, and waits for the end client to upload
its keys. Only then does it start a GPU deploy. The
deploy stops when you press Enter, or after 30 minutes. You are charged for
the time it ran, rounded up to 10 minutes. The file is removed when the script
exits.

## 4. Run the end client

Run the end client in a second terminal, in the same directory:

```bash
uv run client.py                              # or: uv run client.py 20
```

`client.py` reads the token from `.client-token`. To run it on another machine,
copy `client.py` and `mnist.py` there, run `uv sync`, and run the `export` line
the provider printed before `uv run client.py`.

`client.py` encrypts the first test images, has the deploy classify them, and
prints one line per image, such as `digit 7: predicted 7 in 287 ms`, then the
number correct. The first run takes about 1.5 minutes to build the encryption
code, generate keys and upload the evaluation keys. Then it waits for the deploy. It keeps the secret key's
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
