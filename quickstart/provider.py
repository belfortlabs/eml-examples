"""The model provider: compile the model, then serve it to one end client.

    uv run provider.py compile   # compile the model on Belfort
    uv run provider.py serve     # mint a client token and start a GPU deploy

Both are safe to run again: a model already compiled from the same MLIR is
reused, not compiled again.
"""

import sys
from datetime import timedelta
from pathlib import Path

import belfort_ml
from mnist import load_model, test_images

NAME = "quickstart_mnist"
# client.py on this machine reads the token from here; it exists while serve
# runs.
TOKEN_FILE = Path(".client-token")


def compiled_model(provider: belfort_ml.Provider) -> belfort_ml.Model:
    images, _ = test_images()
    # The last 512 test images; client.py sends the first ones.
    calibration = list(images[-512:].split(1))
    artifacts = belfort_ml.compile(
        load_model(), calibration[0], NAME, "./build", calibration_inputs=calibration
    )
    return provider.models.get(
        name=NAME, create_from=artifacts, timeout=1800, use_composite_relu=False
    )


def serve(model: belfort_ml.Model) -> None:
    TOKEN_FILE.unlink(missing_ok=True)  # a killed earlier run may have left one
    token = model.create_client(label="quickstart")
    TOKEN_FILE.touch(mode=0o600)
    TOKEN_FILE.write_text(token.token)
    try:
        print(
            f"\nStart client.py now. On this machine it reads the token from"
            f" {TOKEN_FILE}.\nOn another machine, run this before client.py:\n"
            f"  export BELFORT_CLIENT_TOKEN={token.token}\n",
            flush=True,
        )
        # The GPU bills from here, so start it only once the client's keys
        # are in.
        model.wait_for_keys(token)
        print("Keys uploaded. Starting the deploy; Ctrl-C stops it.", flush=True)
        with model.deploy(
            key_set=token, max_lifetime=timedelta(minutes=30), timeout=1800
        ):
            input("Deploy ready. Press Enter to stop it...")
    finally:
        TOKEN_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command not in ("compile", "serve"):
        sys.exit(__doc__)
    with belfort_ml.Provider() as provider:  # reads BELFORT_API_KEY
        model = compiled_model(provider)
        if command == "compile":
            print(f"{NAME} is compiled ({model.id})")
        else:
            serve(model)
