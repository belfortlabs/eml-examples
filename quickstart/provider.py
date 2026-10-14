"""The model provider: compile the model, then serve it to one end client.

    python provider.py compile   # compile the model on Belfort
    python provider.py serve     # mint a client token and start a GPU deploy

Both are safe to run again: a model already compiled from the same MLIR is
reused, not compiled again.
"""

import sys
from datetime import timedelta

import belfort_ml
from mnist import load_model, test_images

NAME = "quickstart_mnist"


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
    token = model.create_client(label="quickstart")
    print(f"\nBELFORT_CLIENT_TOKEN={token.token}\n", flush=True)
    with model.deploy(key_set=token, max_lifetime=timedelta(minutes=30), timeout=1800):
        print("deploy ready")
        input("Press Enter to stop the deploy...")


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
