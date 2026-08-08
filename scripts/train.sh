set -eu
torchrun --standalone --nproc-per-node=8 -m hacon.entrypoints.train --config configs/main.yaml --output runs/main

