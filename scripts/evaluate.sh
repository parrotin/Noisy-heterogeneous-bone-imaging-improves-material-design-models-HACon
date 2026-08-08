set -eu
hacon-evaluate --predictions runs/main/dice.pt --reference runs/single_site/dice.pt --output runs/evaluation/dice.json

