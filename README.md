# Noisy heterogeneous bone imaging data improves material design models: contrastive self-supervised learning across distributed clinical archives

HACon learns scanner-invariant bone structure representations from heterogeneous clinical CT archives. It combines anatomically constrained cross-site positive pairs with pair-specific contrastive temperatures derived from Hounsfield-unit offset, noise power spectrum distance, and spatial-resolution ratio. The frozen 512-dimensional representation supports bone segmentation and prediction of elastic modulus, porosity, and trabecular thickness.

## Environment

The primary environment is Python 3.10, PyTorch 2.1.0, CUDA 12.1, and MONAI 1.3.0.

    conda env create -f environment.yml
    conda activate hacon
    pip install --no-deps .

A container build is also available.

    docker build -t hacon:1.0 .

## Data

All retained and verified data access points, versions, licenses, and access restrictions are listed in dataset_links.txt. OAI requires registration and acceptance of the NIH Data Use Agreement. Each institution remains responsible for complying with the source license and local data-governance policy.

The pretraining pool contains 7,040 retained volumes after excluding records with more than 30% missing scanner metadata. Volumes are resampled to 1 mm isotropic spacing, windowed to −200 through 1500 HU, normalized to [0, 1], and cropped into 96 × 96 × 96 anatomical patches.

    hacon-prepare --root data/oai --output data/manifests/oai.jsonl --dataset OAI --site Baltimore

Each manifest row records an image, optional label, site, anatomical region, scanner manufacturer, reconstruction kernel, slice thickness, spacing, split, and dataset identity.

## Pretraining

The primary configuration uses a volumetric ResNet-50, a 512-dimensional representation, and a 2048-to-128 projection head. Training uses a global batch of 256 across eight NVIDIA A100 80 GB GPUs for 200 epochs with AdamW, learning rate 3e-4, ten warm-up epochs, cosine decay, and mixed precision.

    torchrun --standalone --nproc-per-node=8 -m hacon.entrypoints.train --config configs/main.yaml --output runs/main

The reported pretraining budget is approximately 47.2 GPU-hours on eight A100 80 GB devices. Storage depends on source archive representation and preprocessing cache policy; retain source manifests and their SHA-256 audit digest with each run.

## Ablations

The component matrix is defined in configs/ablations.yaml. It covers full HACon, removal of CSPM, removal of SMT, pooled SimCLR, single-site pretraining, and an equal-compute random-initialization control. Every configuration retains the main training budget unless the experiment definition explicitly changes the data source.

| Configuration | Dice |
|---|---:|
| HACon | 0.901 ± 0.011 |
| Without CSPM | 0.838 |
| Without SMT | 0.874 |
| Pooled SimCLR | 0.789 ± 0.019 |
| Single-site SSL | 0.871 ± 0.014 |
| Equal-budget control | 0.832 ± 0.019 |

## Evaluation

Cross-scanner evaluation uses leave-one-hospital-out folds over Baltimore, Pittsburgh, Columbus, and Pawtucket. In-distribution and out-of-distribution subsets are separated by scanner vendor. Segmentation reporting includes Dice, AUC, HD95, sensitivity, specificity, precision, Jaccard score, and volume similarity.

    hacon-evaluate --predictions runs/main/dice.pt --reference runs/single_site/dice.pt --output runs/evaluation/dice.json

The primary expected values are OOD Dice 0.901 ± 0.011, OOD AUC 0.879 ± 0.013, HD95 4.21 ± 0.48 mm, and an ID-to-OOD gap of 2.4 percentage points. Paired comparisons use five matched seeds and two-sided paired t-tests with 95% confidence intervals.

Material-property heads contain hidden layers of 256 and 128 ReLU units over frozen encoder features. Expected values are R² 0.872 ± 0.018 for elastic modulus, 0.891 ± 0.015 for porosity, and 0.803 ± 0.023 for trabecular thickness. Report MAE, RMSE, relative error, prediction-interval coverage, calibration error, and uncertainty-error correlation.

## Software layout

code/hacon/data contains manifest handling, DICOM and NIfTI loading, preprocessing, augmentation, balanced sampling, and provenance audits. code/hacon/models contains the volumetric encoder and downstream heads. code/hacon/objectives contains CSPM, scanner profiles, temperature modulation, and contrastive losses. code/hacon/training contains distributed execution, optimization, scheduling, state recording, atomic persistence, pretraining, and downstream fitting. code/hacon/evaluation contains LOHO aggregation, probes, scaling analysis, perturbation analysis, uncertainty, and statistical comparisons. code/hacon/metrics contains segmentation, classification, calibration, and regression metrics.

## Output integrity

Training records the seed, optimizer, scheduler, precision scaler, random states, sample count, best metric, and resolved epoch position. Persistence uses a temporary file followed by an atomic replacement. Evaluation tables are written as JSON or CSV through atomic replacements. Manifest audits provide deterministic SHA-256 digests of sorted records.

