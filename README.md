# W-TransUNet for simulated sparse-view CT

W-TransUNet is an image-domain post-processing model for sparse-view CT FBP
images. It concatenates FBP with single-level Haar detail maps, mixes these
channels through a residual convolutional network at full resolution, and
applies residual TransUNet refinement. The retained study evaluates separate
trained configurations at 1000, 500, 250, 125, and 50 views on LoDoPaB-CT.

This revision source collection accompanies the manuscript
*Full-resolution input mixing and transformer refinement for simulated
sparse-view CT post-processing*. It contains 36 Python source files, including
model definitions, evaluation programs, statistical analysis, and archival
training sources. Data, caches, trained checkpoints, frozen protocol files,
and derived-result records are external assets.

## Repository structure

```text
w-transunet-svct/
├── README.md
├── requirements.txt
├── .gitignore
├── third_party_licenses.py
├── src/
│   ├── evaluate_cached.py           # Retained five-view cached evaluator
│   ├── inference.py                 # Later evaluation and input-control stream
│   ├── compute_metrics_models.py    # Existing model/cost profiling program
│   ├── vit_seg_configs.py
│   ├── vit_seg_modeling.py
│   ├── vit_seg_modeling_resnet_skip.py
│   ├── unet.py
│   ├── wavelet_ops.py
│   └── repro.py
├── analysis/
│   ├── verify_deposit.py            # Verify external deposited numerical records
│   ├── g1_input_control_stats.py
│   └── cluster_paired_stats.py
├── evaluation/
│   ├── finite_correction/           # Frozen projection-correction workflow
│   └── lpd_comparator/              # Learned primal-dual runner/statistics/shim
└── archival_training/
    ├── paper_lineage/               # Retained original-paper source lineage
    └── later_control_source/        # Later training/input-control source
```

Run the examples below from the repository root. Paths such as
`/path/to/research-assets` are locations on your own machine.

## Environment setup

The shared dependencies in `requirements.txt` support cached evaluation and
stored-result statistics. Their versions follow the retained study environment;
the file does not lock every transitive dependency or the platform/CUDA build.
Use Python 3.9-3.11. The examples use Python 3.11.

### macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Linux with CPU PyTorch

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

### Linux with CUDA PyTorch

Activate a Python 3.9-3.11 environment first. This example selects the CUDA 12.4
build; choose the build appropriate for your NVIDIA driver using the
[official PyTorch 2.5.1 installation instructions](https://pytorch.org/get-started/previous-versions/#v251).

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

The cached evaluator selects CUDA, then Apple MPS, then CPU according to device
availability. Installing the shared dependencies does not install the CT
operator stack described next.

### Additional CT operator environment

`src/inference.py`, the correction workflow, the LPD comparator, and archival
training require DIVal and related CT dependencies. The retained numerical
environment used Python 3.9.23, PyTorch 2.5.1, torchvision 0.20.1, NumPy 1.26.4,
SciPy 1.13.1, ODL 0.8.2, ASTRA 2.3.1, and an A100 80 GB PCIe GPU.

The correction/LPD programs use `astra_cuda`; run those workflows on a compatible
NVIDIA CUDA host. Apple MPS does not supply that backend. In the chosen CT
environment, install the shared requirements and additional recorded packages:

```bash
python -m pip install -r requirements.txt
python -m pip install odl==0.8.2 astra-toolbox==2.3.1 \
  h5py==3.14.0 scikit-image==0.24.0 \
  tensorboard==2.20.0 tensorboardX==2.6.4
```

Use a separate DIVal source directory. The retained environment records revision
`dd86f03733593dd6e226263aa1d3abec961c7881` of
[DIVal](https://github.com/jleuschn/dival):

```bash
git clone https://github.com/jleuschn/dival.git /path/to/dival
git -C /path/to/dival checkout dd86f03733593dd6e226263aa1d3abec961c7881
python -m pip install -e /path/to/dival
python -m pip check
```

Archival training sources also reference custom DIVal network modules. Obtain
the original study's custom DIVal tree to use those sources; an upstream checkout
alone does not restore the local modifications. Package installation is not
evidence that the historical numerical environment has been reproduced.

## External data, checkpoints, and configuration

| Asset | How to obtain or supply it | Used by |
| --- | --- | --- |
| LoDoPaB-CT v1.0.0 | Download the [official dataset](https://zenodo.org/records/3384092) and follow [DIVal dataset configuration](https://jleuschn.github.io/docs.dival/dival.datasets.lodopab_dataset.html). | DIVal-based inference, correction, LPD, training |
| View-specific FBP caches and reference cache | Request the retained study caches from the maintainer, or prepare condition-matched caches with the documented study protocol and preserve official test ordering. | Cached evaluation |
| Fifteen evaluated checkpoints | Request the exact retained U-Net, TransUNet, and W-TransUNet checkpoint for each of the five view counts. | Historical five-view evaluation |
| Input-control and LPD checkpoints | Request the matching frozen checkpoints and their configuration/training manifests. | Later input-control and LPD evaluation |
| ImageNet `R50+ViT-B_16.npz` initialization | Follow the pretrained-model instructions in [TransUNet](https://github.com/Beckschen/TransUNet). | Archival initialization and correction CLI |
| Patient maps, frozen protocols, asset manifests, and validation locks | Request the matching reproducibility records; keep their identifiers and hashes with the evaluated checkpoints. | Statistics, correction, LPD |
| Deposited CSV/JSON numerical results | Request the matching release bundle containing `configs/` and `derived_results/`. | Deposit verification |

Study-specific caches, weights, protocols, and numerical deposits are not
distributed in this repository, and this collection does not provide a public
download URL for them. Contact **Sunghoon Choi** at **schoi@etri.re.kr** to request
the materials and confirm their access/distribution terms. Include the repository
commit, view counts, workflow, and requested files in the request. A download of
the official LoDoPaB-CT dataset alone does not provide the study's trained models.

A suggested external layout for cached evaluation is:

```text
/path/to/research-assets/
├── cache/
│   ├── cache_lodopab_test_gt.npy
│   └── {views}angle/cache_lodopab_test_fbp.npy
├── checkpoints/
│   ├── lodopab_unet/{views}angle/epoch_150.pth
│   ├── lodopab_transunet/{views}angle/epoch_150.pth
│   └── lodopab_wavres_transunet_refine/{views}angle/epoch_150.pth
└── release_bundle/
    ├── configs/
    └── derived_results/
```

`{views}` stands for 1000, 500, 250, 125, or 50. Preserve checkpoint identity:
the retained W-TransUNet 250-view `epoch_150.pth` was previously identified as a
renamed epoch-149 best checkpoint. Substituting a different `best_model.pth` can
change the evaluated configuration. Initialization NPZ weights are optional for
the cached evaluator when a complete checkpoint is loaded strictly.

## Cached five-view evaluation

This workflow uses explicit FBP/reference arrays and three complete model
checkpoints. It does not require DIVal, ODL, or ASTRA. The arrays must have the
same sample order; match each checkpoint's model configuration and flags.
The retained evaluation uses central 352 x 352 crops of native 362 x 362 images.

```bash
ASSET_ROOT=/path/to/research-assets
python src/evaluate_cached.py \
  --angle 125 \
  --cache_fbp "$ASSET_ROOT/cache/125angle/cache_lodopab_test_fbp.npy" \
  --cache_gt "$ASSET_ROOT/cache/cache_lodopab_test_gt.npy" \
  --ckpt_unet "$ASSET_ROOT/checkpoints/lodopab_unet/125angle/epoch_150.pth" \
  --ckpt_transunet "$ASSET_ROOT/checkpoints/lodopab_transunet/125angle/epoch_150.pth" \
  --ckpt_wavres "$ASSET_ROOT/checkpoints/lodopab_wavres_transunet_refine/125angle/epoch_150.pth" \
  --img_size 352 --batch 16 --residual_out --unet_no_sigmoid \
  --png_n 0 --out_dir outputs/cached/125
```

Repeat with the corresponding arrays and checkpoints for each other view count.
The evaluator exports per-sample metrics and aggregate reports; use `--png_n`
or `--save_all_png` when image output is needed. Inspect all options with
`python src/evaluate_cached.py --help`.

## Verify deposited numerical results

The verifier reads an external bundle of saved results. Pass its root explicitly
because the program was originally arranged within a different deposit tree.

```bash
python analysis/verify_deposit.py --root /path/to/release_bundle
```

Add `--quick` to skip bootstrap calculations. This checks retained numerical
records and the assertions encoded in that verifier; it does not re-run network
inference or regenerate the revised manuscript's numbering/text.

For the retained input-control statistics, supply the combined per-image CSV
and matching patient map:

```bash
python analysis/g1_input_control_stats.py \
  --combined /path/to/g1_per_image.csv \
  --patient-ids /path/to/patient_ids_rand_test.csv \
  --out-json outputs/g1/stats.json \
  --out-csv outputs/g1/stats.csv
```

## Later evaluation, correction, and LPD

`src/inference.py` uses the later DIVal-reference evaluation stream. Set the
dataset location explicitly with `LODOPAB_DATA` or `LODOPAB_PATH`; the former
takes precedence. Its results and metrics belong to that stream and should be
reported separately from the historical cached five-view evaluation.

The correction runner requires `frozen_dc_protocol.json` and
`expected_assets.json` beside the scripts in `evaluation/finite_correction/`.
Supply the retained dataset manifest, patient map, caches, model checkpoints,
validation lock, and associated sweep/validation records. Its `--wtu-code`
argument must point to this repository root, which contains `src/`.

The LPD runner requires a DIVal source directory (`--dival-root`), fixed LPD
checkpoint, training manifest, FBP cache, and
`lodopab_learnedpd_reference_hyper_params.json` supplied via `--hyper-params`.
The retained `lpd_runner_shim.py` applies the recorded ODL compatibility patch:

```bash
python evaluation/lpd_comparator/lpd_runner_shim.py \
  evaluation/lpd_comparator/lpd_sparseview_runner.py --help
```

Use the retained, validation-locked test settings and corresponding manifests.
Record any local path mapping separately and preserve the meaning of frozen
asset/lock hashes; do not regenerate them merely to bypass a mismatch. Some
analysis helpers retain study-specific archive layouts; inspect their `--help`
and supply the corresponding external records.

## Training sources and reproducibility scope

`archival_training/paper_lineage/` and `archival_training/later_control_source/`
preserve different source lineages. These scripts require their original custom
DIVal environment, initialization weights, caches, and configuration. They are
archival recipes, rather than a fully recovered command sequence for every
evaluated checkpoint.

The W-TransUNet 125-view initial epochs 1-60 source/state linkage and complete
historical U-Net/TransUNet command/log linkage were not recovered. The collection
preserves model and metric implementations but does not establish exact original
training reproduction or training-seed variability. New installation examples
are not a claim that the historical experiments have been re-executed. Profiling
on a different device/environment produces new measurements; it does not replace
the retained A100 forward-cost measurements.

For new results, record the code commit (`git rev-parse HEAD`), package versions,
device, command, checkpoint identity, and input/protocol identifiers. Compare
complete evaluated configurations within their documented evaluation stream.

## License and contact

Third-party notices and license texts are preserved in
[`third_party_licenses.py`](third_party_licenses.py): DIVal U-Net uses MIT, and
the TransUNet-derived files use Apache License 2.0. Retain those notices when
redistributing the corresponding files.

A repository-wide license for the authors' original contributions has not been
selected for this collection. No such grant should be inferred from the
third-party notices. Contact Sunghoon Choi at **schoi@etri.re.kr** for code-use
questions and study-specific external assets. When citing the implementation,
include the repository URL and the exact commit used.
