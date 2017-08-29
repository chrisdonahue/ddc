This directory contains the code used to generate the results in the [Dance Dance Convolution paper](https://arxiv.org/abs/1703.06891).

I am in the process of reimplementing this code (in the root directory of the repository) to be more streamlined. However, you can get started with this if you are eager to play with Dance Dance Convolution. Please email me with any issues: cdonahue \[@at@\] ucsd \(.dot.\) edu

# Requirements

* tensorflow >1.0
* [essentia 2.1 beta 3](https://github.com/MTG/essentia/releases/tag/v2.1_beta3)
* numpy
* tqdm
* scipy

# Directory description

* `dataset/`: code to generate the dataset from StepMania files
* `learn/`: code to train step placement (onset) and selection (sym) models
* `scripts/`: shell scripts to build the dataset (`smd_*`) and train (`sml_*`)

# Building dataset

1. Make a directory named `data` under `~/ddc/reproducability` (or change `scripts/var.sh` to point to a different directory)
1. Under `data`, make directories `raw` and `json_filt`
1. Under `data/raw`, make directories `fraxtil` and `itg`
1. Under `data/raw/fraxil`, download and unzip:
    * [(Fraxtil) Tsunamix III](https://fra.xtil.net/simfiles/data/tsunamix/III/Tsunamix%20III%20[SM5].zip)
    * [(Fraxtil) Fraxtil's Arrow Arrangements](https://fra.xtil.net/simfiles/data/arrowarrangements/Fraxtil's%20Arrow%20Arrangements%20[SM5].zip)
    * [(Fraxtil) Fraxtil's Beast Beats](https://fra.xtil.net/simfiles/data/beastbeats/Fraxtil's%20Beast%20Beats%20[SM5].zip)
1. Under `data/raw/itg`, download and unzip:
    * [(ITG) In The Groove](http://stepmaniaonline.net/downloads/packs/In%20The%20Groove%201.zip)
    * [(ITG) In The Groove 2](http://stepmaniaonline.net/downloads/packs/In%20The%20Groove%202.zip)
1. Navigate to `scripts/`
1. Parse `.sm` files to JSON: `./all.sh ./smd_1_extract.sh`
1. Filter JSON files (removing mines, etc.): `./all.sh ./smd_2_filter.sh`
1. Split dataset 80/10/10: `./all.sh ./smd_3_dataset.sh`
1. Analyze dataset (e.g.): `./smd_4_analyze.sh fraxtil`

# Running training

1. Navigate to `scripts/`
1. Extract features: `./all.sh ./sml_onset_0_extract.sh`
1. Generate chart `.pkl` files (this may take a while): `./all.sh ./sml_onset_1_chart.sh`
1. Train a step placement (onset detection) model on a dataset: `./sml_onset_2_train.sh fraxtil`
1. Train a step selection (symbolic) model on a dataset: `./sml_sym_2_train.sh fraxtil`
