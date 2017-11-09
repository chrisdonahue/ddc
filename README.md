# Dance Dance Convolution

Dance Dance Convolution is an automatic choreography system for Dance Dance Revolution (DDR), converting raw audio into playable dances.

<p align="center">
    <img src="docs/fig1.png" width="650px"/>
</p>

This repository contains the code used to produce the dataset and results in the [Dance Dance Convolution paper](https://arxiv.org/abs/1703.06891). You can find a live demo of our system [here](http://deepx.ucsd.edu/ddc) as well as an example [video](https://www.youtube.com/watch?v=yUc3O237p9M).

The `Fraxtil` and `In The Groove` datasets from the paper are amalgamations of three and two StepMania "packs" respectively. Instructions for downloading these packs and building the datasets can be found below.

This is a streamlined version of the legacy code used to produce our paper (which uses outdated libraries). The legacy code is available at `master_v1` for reproducability.

Please email me with any issues: cdonahue \[@at@\] ucsd \(.dot.\) edu

## Attribution
If you use this dataset in your research, cite via the following BibTex:

```
@inproceedings{donahue2017dance,
  title={Dance Dance Convolution},
  author={Donahue, Chris and Lipton, Zachary C and McAuley, Julian},
  booktitle={Proceedings of the 34th International Conference on Machine Learning},
  year={2017},
}
```

# Requirements

* tensorflow >1.0
* numpy
* tqdm
* scipy

# Directory description

* `ddc/`: Core library with dataset extraction and training code
* `scripts/`: shell scripts to build the dataset (`smd_*`) and train (`sml_*`)

# Building dataset

1. `$ git clone git@github.com:chrisdonahue/ddc.git`
1. `cd ddc`
1. `$ sudo pip install -e .` (installs as editable library)
1. `$ export SM_DATA_DIR=~/ddc/data` (or another directory of your choosing)
1. `$ mkdir $SM_DATA_DIR`
1. `$ cd $SM_DATA_DIR`
1. Download game data
    * [(Fraxtil) Tsunamix III](https://fra.xtil.net/simfiles/data/tsunamix/III/Tsunamix%20III%20[SM5].zip)
    * [(Fraxtil) Fraxtil's Arrow Arrangements](https://fra.xtil.net/simfiles/data/arrowarrangements/Fraxtil's%20Arrow%20Arrangements%20[SM5].zip)
    * [(Fraxtil) Fraxtil's Beast Beats](https://fra.xtil.net/simfiles/data/beastbeats/Fraxtil's%20Beast%20Beats%20[SM5].zip)
    * [(ITG) In The Groove](http://stepmaniaonline.net/downloads/packs/In%20The%20Groove%201.zip)
    * [(ITG) In The Groove 2](http://stepmaniaonline.net/downloads/packs/In%20The%20Groove%202.zip)
1. `cd ~/ddc/scripts`
1. `./smd.sh` (extracts dataset)
