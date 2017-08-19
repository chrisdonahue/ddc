# Dance Dance Convolution

Dataset for [Dance Dance Convolution](https://arxiv.org/abs/1703.06891).

This repository will eventually contain Tensorflow code for models from the paper. For now, it only has code for creating the dataset from separately-hosted ZIP files.

## Usage

To generate a JSON version of the dataset, download any of the following packs to a directory:

* [(Fraxtil) Tsunamix III](https://fra.xtil.net/simfiles/data/tsunamix/III/Tsunamix%20III%20[SM5].zip)
* [(Fraxtil) Fraxtil's Arrow Arrangements](https://fra.xtil.net/simfiles/data/arrowarrangements/Fraxtil's%20Arrow%20Arrangements%20[SM5].zip)
* [(Fraxtil) Fraxtil's Beast Beats](https://fra.xtil.net/simfiles/data/beastbeats/Fraxtil's%20Beast%20Beats%20[SM5].zip)
* [(ITG) In The Groove](http://stepmaniaonline.net/downloads/packs/In%20The%20Groove%201.zip)
* [(ITG) In The Groove 2](http://stepmaniaonline.net/downloads/packs/In%20The%20Groove%202.zip)

The following script will generate the entire dataset used in the paper:

```sh
export DDCDIR=~/ddc
cd ${DDCDIR}
mkdir out
python -m dataset.extract ${DDCDIR}/out ${DDCDIR}/packs/*.zip
python -m dataset.split ${DDCDIR}/out
python -m dataset.filter ${DDCDIR}/out/*.txt
```

You can generate an audio preview of any chart to verify timing and alignment against the song. Run the following script then open both the song and chart preview wav in audacity:

```sh
mkdir previews
python -m dataset.preview_wav ${DDCDIR}/out/Fraxtil/TsunamixIII/TsunamixIII_HotPursuit_Remix_.filt.json ${DDCDIR}/previews/TsunamixIII_HotPursuit_Remix_.wav
```

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

