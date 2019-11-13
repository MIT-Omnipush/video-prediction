# Video Prediction for Omnipush

This repo hosts the code to train video prediction models on Omnipush dataset.

Two video prediction algorithms are contained:

- Stochastic Video Generation (SVG)  [[paper](https://arxiv.org/abs/1802.07687) | [code](https://github.com/edenton/svg)]
- Stochasitic Adversarial Video Prediction (SAVP) [[paper](https://arxiv.org/abs/1804.01523) | [code](https://github.com/alexlee-gk/video_prediction)]

Custom data loaders for Omnipush are provided in this implementation. 

**Disclaimer:** code for training models is borrowed from authors' implementations linked above.

## Getting Started

### Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

### Installation

- Clone this repo:

```
git clone git@github.com:yenchenlin/video-prediction.git
cd video-prediction
```

- Install dependencies

```
pip install -r requirements.txt
```

- Install [ffmpeg](https://ffmpeg.org/), used to generate GIFs for visualization.

SVG and SAVP have different dependencies, we list them separately:

#### SVG

  - PyTorch >= 0.4

#### SAVP

  - TensorFlow >= 1.9
  
### Download Omnipush

```
bash ./dataset/download_data.sh 
```

To train SAVP, pre-process the dataset into tfrecords.

```
python ./dataset/generate_tfrecords.py
```

To verify everything works correctly, `dataset` should contain the following directories.

```
dataset
├── omnipush            # raw image files, will be used for SVG
├── omnipush-tfrecords  # tfrecords, will be used for SAVP
└── ...
```

### Training

#### SVG

```
python train_svg_lp.py --dataset omnipush --model vgg --g_dim 128 --z_dim 64 --beta 0.0001 --n_past 2 --n_future 10 --n_eval 12 --channels 3 --data_root ../dataset/omnipush --log_dir ./logs/ --batch_size 32
```

#### SAVP

```
python scripts/train.py --input_dir ../dataset/omnipush-tfrecords --dataset omnipush --dataset_hparams sequence_length=12 --model savp --model_hparams_dict hparams/bair_action_free/ours_savp/model_hparams.json --output_dir ./logs/omnipush/ours_savp
```

## Citation

If you used the code from this repo, consider citing the following papers

```
@article{bauzaomnipush, 
  title={Omnipush: accurate, diverse, real-world dataset of pushing dynamics with RGB-D video}, 
  author={Bauza, Maria and Alet, Ferran and Lin, Yen-Chen and Lozano-P{\'e}rez, Tom{\'a}s and Kaelbling, Leslie P and Isola, Phillip and Rodriguez, Alberto} 
}

@article{denton2018stochastic,
  title={Stochastic video generation with a learned prior},
  author={Denton, Emily and Fergus, Rob},
  journal={ICML},
  year={2018}
}

@article{lee2018savp,
  title={Stochastic Adversarial Video Prediction},
  author={Alex X. Lee and Richard Zhang and Frederik Ebert and Pieter Abbeel and Chelsea Finn and Sergey Levine},
  journal={arXiv preprint arXiv:1804.01523},
  year={2018}
}
```

