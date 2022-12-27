# Mask-Controlled-Paraphrase-Generation

This repository is for the project Mask Controlled Paraphrase Generation.

## Directory
+ **main/res** - Resources including model check points, datasets, and experiment records
+ **main/src** - Source code including model structures, training pipelines, and utility functions
+ **res/data** - Need to place the data pickle file under here
+ **res/lm** - Need to place the bert-base-uncased repo under here
```
Mask-Controlled-Paraphrase-Generation
├── README.md
└── main
    ├── config.py
    ├── main.py
    ├── res
    │   ├── data
    │   ├── lm
    │   └── log
    ├── script
    │   └── quora
    │       ├── baai.sh
    │       └── narval.sh
    ├── src
    │   ├── augment
    │   │   ├── augmentors.py
    │   │   ├── base.py
    │   │   └── utils.py
    │   ├── datasets
    │   │   └── quora.py
    │   ├── models
    │   │   ├── encoder.py
    │   │   ├── lm.py
    │   │   └── tfm.py
    │   ├── trainers
    │   │   ├── base.py
    │   │   └── seq2seq.py
    │   └── utils
    │       ├── eval.py
    │       ├── helper.py
    │       └── preprocess.py
    └── requirements.txt
```

## Dependencies
+ python >= 3.10.8
+ tqdm >= 4.64.1
+ numpy >= 1.23.3
+ nltk >= 3.7
+ torch >= 1.13.0
+ transformers >= 4.24.0

## Setup
Please ensure required packages are already installed. A virtual environment is recommended.
```
$ cd Text-Editing-AS-Imitation-Game
$ cd main
$ pip install pip --upgrade
$ pip install -r requirements.txt
$ python
……import nltk
……nltk.download('omw-1.4')
……nltk.download('punkt')
……nltk.download('wordnet')
……exit()
$ cd res/lm
$ git clone https://huggingface.co/bert-base-uncased
```

## Authors
* **Ning Shi** - mrshininnnnn@gmail.com