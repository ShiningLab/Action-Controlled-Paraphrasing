# Action-Controlled-Paraphrasing

This repository is for the project Action Controlled Paraphrasing.

## Directory
+ **main/res** - Resources including model check points, datasets, and experiment records
+ **main/src** - Source code including model structures, training pipelines, and utility functions
+ **res/data** - Need to place the data pickle file under here
+ **res/lm** - Need to place the bert-base-uncased repo under here

## Dependencies
+ nltk>=3.8.1
+ rich>=13.7.0
+ wandb>=0.16.0
+ torch>=2.1.1
+ evaluate>=0.4.1
+ lightning>=2.1.2
+ transformers>=4.35.2

## Setup
Please ensure required packages are already installed. A virtual environment is recommended.
```
$ cd Action-Controlled-Paraphrasing
$ pip install pip --upgrade
$ pip install -r requirements.txt
$ python
import nltk
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('wordnet')
exit()
$ cd main
$ mkdir res
$ cd res
$ mkdir lm
$ cd lm
$ git clone https://huggingface.co/bert-base-uncased
...
$ cd ../..
$ python main.py
```

## Authors
* **Ning Shi** - mrshininnnnn@gmail.com
