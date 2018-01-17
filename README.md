# Residual Connected Sentence Encoder
This is a repo for Residual-connected sentence encoder for NLI.

Try to follow the instruction below to successfully run the experiment.

1.Download the additional `data.zip` file, unzip it and place it at the root directory of this repo.
Link for download `data.zip` file: [*DropBox Link*](https://www.dropbox.com/sh/kq81vmcmwktlyji/AADRVQRh9MdcXTkTQct7QlQFa?dl=0)

2.This repo is based on an old version of `torchtext`, the latest version of `torchtext` is not backward-compatible.
We provide a link to download the old `torchtext` that should be used for this repo. Link: [*old_torchtext*](https://www.dropbox.com/sh/n8ipkm1ng8f6d5u/AADg4KhwQMwz4xFkVJafgUMma?dl=0)

3.Install the requirement:
```
python 3.6

torchtext # The one you just download. Or you can use the latest torchtext by fixing the SNLI path problem.
pytorch == 0.2.0
fire
tqdm
numpy
spacy
```

For the installation of torchtext, you can run the following script in the downloaded `torchtext_backward_compatible` directory (in step 2) using the python interpreter of your environment:
```
python setup.py install
```

To fully install spacy, you will need to run the following script.
```
pip install -U spacy
python -m spacy download en
```

Optionally, you can try to match the pip freeze file below to set up the same experiment environment.
```
certifi==2017.11.5
chardet==3.0.4
cymem==1.31.2
cytoolz==0.8.2
dill==0.2.7.1
en-core-web-sm==2.0.0
fire==0.1.2
ftfy==4.4.3
html5lib==1.0.1
idna==2.6
msgpack-numpy==0.4.1
msgpack-python==0.5.1
murmurhash==0.28.0
numpy==1.14.0
pathlib==1.0.1
plac==0.9.6
preshed==1.0.0
PyYAML==3.12
regex==2017.4.5
requests==2.18.4
six==1.11.0
spacy==2.0.5
termcolor==1.1.0
thinc==6.10.2
toolz==0.9.0
torch==0.2.0.post3
torchtext==0.1.1
tqdm==4.19.5
ujson==1.35
urllib3==1.22
wcwidth==0.1.7
webencodings==0.5.1
wrapt==1.10.11
```

4.There is a directory called `saved_model` At the root directory of this repo:
This directory will be used for saving the models that produce best dev result.

Before running the experiments, make sure that the structure of this repo should be something like below.
```
.
├── config.py
├── data
│   ├── multinli_0.9
│   │   ├── multinli_0.9_dev_matched.jsonl
│   │   ├── multinli_0.9_dev_mismatched.jsonl
│   │   ├── multinli_0.9_test_matched_unlabeled.jsonl
│   │   ├── multinli_0.9_test_mismatched_unlabeled.jsonl
│   │   └── multinli_0.9_train.jsonl
│   ├── saved_embd.pt
│   └── snli_1.0
│       ├── README.txt
│       ├── snli_1.0_dev.jsonl
│       ├── snli_1.0_dev.txt
│       ├── snli_1.0_test.jsonl
│       ├── snli_1.0_test.txt
│       ├── snli_1.0_train.jsonl
│       └── snli_1.0_train.txt
├── model
│   └── res_encoder.py
├── saved_model
│   └── trained_model_will_be_saved_in_here.txt
├── setup.sh
├── torch_util.py
└── util
    ├── data_loader.py
    ├── dataset_util.py
    ├── __init__.py
    ├── mnli.py
    └── save_tool.py
```

5.Start training by run the script in the root directory.
```
source setup.sh
python model/res_encoder.py train_snli
```

6.After training completed, there will be a folder created by the script in the `saved_model` directory.
The parameters of the model will be saved in that folder. The path of the model will be something like:
```
$DIR_TMP/saved_model/(TIME_STAMP)_[600,600,600]-3stack-bilstm-maxout-residual/saved_params/(YOUR_MODEL_WITH_DEV_RESULT)
```
Remember to change the bracketed part to the actual file name on your computer.

7.Now, you can evaluate the model on dev set again by running the script below.
```
python model/res_encoder.py eval (PATH_OF_YOUR_MODEL) dev # for evaluation on dev set
python model/res_encoder.py eval (PATH_OF_YOUR_MODEL) test # for evaluation on test set
```

**Pretrained Model:**   
We also provide a link to download the [*pretrained model*](https://www.dropbox.com/s/raa29iwpkv2xldh/pretrained_model_dev%2887.00%29?dl=0).  
After downloading the pretrained model, you can run the script in step 7 for evaluation, however you need to keep the default parameter for pytorch to load the pretrained model.
