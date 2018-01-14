# multiNLI_encoder
This is a repo for Residual-connected sentence encoder for NLI.

Try to follow the instruction below to successfully run the experiment.

1.Download the additional `data.zip` file, unzip it and place it at the root directory of this repo.
Link for download `data.zip` file: [*DropBox Link*](https://www.dropbox.com/sh/kq81vmcmwktlyji/AADRVQRh9MdcXTkTQct7QlQFa?dl=0)

2.This repo is based on an old version of `torchtext`, the latest version of `torchtext` is not backward-compatible.
We provide a link to download the old `torchtext` that should be used for this repo. Link: [*old_torchtext*](https://www.dropbox.com/sh/n8ipkm1ng8f6d5u/AADg4KhwQMwz4xFkVJafgUMma?dl=0)

3.Install the required package below:
```
torchtext # The one you just download. Or you can use the latest torchtext by fixing the SNLI path problem.
pytorch
fire
tqdm
numpy
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