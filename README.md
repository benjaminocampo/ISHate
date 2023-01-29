# An In-depth Analysis of Implicit and Subtle Hate Speech Messages

This repository contains the dataset and implementation details of the paper "An
In-depth Analysis of Implicit and Subtle Hate Speech Messages" accepted at EACL
2023.

# Index

- [Installation](#installation)
- [Dataset](#dataset)
- [Models](#models)
- [Contributing](#contributing)
- [Cite us](#cite-us)

# Installation

First, make sure that `conda` is installed on your machine. You can check if it
is installed by running `conda --version` in your terminal.

Clone the repository that contains of this project onto your local machine.

Once conda is installed, navigate to the directory where the application is
located in the terminal.

Run the command `make create_environment` to create a new conda environment for
your application.

Then, run the command `make requirements` to install the necessary dependencies
for your application in the newly created environment.

Optionally, you can test the environment by running the command `make
test_environment`

Once the environment is created and the dependencies are installed, you should
be able to run the experiments.

# Dataset

The ISHate dataset split into train, dev, and test sets can be found in the
directory `./data/` as compressed parquet files. They can be easily opened with
`pandas`:

```python
import pandas as pd

train = pd.read_parquet("./data/ishate_train.parquet.gzip")
dev = pd.read_parquet("./data/ishate_train.parquet.gzip")
test = pd.read_parquet("./data/ishate_train.parquet.gzip")
```

For simplicity, when training machine learning models for implicit and subtle
detection, we reorganized the dataset in the directories `./data/implicit_task/`
and `./data/subtle_task/`. There you can also find the augmented data used in
our experiments.

# Models

Each model can be accessed through the huggingface hub.
 
# Implementations

In the directory `ISHate`, the implementation of various machine learning models
such as `BERT`, `DeBERTa`, `HateBERT`, and `USE_SVM` can be found, along with
various data augmentation methods including `AAV`, `BT`, `EDA`, `RA`, `RI`,
`RNE`, and `RSA`.

# Reproduce experiments

The directory `./experiments/` contains a notebook that obtains the results of data distribution and annotation agreement.

To train and evaluate on ISHate the same models we used in our paper, go to `./experiments/classification/` and execute the `run_train.py` script.

```shell
python run_train.py
```

Models and results are registered using MLFlow in the same directory. In order to display them, you can use the mlflow ui by running on the shell

```shell
mlflow ui
```

And opening your localhost in the port 5000: [http://127.0.0.1:5000/](http://127.0.0.1:5000/). For more information on MLFlow read [https://mlflow.org/](https://mlflow.org/)

# Contributing

We are thrilled that you are interested in contributing to our work! Your
contributions will help to make our project even better and more useful for the
community.

Here are some ways you can contribute:

- Bug reporting: If you find a bug in our code, please report it to us by
  creating a new issue in our GitHub repository. Be sure to include detailed
  information about the bug and the steps to reproduce it.

- Code contributions: If you have experience with the technologies we are using
  and would like to contribute to the codebase, please feel free to submit a
  pull request. We welcome contributions of all sizes, whether it's a small bug
  fix or a new feature.

- Documentation: If you find that our documentation is lacking or could be
  improved, we would be grateful for your contributions. Whether it's fixing
  typos, adding new examples or explanations, or reorganizing the information,
  your help is greatly appreciated.

- Testing: Testing is an important part of our development process. We would
  appreciate it if you could test our code and let us know if you find any
  issues.

- Feature requests: If you have an idea for a new feature or improvement, please
  let us know by creating a new issue in our GitHub repository.

All contributions are welcome and appreciated! We look forward to working with
you to improve our project.

# Cite us

This is a temporary BibTeX until the publication is uploaded in the ACL
`Anthology`. We will update the BibTeX entry as soon as the accepted paper is
published. You can use the following entry for the moment:

```scss
@inproceedings{OcampoEtAl2023,
  title={An In-depth Analysis of Implicit and Subtle Hate Speech Messages},
  author={Nicolas Benjamin Ocampo and Ekaterina Sviridova and Elena Cabrio and Serena Villata},
  booktitle={The 17th Conference of the European Chapter of the Association for Computational Linguistics},
  year={2023}
}
```