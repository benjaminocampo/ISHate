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

Each model can be accessed through the Huggingface Hub. The first three models (BERT, DeBERTa, HateBERT)
can be loaded using the Huggingface Transformers library, while the SVM model is saved as a pickle file
and can be loaded with the Python `pickle` module.

## Loading Transformer Models

To load the transformer-based models (BERT, HateBERT, and DeBERTa), you can use the following code:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Example for loading the DeBERTa model with the 'ri' augmentation method for the 'subtle_task'
tokenizer = AutoTokenizer.from_pretrained("BenjaminOcampo/task-subtle_task__model-deberta__aug_method-ri")
model = AutoModelForSequenceClassification.from_pretrained("BenjaminOcampo/task-subtle_task__model-deberta__aug_method-ri")

# Alternatively, use a pipeline for easy inference
from transformers import pipeline

pipe = pipeline("text-classification", model="BenjaminOcampo/task-subtle_task__model-deberta__aug_method-ri")

# To use the model for prediction
input_text = "Your input text here"
predictions = pipe(input_text)
print(predictions)
```

SVM models are stored as a pickle files. To load and use this model, you'll need to use the pickle module in Python as shown below:

```python
import pickle

# Load SVM model from a pickle file
with open('path_to_your_svm_model.pkl', 'rb') as model_file:
    svm_model = pickle.load(model_file)

# To predict with the SVM model
# Make sure to preprocess your input features the same way as when the model was trained
input_features = preprocess_input("Your input text or features here")
predictions = svm_model.predict(input_features)
print(predictions)
```

## Model Naming Convention

The models are saved with the following naming convention on my Huggingface profile:
`task-{implicit_task, subtle_task}__model-{bert, hatebert, deberta, svm}__aug_method-{aav, all, bt, eda, gm, gm-revised, ra, ri, rne, rsa, None}`

Replace the placeholders with the specific task, model, and augmentation method you wish to use.

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

```tex
@inproceedings{ocampo-etal-2023-depth,
    title = "An In-depth Analysis of Implicit and Subtle Hate Speech Messages",
    author = "Ocampo, Nicolas  and
      Sviridova, Ekaterina  and
      Cabrio, Elena  and
      Villata, Serena",
    booktitle = "Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.eacl-main.147",
    pages = "1997--2013",
    abstract = "The research carried out so far in detecting abusive content in social media has primarily focused on overt forms of hate speech. While explicit hate speech (HS) is more easily identifiable by recognizing hateful words, messages containing linguistically subtle and implicit forms of HS (as circumlocution, metaphors and sarcasm) constitute a real challenge for automatic systems. While the sneaky and tricky nature of subtle messages might be perceived as less hurtful with respect to the same content expressed clearly, such abuse is at least as harmful as overt abuse. In this paper, we first provide an in-depth and systematic analysis of 7 standard benchmarks for HS detection, relying on a fine-grained and linguistically-grounded definition of implicit and subtle messages. Then, we experiment with state-of-the-art neural network architectures on two supervised tasks, namely implicit HS and subtle HS message classification. We show that while such models perform satisfactory on explicit messages, they fail to detect implicit and subtle content, highlighting the fact that HS detection is not a solved problem and deserves further investigation.",
}
```
