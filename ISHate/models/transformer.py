from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          TextClassificationPipeline, TrainingArguments,
                          Trainer, DataCollatorWithPadding, set_seed)
from datasets import Dataset
from ISHate.models.base import Model
import pandas as pd
import numpy as np
import os
import warnings


class TransformerModel(Model):
    """
    Huggingface Transformer model for classification such as BERT, DeBERTa,
    RoBERTa, etc.

    parameters:
        - `output_dir` (str) Directory path where the model outputs will be
          recorded. That is weights, predictions, etc.

        - `model_name` (str) Identifier of the model. It is used to recognize an
          instance of the class. For example, if multiple runs are executed with
          different parameters, `model_name` can be used to assign a different
          name. Also, when saving an instance of the model, it will create a
          directory using this parameters as its name and will be saved in
          `output_dir`.

        - `huggingface-path` (str) the name of the model in the hub of
          huggingface. For example: `bert-base-uncased` or
          `microsoft/deberta-v3-large`.

        - `checkpoint-path` (str) [optional] path to a huggingface checkpoint
        directory containing its configuration.

        - `epochs` (int) number of epochs for training the transformer.

        - `batch-size` (int) batch size used for training the transformer.

        - `random_state` (int) integer number to initialize the random state
          during the training process.

        - `lr` (float) learning rate for training the transformer.

        - `weight-decay` (float) weight decay penalty applied to the
          transformer.

        - `device` (str) Use `cpu` or `gpu`.
    """

    def __init__(self,
                 huggingface_path: str = "bert-base-uncased",
                 checkpoint_path: str = None,
                 epochs: int = 4,
                 batch_size: int = 32,
                 random_state: int = 42,
                 lr: float = 2e-5,
                 weight_decay: float = 0.01,
                 num_labels: int = 2,
                 output_dir: str = "./default_output_dir",
                 device: str = "cpu") -> None:
        super(TransformerModel, self).__init__(output_dir)

        set_seed(random_state)

        # Load model from hugginface hub.
        model = AutoModelForSequenceClassification.from_pretrained(
            huggingface_path,
            num_labels=num_labels,
            output_attentions=False,
            output_hidden_states=False,
        )

        # Load tokenizer from huggingface hub.
        tokenizer = AutoTokenizer.from_pretrained(huggingface_path,
                                                  do_lower_case=True)
        # Set class attributes.
        self.model = model
        self.tokenizer = tokenizer
        self.checkpoint_path = checkpoint_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.num_labels = num_labels
        self.args = None
        self.trainer = None

    def set_training_args(self):
        self.args = TrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            learning_rate=self.lr,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            weight_decay=self.weight_decay,
            seed=self.random_state,
            #device=self.device,
            #data_seed=self.random_state,
            optim="adamw_hf")

    def tokenize(self, example: str):
        """
        Tokenize a sentence using the model tokenizer.
        """
        return self.tokenizer(example["text"], truncation=True)

    def build_loader(self, sentences: pd.Series, labels: pd.Series = None):
        """
        Create a Dataset loader from huggingface tokenizing each sentence.

        parameters:
            - `sentences` (pd.Series[str])
            - `labels` (pd.Series[int])
        """
        dataset = Dataset.from_dict({"text": sentences}
                                    | ({
                                        "label": labels
                                    } if labels is not None else {}))
        return dataset.map(self.tokenize, batched=True)

    def fit(self,
            x_train: pd.Series,
            y_train: pd.Series,
            x_dev: pd.Series = None,
            y_dev: pd.Series = None) -> None:
        """
        Fit method that takes training text documents `x_train` and their labels
        `y_train` and train a transformer based model. In this case the `x_dev`
        and `y_dev` are used to evaluate the model in each epoch. When saving
        the model, train and dev losses are saved too.

        parameters:
            - `x_train` (pd.Series[str]) training text documents.
            - `y_train` (pd.Series[int]) training labels.
            - `x_dev` (pd.Series[str]) dev text documents.
            - `y_dev` (pd.Series[int]) dev labels.
        """
        self.set_training_args()

        # Create data collator.
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer,
                                                padding=True)

        # Create dataset loaders for train and dev sets.
        train = self.build_loader(sentences=x_train, labels=y_train)
        dev = self.build_loader(sentences=x_dev, labels=y_dev)

        # Move huggingface model to the device indicated.
        self.model = self.model.to(self.device)

        # Instance huggingface Trainer.
        self.trainer = Trainer(model=self.model,
                               args=self.args,
                               train_dataset=train,
                               eval_dataset=dev,
                               tokenizer=self.tokenizer,
                               data_collator=data_collator)

        # If there is any checkpoint provided, training is resumed from it.
        if self.checkpoint_path is not None:
            self.trainer.train(self.checkpoint_path)
        else:
            self.trainer.train()

    def predict_proba(self, x: pd.Series) -> np.array:
        """
        Estimate classification probabilities on samples in `x`.

        parameters:
            - `x` (pd.Series[str]) sample to predict.

        returns:
            - `y_pred` (np.array of floats with n classes columns) probability
              labels for sample `x`.
        """
        # Use text classification pipeline to make predictions.
        pipe = TextClassificationPipeline(model=self.model,
                                          tokenizer=self.tokenizer,
                                          return_all_scores=True,
                                          framework="pt")
        preds = pipe(x.tolist())
        y_prob = np.array([[pred[i]["score"] for i in range(self.num_labels)]
                           for pred in preds])
        return y_prob

    def predict(self, x: pd.Series) -> np.array:
        """
        Perform classification on samples in `x`.

        parameters:
            - `x` (pd.Series[str]) sample to predict.

        returns:
            - `y_pred` (np.array[int]) class labels for sample `x`.
        """
        y_prob = self.predict_proba(x)
        y_pred = np.argmax(y_prob, axis=1)
        return y_pred

    def save_model(self):
        """
        Save model weights and its configuration in `self.output_dir`. It
        follows huggingface save standards so the model can be re-loaded using
        huggingface `from_pretrained()` functionality.
        """
        if self.trainer is not None:
            os.makedirs(f"{self.output_dir}/model", exist_ok=True)
            self.trainer.save_model(output_dir=f"{self.output_dir}/model")
        else:
            warnings.warn(
                "Method ignored. Trying to save model without training it."
                "Please use `fit` before `save_model`",
                UserWarning,
            )

    def load_model(self, model_dirpath):
        """
        Load model weights. It takes directory path `model_dirpath` where the
        model necessary data is in.

        parameters:
            - `model_dirpath` (str) Directory path where the model is saved.
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dirpath)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dirpath)

    def embed(self, x: pd.Series) -> np.array:
        inputs = self.tokenizer(x.tolist(),
                                truncation=True,
                                padding=True,
                                return_tensors="pt")
        outputs = self.model(**inputs, output_hidden_states=True)

        # Get the last hidden state
        last_hidden_states = outputs.hidden_states[-1]

        # Get only the CLS token for each instance in `x` (the one used for classification).
        cls = last_hidden_states[:, 0, :]

        # Detach Pytorch tensor to Numpy array.
        return cls.detach().numpy()
