from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class Model(ABC):
    """
    Abstract class for a machine learning model. Whenever it is needed to
    implement a new model it should inherit and implement each of its methods.
    Each inheritted model might be implemented differently but should respect
    the signature of the abstract class.
    """

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir

    @abstractmethod
    def fit(self,
            x_train: pd.Series,
            y_train: pd.Series,
            x_dev: pd.Series = None,
            y_dev: pd.Series = None):
        """
        Abstract fit method that takes training text documents `x_train` and
        their labels `y_train` and train a model. `x_dev` and `y_dev` can be
        used to obtain cross-validation insights, early stopping, or simply
        ignore them.

        parameters:
            - `x_train` (pd.Series[str]) training text documents.
            - `y_train` (pd.Series[int]) training labels.
            - `x_dev` (pd.Series[str]) dev text documents.
            - `y_dev` (pd.Series[int]) dev labels.
        """
        pass

    @abstractmethod
    def predict(self, x: pd.Series) -> np.array:
        """
        Abstract method to perform classification on samples in `x`.

        parameters:
            - `x` (pd.Series[str]) sample to predict.

        returns:
            - `y_pred` (np.array[int]) class labels for sample `x`.
        """
        pass

    @abstractmethod
    def predict_proba(self, x: pd.Series) -> np.array:
        """
        Abstract method to estimate classification probabilities on samples in
        `x`.

        parameters:
            - `x` (pd.Series[str]) sample to predict.

        returns:
            - `y_pred` (np.array of floats with n classes columns) probability
              labels for sample `x`.
        """
        pass

    @abstractmethod
    def save_model(self) -> None:
        """
        Save model weights as a pickle python file in `self.output_dir` using
        its identifier `self.model_name`.
        """
        pass

    @abstractmethod
    def load_model(self, model_dirpath: str) -> None:
        """
        Load model weights. It takes directory path `model_dirpath` where the
        model necessary data is in.

        parameters:
            - `model_dirpath` (str) Directory path where the model is saved.
        """
        pass