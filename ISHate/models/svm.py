from sklearn.svm import SVC
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from ISHate.models.base import Model
import tensorflow_hub as hub
import pickle
import pandas as pd
import numpy as np


class USETransformer(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn wrapper encoder/transformer that implements Universal
    Sentence Encoder. It follows scikit-learn conventions to be used in
    scikit-learn pipelines.
    """

    def fit(self, X, y):
        """
        Dummy fit implementation that implements identity function and
        passthrough its own instance classifier.
        """
        return self

    def transform(self, X):
        """
        Encode text documents and returns an array like of features.
        """
        module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        encode = hub.load(module_url)
        return encode(X)


class USE_SVM(Model):
    """
    Support Vector Machine with Universal Sentence Encoder for codification.

    parameters:
        - `output_dir` (str) Directory path where the model outputs will be
          recorded. That is weights, predictions, etc.

        - `model_name` (str) Identifier of the model. It is used to recognize an
          instance of the class. For example, if multiple runs are executed with
          different parameters, `model_name` can be used to assign a different
          name. Also, when saving an instance of the model, it will create a
          directory using this parameters as its name and will be saved in
          `output_dir`.

        - `C` (float) Regularization parameter. The strength of the
          regularization is inversely proportional to C. Must be strictly
          positive. The penalty is a squared l2 penalty.

        - `kernel` (str) Specifies the kernel type to be used in the algorithm.
          If none is given, `rbf` will be used:
            - `linear`
            - `poly`
            - `rbf`
            - `sigmoid`
            - `precomputed`

        - `gamma` (float) Kernel coefficient for `rbf`, `poly` and `sigmoid`.

        - `probability` (bool) Whether to enable probability estimates.

        - `verbose` (bool) Enable verbose output during SVM training.

        - `class-weight` (bool) Set the parameter C of class i to
          class_weight[i]*C for SVC. If not given, all classes are supposed to
          have weight one. Good for unbalanced datasets.

        - `random_state` (int) Controls the pseudo random number generation.
    """

    def __init__(self,
                 output_dir: str = "./default_output_dir",
                 C: float = 1.0,
                 kernel: str = "rbf",
                 degree: int = 3,
                 gamma: str = "scale",
                 probability: bool = True,
                 verbose: bool = True,
                 class_weight: bool = True,
                 random_state: int = 0) -> None:
        # Define attributes.
        super().__init__(output_dir)
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.probability = probability
        self.verbose = verbose
        self.class_weight = class_weight
        self.random_state = random_state

        # Instance Universal Sentence Encoder. Note that is an custom
        # scikit-learn transformer.that can be used with the Pipeline
        # scikit-learn class.
        self.use = USETransformer()

        # Instance Support Vector Machine algorithm from scikit-learn.
        self.svm = SVC(C=C,
                       kernel=kernel,
                       degree=degree,
                       gamma=gamma,
                       probability=probability,
                       verbose=verbose,
                       class_weight="balanced" if class_weight else None,
                       random_state=random_state)

        # Make a scikit-learn pipeline combining the Universal Sentence Encoder,
        # and SVM.
        self.model = make_pipeline(self.use, self.svm)

    def fit(self,
            x_train: pd.Series,
            y_train: pd.Series,
            x_dev: pd.Series = None,
            y_dev: pd.Series = None) -> None:
        """
        Fit method that takes training text documents `x_train` and their labels
        `y_train` and train the pipeline USE + SVM. In this case the `x_dev` and
        `y_dev` sets are not used as dev sets in scikit-learn algorithms do not
        use early stopping criterias. All the series need to have the same
        shape.

        parameters:
            - `x_train` (pd.Series[str]) training text documents.
            - `y_train` (pd.Series[int]) training labels.
            - `x_dev` (pd.Series[str]) dev text documents.
            - `y_dev` (pd.Series[int]) dev labels.
        """
        self.model.fit(x_train, y_train)

    def predict(self, x: pd.Series) -> np.array:
        """
        Perform classification on samples in `x`.

        parameters:
            - `x` (pd.Series[str]) sample to predict.

        returns:
            - `y_pred` (np.array[int]) class labels for sample `x`.
        """
        return self.model.predict(x)

    def predict_proba(self, x: pd.Series) -> np.array:
        """
        Estimate classification probabilities on samples in `x`.

        parameters:
            - `x` (pd.Series[str]) sample to predict.

        returns:
            - `y_pred` (np.array of floats with n classes columns) probability
              labels for sample `x`.
        """
        return self.model.predict_proba(x)

    def save_model(self) -> None:
        """
        Save model weights as a pickle python file in `self.output_dir` using
        its identifier `self.model_name`.
        """
        pickle.dump(self.model, open(f"{self.output_dir}/model.pkl", "wb"))

    def load_model(self, model_dirpath: str) -> None:
        """
        Load model weights. It takes directory path `model_dirpath` and the
        refered directory has to contain a pickle file in it named `model.pkl`.

        parameters:
            - `model_dirpath` (str) Directory path where the model is saved.
        """
        with open(f"{model_dirpath}/model.pkl", 'rb') as model_pkl:
            self.model = pickle.load(model_pkl)