from ISHate.models.transformer import TransformerModel
from ISHate.models.svm import USE_SVM
from ISHate.preprocessing.manipulation import flatten_dict

from tempfile import TemporaryDirectory
from omegaconf import DictConfig, OmegaConf

from pathlib import Path
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, accuracy_score,
                             precision_score, recall_score, f1_score)

import hydra
import logging
import pandas as pd
import shlex
import sys
import mlflow

logger = logging.getLogger(__name__)
logger.basicConfig(level=logger.INFO,
                            format="%(asctime)s [%(levelname)s] %(message)s")

def run_experiment(cfg: DictConfig, run: mlflow.ActiveRun):
    """
    Script that finetunes/train a model.
    """
    with TemporaryDirectory() as tmpfile:
        output_dir = Path(tmpfile)

        # Instance the model class. NOTE: MODELS is a dictionary where the values
        # are the model classes and the keys their corresponding class names.
        # cfg.model_name matches a key of MODELS.
        model = cfg.train.model.module(output_dir=output_dir,
                                       **cfg.train.model.params)

        logger.info("Command-line Arguments:")
        logger.info(
            f"Raw command-line arguments: {' '.join(map(shlex.quote, sys.argv))}")

        # Load train and dev datasets from `cfg.input.train_file` and `cfg.input.dev_file`.
        train = pd.read_parquet(cfg.input.train_file)
        dev = pd.read_parquet(cfg.input.dev_file)
        test = pd.read_parquet(cfg.input.test_file)

        # Separate in messages and labels.
        x_train, y_train = train["text"], train["label"].astype(int)
        x_dev, y_dev = dev["text"], dev["label"].astype(int)
        x_test, y_test = test["text"], test["label"].astype(int)

        if cfg.input.augmentation_file is not None:
            aug = pd.read_parquet(cfg.input.augmentation_file)
            x_train_aug, y_train_aug = aug["text"], aug["label"].astype(int)

            # Concat original and augmented data.
            x_train = pd.concat([x_train, x_train_aug])
            y_train = pd.concat([y_train, y_train_aug])

        # Use only a proportion of the train set to train the model
        if cfg.input.train_size is not None:
            x_train, _, y_train, _ = train_test_split(
                x_train, y_train, train_size=cfg.input.train_size)

        # Shuffle train and dev sets.
        x_train, y_train = shuffle(x_train, y_train, random_state=0)
        x_dev, y_dev = shuffle(x_dev, y_dev, random_state=0)

        logger.info("training model...")
        # Fit using train and dev sets.
        model.fit(x_train=x_train, y_train=y_train, x_dev=x_dev, y_dev=y_dev)

        logger.info("saving model...")
        # Save resultant model in `save_dir`.
        model.save_model()

        logger.info("training finished succesfully.")
        # Log model to mlflow.
        mlflow.log_artifact(output_dir)

        # Make predictions.
        y_pred_dev = model.predict(x_dev)
        y_pred_test = model.predict(x_test)

        # Calculate and log metrics.
        report = (
            f"**Classification results dev set**\n```\n{classification_report(y_pred=y_pred_dev, y_true=y_dev)}```\n"
            +
            f"**Classification results test set**\n```\n{classification_report(y_pred=y_pred_test, y_true=y_test)}```\n"
        )
        mlflow.set_tag("mlflow.note.content", report)

        mlflow.log_metric("accuracy_dev", accuracy_score(y_pred_dev, y_dev))
        mlflow.log_metric("precision_dev", precision_score(y_pred_dev, y_dev, average="macro"))
        mlflow.log_metric("recall_dev", recall_score(y_pred_dev, y_dev, average="macro"))
        mlflow.log_metric("f1_score_dev", f1_score(y_pred_dev, y_dev, average="macro"))

        mlflow.log_metric("accuracy_test", accuracy_score(y_pred_test, y_test))
        mlflow.log_metric("precision_test", precision_score(y_pred_test, y_test, average="macro"))
        mlflow.log_metric("recall_test", recall_score(y_pred_test, y_test, average="macro"))
        mlflow.log_metric("f1_score_test", f1_score(y_pred_test, y_test, average="macro"))


@hydra.main(config_path='conf', config_name='config', version_base=None)
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver('eval', lambda x: eval(x))

    mlflow.set_tracking_uri(cfg.input.uri_path)
    assert cfg.input.uri_path == mlflow.get_tracking_uri()

    logger.info(f"Current tracking uri: {cfg.input.uri_path}")

    mlflow.set_experiment(cfg.input.experiment_name)
    mlflow.set_experiment_tag('mlflow.note.content',
                              cfg.input.experiment_description)

    with mlflow.start_run(run_name=cfg.input.run_name) as run:
        logger.info("Logging configuration as artifact")
        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'config.yaml'
            with open(config_path, "wt") as fh:
                print(OmegaConf.to_yaml(cfg, resolve=False), file=fh)
            mlflow.log_artifact(config_path)

        logger.info("Logging configuration parameters")
        # Log params expects a flatten dictionary, since the configuration has nested
        # configurations (e.g. train.model), we need to use flatten_dict in order to
        # transform it into something that can be easilty logged by MLFlow.
        mlflow.log_params(
            flatten_dict(OmegaConf.to_container(cfg, resolve=False)))
        run_experiment(cfg, run)


if __name__ == '__main__':
    main()
