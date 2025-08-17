import os
import tensorflow as tf
import dagshub
import mlflow
import mlflow.keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.parse import urlparse
from pathlib import Path
from sklearn.metrics import confusion_matrix
from kidney_classifier.entity.config_entity import EvaluationConfig
from kidney_classifier.utils.common import save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model = None
        self.test_ds = None

    def get_test_dataset(self):
        """
        Loads the test dataset.
        """
        
        full_dataset = tf.keras.utils.image_dataset_from_directory(
            directory=self.config.data_dir,
            labels='inferred',
            label_mode='categorical',
            image_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            shuffle=True,
            seed=123
        )
        self.class_names = full_dataset.class_names

        
        dataset_size = tf.data.experimental.cardinality(full_dataset).numpy()
        val_size = int(dataset_size * self.config.params_val_split_size)
        test_size = int(dataset_size * self.config.params_test_split_size)
        train_size = dataset_size - val_size - test_size
        
        
        test_ds_unbatched = full_dataset.skip(train_size + val_size)
        
        
        rescale = tf.keras.layers.Rescaling(1./255)
        self.test_ds = test_ds_unbatched.map(lambda x, y: (rescale(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        self.test_ds = self.test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)


    def load_model(self):
        self.model = tf.keras.models.load_model(self.config.path_of_model)

    def log_into_mlflow(self):
        
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            scores = self.model.evaluate(self.test_ds)
            metrics = {"loss": scores[0], "accuracy": scores[1]}
            scores_filepath = self.config.root_dir / "scores.json"
            save_json(path=scores_filepath, data=metrics)
            mlflow.log_metrics(metrics)
            
            
            y_true = np.concatenate([y for x, y in self.test_ds], axis=0)
            y_true_indices = np.argmax(y_true, axis=1)
            predictions = self.model.predict(self.test_ds)
            y_pred_indices = np.argmax(predictions, axis=1)
            
            cm = confusion_matrix(y_true_indices, y_pred_indices)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=self.class_names, yticklabels=self.class_names)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            mlflow.log_figure(plt.gcf(), "confusion_matrix.png")

            if tracking_url_type_store != "file":
                mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16KidneyClassifier")
            else:
                mlflow.keras.log_model(self.model, "model")

    def evaluate_and_log(self):
        repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
        repo_name = os.getenv("DAGSHUB_REPO_NAME")
        
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)

        self.load_model()
        self.get_test_dataset()
        self.log_into_mlflow()