import os
from pathlib import Path
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from kidney_classifier.entity.config_entity import PrepareBaseModelConfig
from kidney_classifier import logger
from keras.applications import VGG16, ResNet50

class PrepareBaseModel:

    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
    
    def get_base_model(self):
        if os.path.exists(self.config.base_model_path):
            self.model = tf.keras.models.load_model(self.config.base_model_path)
            if not hasattr(self.model, 'optimizer') or self.model.optimizer is None:
                self.model.compile(
                    optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
                    loss=tf.keras.losses.CategoricalCrossentropy(),
                    metrics=["accuracy"]
                )
            logger.info(f"Base Model loaded from: {self.config.base_model_path}")
        else:
            if self.config.params_model_name == "VGG16":    
                self.model = tf.keras.applications.VGG16(
                    input_shape=self.config.params_image_size,
                    weights=self.config.params_weights,
                    include_top=self.config.params_include_top
                )
            elif self.config.params_model_name == "ResNet50":
                self.model = tf.keras.applications.ResNet50(
                    input_shape=self.config.params_image_size,
                    weights=self.config.params_weights,
                    include_top=self.config.params_include_top
                )
            else:
                raise ValueError(f"Unsupported model name: {self.config.params_model_name}")
            
            self.model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
    )
            self.save_model(path=self.config.base_model_path, model=self.model)
            logger.info(f"Base Model '{self.config.params_model_name}' created and saved to: {self.config.base_model_path}")
    
    @staticmethod
    def prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            model.trainable = False
        elif (freeze_till is not None) and (freeze_till>0):
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        flatten_in = tf.keras.layers.Flatten()(model.output)
        x = tf.keras.layers.Dense(512, activation="relu")(flatten_in)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(x)

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model

    def update_base_model(self):
        if os.path.exists(self.config.updated_base_model_path):
            self.model = tf.keras.models.load_model(self.config.updated_base_model_path)
            logger.info(f"Updated base Model loaded from: {self.config.updated_base_model_path}")
        
        else:
            self.full_model = self.prepare_full_model(
                model=self.model,
                classes=self.config.params_classes,
                freeze_all=True,
                freeze_till=None,
                learning_rate=self.config.params_learning_rate
            )
            self.save_model(path=self.config.updated_base_model_path, model=self.full_model)
            logger.info(f"Updated base model created and saved to: {self.config.updated_base_model_path}")
    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)