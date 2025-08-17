import time
import tensorflow as tf
from pathlib import Path
from kidney_classifier.entity.config_entity import TrainingConfig

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.train_ds = None
        self.valid_ds = None
        self.callbacks = None

    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def get_datasets(self):
        """
        Loads and splits the dataset into training and validation sets.
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

        
        dataset_size = tf.data.experimental.cardinality(full_dataset).numpy()
        val_size = int(dataset_size * self.config.params_val_split_size)
        test_size = int(dataset_size * self.config.params_test_split_size)
        train_size = dataset_size - val_size - test_size

        
        self.train_ds = full_dataset.take(train_size)
        self.valid_ds = full_dataset.skip(train_size).take(val_size)
        

        rescale = tf.keras.layers.Rescaling(1./255)
        self.train_ds = self.train_ds.map(lambda x, y: (rescale(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        self.valid_ds = self.valid_ds.map(lambda x, y: (rescale(x), y), num_parallel_calls=tf.data.AUTOTUNE)

        
        if self.config.params_is_augmentation:
            data_augmentation = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal_and_vertical"),
                tf.keras.layers.RandomRotation(0.2),
            ])
            self.train_ds = self.train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

        
        self.train_ds = self.train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        self.valid_ds = self.valid_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        
    def get_callbacks(self):
        """
        Updates the callbacks list.
        """
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_log_dir = str(self.config.tensorboard_root_log_dir / f"tb_logs_at_{timestamp}")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_log_dir)

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(self.config.checkpoint_model_filepath),
            save_best_only=True,
            monitor="val_loss"
        )
        
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor=self.config.params_early_stopping_monitor,
            patience=self.config.params_early_stopping_patience,
            restore_best_weights=True
        )
        
        self.callbacks = [tensorboard_callback, checkpoint_callback, early_stopping_callback]


    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self):
        self.get_base_model()
        self.get_datasets()
        self.get_callbacks()

        self.model.fit(
            self.train_ds, 
            epochs=self.config.params_epochs,
            validation_data=self.valid_ds,
            callbacks=self.callbacks
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )