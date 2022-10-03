# Copyright 2022 AI Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Trains model to recognise hand gestures.
"""

from time import perf_counter
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from hand import connect_db

# Constants
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 1000
NUM_GESTURES = 9
PATIENCE = 50
MODEL_SAVE_PATH = "hand_gesture_classifier.hdf5"
RAND_SEED = 314159


class Trainer:
    def __init__(self) -> None:
        self.conn = connect_db()

    def load_training_data(self) -> pd.DataFrame:
        df = pd.read_sql("select * from hands order by gesture_id, id", self.conn)
        df = df.drop(["id"], axis=1)
        self.num_data = len(df)
        return df

    # Model routines
    def create_model(self) -> None:
        self.model = tf.keras.models.Sequential(
            [
                # tf.keras.layers.Input((21 * 2,)),
                # tf.keras.layers.Dropout(0.2),
                # tf.keras.layers.Dense(20, activation="relu"),
                # tf.keras.layers.Dropout(0.4),
                # tf.keras.layers.Dense(10, activation="relu"),
                # tf.keras.layers.Dense(NUM_GESTURES, activation="softmax"),
                tf.keras.layers.Input((21 * 2,)),
                tf.keras.layers.Dropout(0.0),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dropout(0.0),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dropout(0.0),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(NUM_GESTURES, activation="softmax"),
            ]
        )
        print(self.model.summary())
        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH, verbose=1, save_weights_only=False, save_best_only=True
        )
        self.es_callback = tf.keras.callbacks.EarlyStopping(
            patience=PATIENCE, verbose=1
        )
        the_optimizer = tf.keras.optimizers.Adam(lr=LEARNING_RATE)
        self.model.compile(
            optimizer=the_optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    def train(self) -> None:
        df = self.load_training_data()
        # print(df)
        y_data = df["gesture_id"]
        x_data = df.drop(["gesture_id"], axis=1)
        # print(y_data)
        # print(x_data)
        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, train_size=0.80, random_state=RAND_SEED
        )
        self.create_model()
        self.model.fit(
            x_train,
            y_train,
            epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=[self.cp_callback, self.es_callback],
            verbose=1,
        )
        self.x_test = x_test
        self.y_test = y_test

    def evaluate(self) -> None:
        eval_model = tf.keras.models.load_model(MODEL_SAVE_PATH)
        val_loss, val_acc = eval_model.evaluate(
            self.x_test, self.y_test, batch_size=BATCH_SIZE
        )
        y_predicts = eval_model.predict(self.x_test)
        y_pred = np.argmax(y_predicts, axis=1)
        print(classification_report(self.y_test, y_pred))


if __name__ == "__main__":
    st = perf_counter()
    trainer = Trainer()
    trainer.train()
    trainer.evaluate()
    et = perf_counter()
    print(f"Training time: {et - st:.0f} secs")
