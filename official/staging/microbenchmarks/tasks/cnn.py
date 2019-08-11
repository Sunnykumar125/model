# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app as absl_app
from absl import flags

from tensorflow import keras

from official.staging.microbenchmarks.tasks import base


def input_fn():
  return base.make_random_data(
      x_shapes=((28, 28, 1),),
      y_shapes=((10,),),
      batch_size=flags.FLAGS.batch_size,
      num_examples=50000,
      data_mode=flags.FLAGS.data_mode,
  )


def model_fn():
  model = keras.Sequential()
  model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
                                activation='relu',
                                input_shape=(28, 28, 1)))
  model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
  model.add(keras.layers.Dropout(0.25))
  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(128, activation='relu'))
  model.add(keras.layers.Dropout(0.5))
  model.add(keras.layers.Dense(10, activation='softmax'))
  return model


def main(_):
  base.run_model(model_fn, input_fn)


if __name__ == "__main__":
  base.define_flags()
  absl_app.run(main)
