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

import contextlib
import itertools as it
import json
import timeit

from absl import flags
import tensorflow as tf
from tensorflow import keras
import numpy as np

from official.staging.microbenchmarks import constants


def define_flags():
  flags.DEFINE_integer(
      name='num_cores', default=-1,
      help='Number of cores allotted to this run.')
  flags.DEFINE_integer(
    name='num_gpus', default=-1,
    help='Number of GPUs allotted to this run.')
  flags.DEFINE_integer(
      name='batch_size', default=32,
      help='Minibatch size for training.')
  flags.DEFINE_enum(
      "data_mode", constants.NUMPY, [constants.NUMPY, constants.DATASET, constants.FROM_TENSOR_SLICES],
      "What kind of data to test. (NumPy array, Dataset, etc.)")
  flags.DEFINE_string(
      "run_mode_kwargs", default="{}",
      help="Compile with new single exection path. The caller handles "
           "availability and keyword rename.")
  flags.DEFINE_string(
      name='result_path', default=None,
      help='Path where results should be written.')


# TODO(robieta): Merge with existing timing callback.
class TimerCallback(keras.callbacks.Callback):

  def __init__(self):
    self.create_time = timeit.default_timer()
    self.epoch_start_time = None
    self.batch_start_time = None

    self.time_to_first_step = None
    self.first_step_time = None
    self.compile_time = None
    self.batch_times = []
    self.epoch_times = []

  @contextlib.contextmanager
  def time_compile(self):
    start_time = timeit.default_timer()
    try:
      yield
    finally:
      self.compile_time = timeit.default_timer() - start_time

  def on_epoch_begin(self, e, logs):
    if self.time_to_first_step is None:
      self.time_to_first_step = timeit.default_timer() - self.create_time

    self.epoch_start_time = timeit.default_timer()

  def on_batch_begin(self, e, logs):
    self.batch_start_time = timeit.default_timer()

  def on_batch_end(self, e, logs):
    if self.first_step_time is None:
      self.first_step_time = timeit.default_timer() - self.epoch_start_time
      self.epoch_start_time = timeit.default_timer()
    else:
      self.batch_times.append(timeit.default_timer() - self.batch_start_time)

  def on_epoch_end(self, e, logs):
    self.epoch_times.append(timeit.default_timer() - self.epoch_start_time)

  def summarize(self):
    return {
        "model_creation_time": self.time_to_first_step,
        "compile_time": self.compile_time,
        "startup_time": self.first_step_time,
        "batch_times": self.batch_times,
        "epoch_times": self.epoch_times,
        "end_to_end_time": timeit.default_timer() - self.create_time,
    }


def make_random_data(x_shapes, y_shapes, x_dtypes=None, y_dtypes=None,
                     x_maxvals=None, y_maxvals=None,
                     batch_size=32, num_examples=512,
                     data_mode=constants.NUMPY):

  x_dtypes = x_dtypes or [tf.float32 for _ in x_shapes]
  y_dtypes = y_dtypes or [tf.float32 for _ in y_shapes]

  x_maxvals = x_maxvals or [1 for _ in x_shapes]
  y_maxvals = y_maxvals or [1 for _ in y_shapes]

  AUTOTUNE = tf.data.experimental.AUTOTUNE
  if data_mode in (constants.NUMPY, constants.FROM_TENSOR_SLICES):
    flat_dtypes = [i.as_numpy_dtype() for i in x_dtypes + y_dtypes]
    data = tuple(
        np.random.uniform(
            high=maxval, size=(num_examples,) + shape).astype(dtype)
        for shape, dtype, maxval in
        zip(x_shapes + y_shapes, flat_dtypes, x_maxvals + y_maxvals)
    )
    if data_mode == constants.NUMPY:
      return {
          "x": data[:len(x_shapes)],
          "y": data[len(x_shapes):],
          "batch_size": batch_size
      }
    else:
      # Make NumPy data then read into Dataset.
      x = {"x": tf.data.Dataset.from_tensor_slices((
          data[:len(x_shapes)],
          data[len(x_shapes):],
      )).batch(batch_size).prefetch(AUTOTUNE)}
      return x

  elif data_mode == constants.DATASET:
    def map_fn(_):
      x = tuple(tf.random.uniform(shape=shape, dtype=dtype, maxval=maxval)
                for shape, dtype, maxval in zip(x_shapes, x_dtypes, x_maxvals))
      y = tuple(tf.random.uniform(shape=shape, dtype=dtype, maxval=maxval)
                for shape, dtype, maxval in zip(y_shapes, y_dtypes, y_maxvals))
      return x, y

    dataset = tf.data.Dataset.range(num_examples)
    dataset = dataset.map(map_fn, num_parallel_calls=AUTOTUNE)
    return {"x": dataset.batch(batch_size).prefetch(AUTOTUNE)}

  raise NotImplementedError("TODO(robieta)")


def maybe_check_gpu_present():
  if flags.FLAGS.num_gpus == 0:
    return

  with tf.device("GPU:0"):
    x = tf.ones(())

  if getattr(x, "graph", None):
    # V1 Tensor needs to be explicitly run.
    tf.compat.v1.Session(graph=x.graph).run(x)


def run_model(model_fn, input_fn):
  # Hey, this looks suspiciously like an Estimator!

  # Don't time setup.
  # Don't clear session since this is expected to start in a new process.
  data = input_fn()

  timer = TimerCallback()
  maybe_check_gpu_present()
  model = model_fn()

  model.compile('rmsprop', 'binary_crossentropy',
                **json.loads(flags.FLAGS.run_mode_kwargs))

  model.fit(**data, epochs=4, callbacks=[timer], verbose=2)

  results = timer.summarize()
  if flags.FLAGS.result_path is None:
    print(results)
  else:
    with open(flags.FLAGS.result_path, "wt") as f:
      json.dump(results, f)
