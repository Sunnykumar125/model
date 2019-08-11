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

import itertools as it

from official.staging.microbenchmarks import constants
from official.staging.microbenchmarks import schedule_base
from official.utils.testing.perfzero_benchmark import PerfZeroBenchmark


class MNISTRunner(schedule_base.Runner):
  pass

  # def params_to_key(self, params, gpu_used):
  #   return "{}{}\n    batchsize:{:>5}".format(params["model_type"].upper(), "(GPU)" if gpu_used else "", params["batch_size"])
  #
  # def get_cmd(self, num_cores, num_gpus, params, result_path):
  #   model_type = params["model_type"]
  #   model_path = {"MLP": "mlp.py"}[model_type]
  #
  #   template = (
  #       "python tasks/{task_py} --num_cores {num_cores} --num_gpus {num_gpus} "
  #       "--batch_size {batch_size} --result_path {result_path}")
  #   raise NotImplementedError
  #   # return template.format(
  #   #     task_py=model_path,
  #   # )
  #   #
  #   # return (
  #   #     "python tasks/{} --num_cores {} --num_gpus {} --result_path {} "
  #   #     "--batch_size {}".format(
  #   #         model_path, num_cores, num_gpus, result_path, params["batch_size"]))


class MicroBenchmark(PerfZeroBenchmark):
  def __init__(self, output_dir=None, default_flags=None):
    super(MicroBenchmark, self).__init__(
        output_dir=output_dir,
        default_flags=default_flags,
        flag_methods=[])

  def run_mnist_mlp(self):
    tasks = []

    """
    # CPU benchmark.
    for num_cores, num_gpus, data_mode, batch_size in it.product(
        [1, 2, 4, 8, 16],
        [0],
        [constants.NUMPY],
        [32, 64, 128, 256, 512]):
      tasks.append(constants.TaskConfig(
          task="MLP", num_cores=num_cores, num_gpus=num_gpus,
          batch_size=batch_size, data_mode=data_mode)
      )

    # GPU benchmark.
    for num_cores, num_gpus, data_mode, batch_size in it.product(
        [12],
        [1],
        [constants.NUMPY],
        [32, 64, 128, 256, 512]):
      tasks.append(constants.TaskConfig(
          task="MLP", num_cores=num_cores, num_gpus=num_gpus,
          batch_size=batch_size, data_mode=data_mode)
      )
    """

    tasks.append(constants.TaskConfig(
        task="MLP", num_cores=4, num_gpus=0,
        batch_size=32, data_mode=constants.NUMPY))

    tasks.append(constants.TaskConfig(
        task="MLP", num_cores=4, num_gpus=1,
        batch_size=32, data_mode=constants.NUMPY))

    print(MNISTRunner().run(tasks, repeats=3))

MicroBenchmark().run_mnist_mlp()
