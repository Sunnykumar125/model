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
import os

from official.staging.microbenchmarks import constants
from official.staging.microbenchmarks import schedule_base
from official.utils.testing.perfzero_benchmark import PerfZeroBenchmark


TASK_DIR = os.path.join(os.path.split(os.path.realpath(__file__))[0], "tasks")


class MNISTRunner(schedule_base.Runner):
  pass

  def get_cmd(self, task, result_path):
    model_path = {"MLP": "mlp.py"}[task.name]
    template = (
        "python {task_dir}/{task_file} --num_cores {num_cores} --num_gpus {num_gpus} "
        "--batch_size {batch_size} --result_path {result_path}")

    return template.format(
        task_dir=TASK_DIR, task_file=model_path, num_cores=task.num_cores,
        num_gpus=task.num_gpus, batch_size=task.batch_size,
        result_path=result_path,
    )


class MicroBenchmark(PerfZeroBenchmark):
  def __init__(self, output_dir=None, default_flags=None, root_data_dir=None):
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

    for mode in [constants.NUMPY, constants.DATASET]:
      tasks.append(constants.TaskConfig(
          name="MLP", num_cores=4, num_gpus=0,
          batch_size=32, data_mode=mode))

      tasks.append(constants.TaskConfig(
          name="MLP", num_cores=4, num_gpus=1,
          batch_size=32, data_mode=mode))

    for i in MNISTRunner(num_gpus=8).run(tasks, repeats=3):
      print(str(i)[:300])
    print()
    print(self.output_dir)
    import multiprocessing
    print(multiprocessing.cpu_count())


if __name__ == "__main__":
  MicroBenchmark().run_mnist_mlp()
