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

import functools
import itertools as it
import os
import timeit
import typing
import uuid

import numpy as np
import tensorflow as tf

from official.staging.microbenchmarks import constants
from official.staging.microbenchmarks import schedule_base
from official.utils.testing.perfzero_benchmark import PerfZeroBenchmark


_NUM_GPUS = 4  # This expects to run on a 4xP100 machine.
TASK_DIR = os.path.join(os.path.split(os.path.realpath(__file__))[0], "tasks")
MODEL_PATHS = {
    "CNN": "cnn.py",
    "MLP": "mlp.py",
    "LOGREG": "logreg.py",
    "LSTM": "lstm.py"
}


class TaskRunner(schedule_base.Runner):
  def __init__(self, num_gpus=_NUM_GPUS):
    super(TaskRunner, self).__init__(num_gpus=num_gpus)

  def get_cmd(self, task, result_path):
    # PerfZero seems to need `python3` rather than `python`.
    template = (
        "python3 {task_dir}/{task_file} --num_cores {num_cores} "
        "--num_gpus {num_gpus} --batch_size {batch_size} "
        "--result_path {result_path} "
        "--run_mode_kwargs='{run_mode_kwargs}'")

    return template.format(
        task_dir=TASK_DIR, task_file=MODEL_PATHS[task.name],
        num_cores=task.num_cores, num_gpus=task.num_gpus,
        batch_size=task.batch_size, result_path=result_path,
        run_mode_kwargs=schedule_base.RUN_MODE_STR[
          task.experimental_run_tf_function],
    )


_NAME_STACK = []
def preserve_name(f):
  @functools.wraps(f)
  def wrapped(self):
    _NAME_STACK.append(f.__name__)
    out = f(self)
    _NAME_STACK.pop()
    return out
  return wrapped


class MicroBenchmark(PerfZeroBenchmark):
  def __init__(self, output_dir=None, default_flags=None, root_data_dir=None):
    self._experiment_id = str(uuid.uuid4())
    super(MicroBenchmark, self).__init__(
        output_dir=output_dir,
        default_flags=default_flags,
        flag_methods=[])

  def _get_name(self, overwrite_name=None):
    # This must be overridden to avoid an Estimator dependency issue.
    if _NAME_STACK:
      return "{}.{}".format(self.__class__.__name__, _NAME_STACK[-1])
    return super(MicroBenchmark, self)._get_name(overwrite_name)

  def _run_and_report_benchmark(self, tasks, runner, repeats):
    # type: (typing.List[constants.TaskConfig], schedule_base.Runner, int) -> None
    start_time = timeit.default_timer()
    results = runner.run(tasks, repeats=repeats)
    wall_time = timeit.default_timer() - start_time

    template = ("{experiment_id}, {version}, {git_version}, {name}, "
                "{batch_size}, {num_cores}, {num_gpus}, {data_mode}, "
                "{experimental_run_tf_function}, {misc_params}, "
                "{model_creation_time}, {data_creation_time}, {compile_time}, "
                "{startup_time}, {mean_epoch_time}, {end_to_end_time}, "
                "{framework_measured_end_to_end}, {mean_batch_time}\n")

    result_file = os.path.join(self.output_dir, "results.csv")
    with open(result_file, "wt") as f:
      f.write(template.replace("{", "").replace("}", ""))
      for task, result in results:
        line = template.format(
          experiment_id=self._experiment_id,
          version=tf.__version__,
          git_version=tf.__git_version__,
          name=task.name,
          batch_size=task.batch_size,
          num_cores=task.num_cores,
          num_gpus=task.num_gpus,
          data_mode=task.data_mode,
          experimental_run_tf_function=task.experimental_run_tf_function,
          misc_params=task.misc_params,
          model_creation_time=result['model_creation_time'],
          data_creation_time=result['data_creation_time'],
          compile_time=result['compile_time'],
          startup_time=result['startup_time'],
          mean_epoch_time=np.mean(result['epoch_times']),
          end_to_end_time=result['end_to_end_time'],
          framework_measured_end_to_end=result['framework_measured_end_to_end'],
          mean_batch_time=np.mean(result['batch_times']),
        )
        f.write(line)

    print("Results written to {}".format(result_file))

    self.report_benchmark(iters=-1, wall_time=wall_time)

  def _run_detailed_task(self, name):
    """Perform a detailed characterization of a single model."""
    tasks = []

    for data_mode, batch_size, experimental_run_tf_function in it.product(
        [constants.NUMPY, constants.DATASET, constants.DATASET_WITH_PREFETCH, constants.FROM_TENSOR_SLICES, constants.FROM_TENSOR_SLICES_WITH_PREFETCH],
        [32, 128, 512],  # TODO(robieta): run full [32, 64, 128, 256, 512]
        schedule_base.RUN_MODE_STR.keys()):

      # CPU benchmarks.
      for num_cores in [2, 4, 8]:
        tasks.append(constants.TaskConfig(
            name=name, num_cores=num_cores, num_gpus=0,
            batch_size=batch_size, data_mode=data_mode,
            experimental_run_tf_function=experimental_run_tf_function)
        )

      # GPU benchmark.
      tasks.append(constants.TaskConfig(
          name=name, num_cores=4, num_gpus=1,
          batch_size=batch_size, data_mode=data_mode,
          experimental_run_tf_function=experimental_run_tf_function)
      )

    self._run_and_report_benchmark(tasks, TaskRunner(), repeats=10)

  def _run_task(self, name):
    """Perform a detailed characterization of a single model."""
    tasks = []

    for data_mode, batch_size, experimental_run_tf_function in it.product(
        [constants.NUMPY, constants.DATASET, constants.FROM_TENSOR_SLICES],
        [32, 128, 512],  # TODO(robieta): run full [32, 64, 128, 256, 512]
        schedule_base.RUN_MODE_STR.keys()):

      # CPU benchmarks.
      for num_cores in [2, 4, 8]:
        tasks.append(constants.TaskConfig(
            name=name, num_cores=num_cores, num_gpus=0,
            batch_size=batch_size, data_mode=data_mode,
            experimental_run_tf_function=experimental_run_tf_function)
        )

      # GPU benchmark.
      tasks.append(constants.TaskConfig(
          name=name, num_cores=4, num_gpus=1,
          batch_size=batch_size, data_mode=data_mode,
          experimental_run_tf_function=experimental_run_tf_function)
      )

    self._run_and_report_benchmark(tasks, TaskRunner(), repeats=5)

  def _run_broad_task(self, num_cores, batch_size, repeats):
    """Perform a shallow characterization of all models."""
    tasks = []
    for name, num_gpus, experimental_run_tf_function in it.product(
        ["MLP", "CNN", "LOGREG", "LSTM"], [0, 1],
        schedule_base.RUN_MODE_STR.keys()):
      tasks.append(constants.TaskConfig(
          name=name,
          num_cores=num_cores,
          num_gpus=num_gpus,
          batch_size=batch_size,
          data_mode=constants.NUMPY,
          experimental_run_tf_function=experimental_run_tf_function)
      )
    self._run_and_report_benchmark(tasks, TaskRunner(), repeats=repeats)

  @preserve_name
  def run_mlp(self):
    self._run_task("MLP")

  @preserve_name
  def run_cnn(self):
    self._run_task("CNN")

  @preserve_name
  def run_logreg(self):
    self._run_task("LOGREG")

  @preserve_name
  def run_lstm(self):
    self._run_task("LSTM")

  @preserve_name
  def run_detailed_mlp(self):
    self._run_detailed_task("MLP")

  @preserve_name
  def run_detailed_logreg(self):
    self._run_detailed_task("LOGREG")

  @preserve_name
  def debug_test(self):
    self._run_broad_task(num_cores=12, batch_size=256, repeats=2)

  @preserve_name
  def run_baseline(self):
    self._run_broad_task(num_cores=2, batch_size=32, repeats=10)

  @preserve_name
  def run_trivial(self):
    tasks = [
        constants.TaskConfig(
            name="MLP",
            num_cores=4,
            num_gpus=0,
            batch_size=512,
            data_mode=data_mode,
            experimental_run_tf_function=False)
        for data_mode in [constants.NUMPY, constants.DATASET, constants.FROM_TENSOR_SLICES]
    ]
    self._run_and_report_benchmark(tasks, TaskRunner(1), repeats=2)

MicroBenchmark().run_trivial()
