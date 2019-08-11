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

import atexit
import json
import multiprocessing
import multiprocessing.dummy
import queue
import random
import shutil
import subprocess
import tempfile
import threading
import typing


from official.staging.microbenchmarks import constants


# Leave some headroom so the run manager doesn't contend with the benchmarks.
_NUM_CORES = multiprocessing.cpu_count() - 2
_TIMEOUT = 10 * 60


class CoreScheduler(object):
  """Simple class for pinning benchmarks to CPU cores.

  TODO(robieta): Possibly coalesce with re-pinning.
  """

  def __init__(self, num_cores=_NUM_CORES, num_gpus=1):
    self.num_cores = num_cores
    self.num_gpus = num_gpus

    self.core_available = [True for _ in range(self.num_cores)]
    self.gpus_available = {i for i in range(num_gpus)}
    self.lock = threading.Lock()
    self.active_allocations = {}
    self.active_gpu_allocations = {}

  def as_str(self):
    with self.lock:
      contents = [" " for _ in range(self.num_cores)]
      for start, num_cores in self.active_allocations.items():
        if num_cores == 1:
          contents[start] = "|"
        else:
          for i in range(num_cores):
            contents[start + i] = "-"
          contents[start] = "<"
          contents[start + num_cores - 1] = ">"
      out_str = "[" + "".join(contents) + "]"
      if self.num_gpus:
        gpu_str = "".join([" " if i in self.gpus_available else "X"
                           for i in range(self.num_gpus)])
        out_str += "....[" + gpu_str + "]"
      return out_str

  def allocate(self, requested_cores, requested_gpus=0):
    with self.lock:
      return self._allocate(requested_cores, requested_gpus)

  def _allocate(self, requested_cores, requested_gpus):
    if requested_cores > self.num_cores:
      raise ValueError("Cannot satisfy request of {} cores."
                       .format(requested_cores))

    if requested_gpus > self.num_gpus:
      raise ValueError("Not enough GPUs registered.")

    starts, counts, in_block = [], [], False
    for i, is_avail in enumerate(self.core_available):
      if is_avail and in_block:
        counts[-1] += 1
      elif is_avail:
        in_block = True
        starts.append(i)
        counts.append(1)
      else:
        in_block = False

    if not counts or max(counts) < requested_cores:
      return -1, ""

    if requested_gpus > len(self.gpus_available):
      return -1, ""

    # Prefer contiguous GPUs, but it's not as important as with CPUs.
    gpus = sorted(self.gpus_available)[:requested_gpus]
    self.gpus_available.difference_update(gpus)
    cuda_str = ",".join([str(i) for i in gpus])

    for start, count in zip(starts, counts):
      if count >= requested_cores:
        for i in range(requested_cores):
          self.core_available[start + i] = False
        self.active_allocations[start] = requested_cores
        self.active_gpu_allocations[start] = gpus
        return start, cuda_str

  def free(self, start, num_cores):
    with self.lock:
      return self._free(start, num_cores)

  def _free(self, start, num_cores):
    self.active_allocations.pop(start)
    for i in range(num_cores):
      self.core_available[start + i] = True
    self.gpus_available.update(self.active_gpu_allocations.pop(start))


class Runner(object):
  def __init__(self):
    self.scheduler = CoreScheduler()
    self.result_dir = tempfile.mkdtemp()
    atexit.register(shutil.rmtree, path=self.result_dir)

    self.free_queue = queue.Queue()
    self.result_queue = queue.Queue()

  def run(self, task_list, repeats=1):
    # type: (typing.List[constants.TaskConfig], int) -> None

    task_list = [i for i in task_list * repeats]
    random.shuffle(task_list)

    # The various map functions in pool seem to want to do some sort of grouping
    # of the iterable (even with chunksize=1), and wind up deadlocking. So
    # instead we manage the loop ourselves.
    with multiprocessing.dummy.Pool(24) as pool:
      for args in self.task_iter(task_list):
        pool.apply_async(self.map_fn, args=args)

      results = []
      for _ in task_list:
        results.append(self.result_queue.get(timeout=_TIMEOUT))

    results = self.collect_results(results)
    return results

  def collect_results(self, results):
    output = []
    for task, result_path in results:
      if result_path is None:
        print("skipping failed run.")
        continue

      with open(result_path, "rt") as f:
        output.append((task, json.load(f)))
    return output

  def task_iter(self, task_list):
    # type: (typing.List[constants.TaskConfig]) -> (constants.TaskConfig, int, str)
    for task in task_list:
      start = -1
      while start == -1:
        start, cuda_devices = self.scheduler.allocate(
            task.num_cores, task.num_gpus)

        if start == -1:
          # Tasks have a timeout of `_TIMEOUT`, so we never expect the main
          # thread to timout under normal conditions.
          self.scheduler.free(*self.free_queue.get(timeout=_TIMEOUT + 5))
        else:
          print(self.scheduler.as_str())
          yield task, start, cuda_devices


  def map_fn(self, task, start, cuda_devices):
    # type: (constants.TaskConfig, int, str) -> None

    success = False
    _, result_path = tempfile.mkstemp(prefix=self.result_dir + "/",
                                      suffix=".json")
    try:
      cmd = self.get_cmd(task, result_path)
      cmd = ("CUDA_VISIBLE_DEVICES='{}' taskset --cpu-list {}-{} {}"
             .format(cuda_devices, start, start + task.num_cores, cmd))
      try:
        # TODO(robieta): store output.
        result = subprocess.check_output(
            cmd, stderr=subprocess.STDOUT, shell=True,
            timeout=_TIMEOUT).decode("utf-8")  # type: str

        self.result_queue.put((task, result_path))
        success = True

      except subprocess.CalledProcessError as e:
        print("failed cmd: {}\n{}".format(cmd, e.output))

    finally:
      self.free_queue.put((start, task.num_cores))
      if not success:
        self.result_queue.put((task, None))

  def get_cmd(self, task, result_path):
    # type: (constants.TaskConfig, str) -> str
    return "echo '{{}}' > {}".format(result_path)
