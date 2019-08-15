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

import collections

import six

NUMPY = "numpy"
DATASET = "dataset"
DATASET_WITH_PREFETCH="dataset_with_prefetch"
FROM_TENSOR_SLICES = "from_tensor_slices"
FROM_TENSOR_SLICES_WITH_PREFETCH = "from_tensor_slices_with_prefetch"

# TODO(robieta):
# GENERATOR = "generator"
# EAGER_TENSOR = "eager_tensor"


assert six.PY3, "This assumes Py3 dicts."
_TaskDefaults = dict(
    name="N/A",
    num_cores=1,
    num_gpus=0,
    data_mode=NUMPY,
    batch_size=32,
    experimental_run_tf_function=None,
    misc_params=None,
)

TaskConfig = collections.namedtuple(
    "TaskConfig",
    field_names=_TaskDefaults.keys(),
)

# `defaults` was only added in Python 3.7
TaskConfig.__new__.__defaults__ = tuple(_TaskDefaults.values())
