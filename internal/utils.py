# Copyright 2022 Google LLC
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

"""Functions for reflection directions and directional encodings."""

import math
import jax.numpy as jnp
import numpy as np
import enum
import os
from typing import Any, Dict, Optional, Union
import flax
import jax
from PIL import ExifTags
from PIL import Image

_Array = Union[np.ndarray, jnp.ndarray]


def refractive(viewdirs, normals):
    n_dot_w = jnp.sum(
        normals * viewdirs, axis=-1, keepdims=True)
    index = 1 / 1.5
    return (index * n_dot_w + jnp.sqrt(1 - index ** 2 * (1 - n_dot_w ** 2))) * normals - index * viewdirs


def l2_normalize(x, eps=jnp.finfo(jnp.float32).eps):
    """Normalize x to unit length along last axis."""
    return x / jnp.sqrt(jnp.maximum(jnp.sum(x ** 2, axis=-1, keepdims=True), eps))


def namedtuple_map(fn, tup):
    """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
    return type(tup)(*map(fn, tup))
