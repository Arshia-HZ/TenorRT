#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import sys
import numpy as np
from PIL import Image


class DataLoaderBatcher:
    """
    Handles batches from a PyTorch DataLoader for calibration.
    """
    def __init__(self, dataloader, shape, dtype, max_batches=None):
        """
        :param dataloader: The PyTorch DataLoader object.
        :param shape: The tensor shape of the batch to prepare, either in NCHW or NHWC format.
        :param dtype: The (numpy) datatype to cast the batched data to.
        :param max_batches: Optional: Max number of batches to process for calibration.
        """
        self.dataloader = dataloader
        self.batch_size = shape[0]
        self.shape = shape
        self.dtype = dtype
        self.max_batches = max_batches
        self.batch_index = 0


    def get_batch(self):
        """
        Retrieves batches from the PyTorch DataLoader.
        :return: A generator yielding a numpy array with a batch of images and the respective labels.
        """
        for batch_idx, (batch_data, _) in enumerate(self.dataloader):
            if self.max_batches and batch_idx >= self.max_batches:
                break
            batch_data = batch_data.numpy().astype(self.dtype)  # Convert to numpy
            yield batch_data, None, None  # Ignore labels and scales, not needed for calibration
