"""
   Copyright 10/2020 - 04/2021 Paul Seibert for Diploma Thesis at TU Dresden
   Copyright 05/2021 - 12/2021 TU Dresden (Paul Seibert as Scientific Assistant)
   Copyright 2022 TU Dresden (Paul Seibert as Scientific Employee)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import logging

import numpy as np
import tensorflow as tf

from descriptors.Descriptor import Descriptor

class PhaseDescriptor(Descriptor):

    @classmethod
    def make_descriptor(
            cls, 
            desired_shape_2d=None, 
            desired_shape_extended=None, 
            use_multigrid_descriptor=True, 
            use_multiphase=True, 
            limit_to = 8,
            **kwargs) -> callable:
        """By default wraps self.make_single_phase_descriptor."""
        print("Called make descriptor")
        if use_multigrid_descriptor:
            print("no")
            singlephase_descriptor =  cls.make_multigrid_descriptor(
                limit_to=limit_to,
                desired_shape_2d=desired_shape_2d,
                desired_shape_extended=desired_shape_extended,
                **kwargs)
        else:
            print("here")
            singlephase_descriptor = cls.make_singlegrid_descriptor(
                limit_to=limit_to, 
                desired_shape_2d=desired_shape_2d,
                desired_shape_extended=desired_shape_extended,
                **kwargs) 
        print("Made it here")
        ms_shape = desired_shape_extended
        n_phases = ms_shape[-1]
        n_pixels = np.prod(ms_shape[:-1])

        @tf.function
        def singlephase_wrapper(x: tf.Tensor) -> tf.Tensor:
            print("yooooo")
            print(x.shape)
            print(singlephase_descriptor)
            phase_descriptor = tf.expand_dims(singlephase_descriptor(x), axis=0)
            print("here")
            print(phase_descriptor)
            return phase_descriptor

        @tf.function
        def multiphase_wrapper(x: tf.Tensor) -> tf.Tensor:
            phase_descriptors = []
            for phase in range(n_phases):
                x_phase = x[:, :, :, phase]
                phase_descriptor = tf.expand_dims(singlephase_descriptor(tf.expand_dims(x_phase, axis=-1)), axis=0)
                phase_descriptors.append(phase_descriptor)
            return tf.concat(phase_descriptors, axis=0)
        return multiphase_wrapper if use_multiphase else singlephase_wrapper
