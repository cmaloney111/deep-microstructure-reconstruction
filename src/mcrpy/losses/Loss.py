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
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from src import loss_factory


class Loss(ABC):
    @abstractmethod
    def define_norm():
        pass

    @classmethod
    def make_loss(cls,
                  descriptor_list: List[callable] = None,
                  desired_descriptor_list: List[np.ndarray] = None,
                  descriptor_weights: List[float] = None,
                  descriptor_comparisons: List[callable] = None,
                  dropout_rate: float = None, # if you change this you better know what you are doing
                  tvd: float = None,
                  oor_multiplier: float = 1000.0,
                  phase_sum_multiplier: float = 1.0,
                  mg_level: int = None,
                  mg_levels: int = None,
                  desired_shape_extended: Tuple[int] = None,
                  penalize_uncertainty: bool = False,
                  use_multiphase: bool = False,
                  descriptor_is_multiphase: List[bool] = None,
                  use_orientations: bool = False,
                  **kwargs) -> callable:

        if descriptor_list is None:
            raise ValueError('A descriptor_list must be passed that is not None.')
        if desired_descriptor_list is None:
            raise ValueError('A desired_descriptor_list must be passed that is not None.')
        if descriptor_weights is None:
            raise ValueError('Some descriptor_weights must be passed that is not None.')
        if descriptor_comparisons is None:
            raise ValueError('Some descriptor_comparisons must be passed that is not None.')
        if not (len(descriptor_list) == len(desired_descriptor_list) == len(descriptor_weights)):
            raise ValueError(
                (
                    'The lengths of the descriptor_list, desired_descriptor_list '
                    + f'and descriptor_weights are {len(descriptor_list)}, '
                    + f'{len(desired_descriptor_list)} and {len(descriptor_weights)} '
                    + 'respectively, but should be all the same.'
                )
            )

        print("registering MSE")
        if dropout_rate is not None:
            num_samples = tf.constant(sum(np.product(desired_descriptor.shape) 
                for desired_descriptor in desired_descriptor_list) * ((mg_level + 1) / desired_descriptor_list[0].shape[1]), dtype=tf.int32)

            @tf.function
            def draw_kept_components() -> tf.Tensor:
                component_mask = tf.cast(tf.random.categorical(tf.math.log(
                    [[dropout_rate, 1 - dropout_rate]]), num_samples, seed=1), tf.float64)
                if tf.norm(component_mask, ord=1) < 0.5:
                    component_mask = tf.cast(tf.random.categorical(tf.math.log(
                        [[dropout_rate, 1 - dropout_rate]]), num_samples, seed=1), tf.float64)
                return component_mask

        if tvd is not None and tvd > 0:
            @tf.function
            def tvd_term(x):
                return tf.image.total_variation(x)
        else:
            tvd_term = None


        n_descriptors = len(descriptor_list)
        n_phases = desired_descriptor_list[0].shape[0] if use_multiphase else 1
        phases = list(range(n_phases))
        phases_desired = phases
        if desired_descriptor_list[0].shape[0] > n_phases:
            logging.warning('desired_descriptor has multiphase, but multiphase is switched off for reconstruction. Using phase 1')
            phases_desired = [1]
        n_mg_levels = min(desired_descriptor_list[0].shape[1], mg_levels)
        n_pixels = np.prod(desired_shape_extended[:-1])
        iterate_normal_descriptor = list(zip(phases, phases_desired))
        iterate_multiphase_descriptor = list(zip([0], [0]))

        norm = cls.define_norm()

        @tf.function
        def compute_energy(x):
            flattened_difference_list = []
            for descriptor_function, desired_descriptor, descriptor_weight, descriptor_comparison, d_is_multiphase in zip(
                    descriptor_list, desired_descriptor_list, descriptor_weights, descriptor_comparisons, descriptor_is_multiphase):
                current_descriptor = descriptor_function(x)
                desired_descriptor = tf.cast(tf.constant(desired_descriptor), tf.float64)
                descriptor_differences = []
                phase_iterator = iterate_multiphase_descriptor if d_is_multiphase else iterate_normal_descriptor
                for phase, phase_desired in phase_iterator:
                    if use_multiphase and n_phases > 1:
                        if d_is_multiphase:
                            phase_weight = 1.0
                        else:
                            x_phase = x[:, :, :, phase]
                            phase_vf = tf.math.reduce_sum(x_phase) / n_pixels
                            phase_weight = (1 - phase_vf) * (1 - phase_vf) + 0.01
                    else:
                        phase_weight = 1.0
                    for n_mg_level in range(n_mg_levels - mg_level):
                        level_weight = 2**n_mg_level
                        descriptor_differences.append(tf.reshape( 
                            phase_weight * level_weight * descriptor_comparison(
                                current_descriptor[phase, n_mg_level], 
                                desired_descriptor[phase_desired, n_mg_level + mg_level]), [1, -1]))
                try:
                    weighted_descriptor_difference = tf.concat(descriptor_differences, axis=0) if len(descriptor_differences) > 1 else descriptor_differences[0]
                    weighted_descriptor_difference = tf.reshape(weighted_descriptor_difference, [-1])
                except IndexError as e:
                    raise ValueError("""Combination of multigrid settings and limit_to not 
                            possible for given descriptor file. Try changing 
                            use_multigrid_descriptor, use_multigrid_reconstruction or 
                            limit_to or change the characterization settings.""")
                flattened_difference_list.append(weighted_descriptor_difference * descriptor_weight)
            difference = tf.concat(flattened_difference_list, 0) if len(flattened_difference_list) > 1 else flattened_difference_list[0]
            if dropout_rate is None:
                energy = norm(difference)
                return energy
            else:
                component_mask = draw_kept_components()
                difference_mb = tf.math.multiply(
                        difference, 
                        tf.reshape(component_mask,tf.shape(difference)))
                energy = norm(difference_mb)
                return energy

        @tf.function
        def uncertainty(x):
            penalty_uncertainty = tf.math.reduce_mean(tf.nn.relu(0.25 - tf.math.square(x - 0.5)), axis=None)
            return penalty_uncertainty

        @tf.function
        def phase_sum(x):
            penalty_phase_sum = tf.math.reduce_mean(tf.math.square(tf.math.reduce_sum(x, axis=-1) - 1), axis=None)
            return penalty_phase_sum

        @tf.function
        def compute_oor_penalty(x):
            out_of_range = tf.math.square(tf.nn.relu(
                x - 1) + tf.nn.relu(0 - x))
            penalty_oor = tf.math.reduce_mean(out_of_range)
            return penalty_oor

        #@tf.function
        def compute_loss(x):
            energy = compute_energy(x)
            standard_loss = energy
            if not use_orientations:
                lambda_oor = energy * oor_multiplier
                penalty_oor = compute_oor_penalty(x)
                standard_loss = standard_loss + lambda_oor * penalty_oor
            if use_multiphase and not use_orientations:
                lambda_phase_sum = energy * phase_sum_multiplier
                penalty_phase_sum = phase_sum(x)
                standard_loss = standard_loss + lambda_phase_sum * penalty_phase_sum
            if tvd_term is not None:
                lambda_var = energy * tvd
                penalty_var = tvd_term(x)
                standard_loss = standard_loss + lambda_var * penalty_var
                raise NotImplementedError
            if penalize_uncertainty:
                lambda_uncertainty = energy * 1
                uncertainty_term = uncertainty(x)
                standard_loss = standard_loss + lambda_uncertainty * uncertainty_term
                raise NotImplementedError
            return standard_loss

        return compute_loss
