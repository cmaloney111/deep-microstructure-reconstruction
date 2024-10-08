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

import logging
from typing import Tuple
import tensorflow as tf
import numpy as np

from src import descriptor_factory
from descriptors.Descriptor import make_image_padder
from descriptors.PhaseDescriptor import PhaseDescriptor


class LinealPath(PhaseDescriptor):
    is_differentiable = False

    @staticmethod
    def make_singlephase_descriptor( 
            desired_shape_2d=(64, 64), 
            limit_to=4, 
            l_threshold_value=0.75, 
            threshold_steepness=10, 
            tol=0.1,
            **kwargs) -> callable:
        """Make multigrid version of lineal path function descriptor.
        Limit is from center to outer, like in correlations, i.e. a limit of 4 implies a
        symmetric mask width of 7"""
        H, W = desired_shape_2d
        limit_linealpath_to = limit_to * 2 + 1
        H_conv = limit_linealpath_to
        W_conv = limit_linealpath_to
        z_lower_bound = tf.cast(1.0 / (1.0 + tf.math.exp(-((0.0 - l_threshold_value) * threshold_steepness))), dtype=tf.float64)
        z_upper_bound = tf.cast(1.0 / (1.0 + tf.math.exp(-((1.0 - l_threshold_value) * threshold_steepness))), dtype=tf.float64)
        a = tf.cast(1.0 / (z_upper_bound - z_lower_bound), dtype=tf.float64)
        b = tf.cast(- a * z_lower_bound, dtype=tf.float64)

        tile_img = make_image_padder(min(W_conv, W) - 1, min(H_conv, H) - 1)

        @tf.function
        def make_dense_filters() -> tf.Tensor:
            """Make filters for lineal path function. First diagonals and straight lines, then 
            uses Bresenham line algorithm in the first octant, then swaps x and y for
            the second octant and finally mirrors y for the remainder. Surrounded by loop
            over line length. Future work: try Xiaolin Wu line algorithm."""
            in_channels = 1
            out_channels = sum((i - 1) * 2 for i in range(3, limit_linealpath_to + 1, 2)) + 1
            filters = np.zeros((limit_linealpath_to, limit_linealpath_to, in_channels, out_channels), dtype=np.float32)
            filters[0, 0, 0, 0] = 1
            filter_index = 1
            for sublim in range(3, limit_linealpath_to + 1, 2):
                start_filter_index = filter_index
                center_index = sublim // 2
                i_max = sublim - 1
                for k in range(sublim):
                    filters[k, k, 0, filter_index] = 1
                    filters[k, i_max - k, 0, filter_index + 1] = 1
                filter_index += 2
                filters[center_index, :sublim, 0, filter_index] = 1
                filter_index += 1
                filters[:sublim, center_index, 0, filter_index] = 1
                filter_index += 1
                for i in range(1, center_index):
                    slope = i / center_index
                    current_y = center_index - i
                    current_x = 0
                    filters[current_x, current_y, 0, filter_index] = 1
                    filters[current_y, current_x, 0, filter_index + 1] = 1
                    filters[current_x, i_max - current_y, 0, filter_index + 2] = 1
                    filters[current_y, i_max - current_x, 0, filter_index + 3] = 1
                    for current_x in range(1, i_max):
                        current_y += slope
                        if round(current_y - tol) == round(current_y + tol):
                            filters[current_x, round(current_y), 0, filter_index] = 1
                            filters[round(current_y), current_x, 0, filter_index + 1] = 1
                            filters[current_x, i_max - round(current_y), 0, filter_index + 2] = 1
                            filters[round(current_y), i_max - current_x, 0, filter_index + 3] = 1
                        else:
                            filters[current_x, round(current_y - tol), 0, filter_index] = 0.5
                            filters[current_x, round(current_y + tol), 0, filter_index] = 0.5
                            filters[round(current_y - tol), current_x, 0, filter_index + 1] = 0.5
                            filters[round(current_y + tol), current_x, 0, filter_index + 1] = 0.5
                            filters[current_x, i_max - round(current_y - tol), 0, filter_index + 2] = 0.5
                            filters[current_x, i_max - round(current_y + tol), 0, filter_index + 2] = 0.5
                            filters[round(current_y - tol), i_max - current_x, 0, filter_index + 3] = 0.5
                            filters[round(current_y + tol), i_max - current_x, 0, filter_index + 3] = 0.5
                    filter_index += 4
                filters[:, :, :, start_filter_index:filter_index] /= sublim
            filters_tf = tf.cast(tf.constant(filters), tf.float64)
            return filters_tf


        filters = make_dense_filters()

        def model(mg_input):
            img_tiled = tile_img(tf.constant(mg_input, dtype=tf.float64))
            img_convolved = tf.nn.conv2d(img_tiled, filters=filters,
                    strides=[1, 1, 1, 1], padding='VALID')
            img_thresholded = tf.cast(img_convolved > 0.99, tf.float64)
            img_thresholded = tf.nn.sigmoid((img_convolved - l_threshold_value) * threshold_steepness) * a + b
            img_reduced_x = tf.math.reduce_mean(img_thresholded, axis=1, keepdims=True)
            img_reduced_xy = tf.math.reduce_mean(img_reduced_x, axis=2, keepdims=True)
            return img_reduced_xy.numpy()
        return model

    @staticmethod
    def define_comparison_mask(
            desired_descriptor_shape: Tuple[int] = None, 
            limit_to: int = None, 
            **kwargs):
        assert desired_descriptor_shape[-1] == np.product(desired_descriptor_shape)
        current_descriptor_n = sum((i - 1) * 2 for i in range(3, 2*(limit_to + 1), 2)) + 1
        desired_descriptor_n = desired_descriptor_shape[-1]

        if current_descriptor_n == desired_descriptor_n:
            return None, False

        larger_n = max(current_descriptor_n, desired_descriptor_n)
        smaller_n = min(current_descriptor_n, desired_descriptor_n)
        mask = np.zeros(tuple(list(desired_descriptor_shape[:-1]) + [larger_n]), dtype=np.bool8)
        mask[..., :smaller_n] = True
        return mask, current_descriptor_n > desired_descriptor_n


    @classmethod
    def visualize_subplot(
            cls,
            descriptor_value: np.ndarray,
            ax,
            descriptor_type: str = None,
            mg_level: int = None,
            n_phase: int = None):
        import matplotlib.pyplot as plt
        x = descriptor_value.flatten()
        width = np.round(0.5 + np.sqrt(0.5 * descriptor_value.size - 0.25), decimals=0).astype(int) * 2 - 1
        center = width // 2
        xticks = [0, center, 2 * (center)]
        yticks = [0, center, 2 * (center)]
        lp = np.zeros((width, width))
        lp[center, center] = x[0]
        x_entry = 1
        for sublim in range(3, width + 1, 2):
            delta = sublim // 2
            lp[center + delta, center + delta] = x[x_entry]
            lp[center - delta, center - delta] = x[x_entry]
            x_entry += 1
            lp[center + delta, center - delta] = x[x_entry]
            lp[center - delta, center + delta] = x[x_entry]
            x_entry += 1
            lp[center, center + delta] = x[x_entry]
            lp[center, center - delta] = x[x_entry]
            x_entry += 1
            lp[center + delta, center] = x[x_entry]
            lp[center - delta, center] = x[x_entry]
            x_entry += 1
            for i in range(1, delta):
                lp[center + delta, center + i] = x[x_entry]
                lp[center - delta, center - i] = x[x_entry]
                x_entry += 1
                lp[center + i, center + delta] = x[x_entry]
                lp[center - i, center - delta] = x[x_entry]
                x_entry += 1
                lp[center + delta, center - i] = x[x_entry]
                lp[center - delta, center + i] = x[x_entry]
                x_entry += 1
                lp[center + i, center - delta] = x[x_entry]
                lp[center - i, center + delta] = x[x_entry]
                x_entry += 1
        assert x_entry == x.size
        ax.imshow(lp, cmap='cividis')
        ax.set_title(f'L: l={mg_level}, p={n_phase}')
        ax.set_xlabel(r'$r_x$ in Px')
        ax.set_ylabel(r'$r_y$ in Px')
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels([-center, 0, center])
        ax.set_yticklabels(reversed([-center, 0, center]))
        


def register() -> None:
    descriptor_factory.register("LinealPath", LinealPath)
