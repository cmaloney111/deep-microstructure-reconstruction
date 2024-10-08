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

from typing import List

import numpy as np
import tensorflow as tf
import scipy.optimize as sopt

from optimizers.Optimizer import Optimizer
from src.Microstructure import Microstructure

class SPOptimizer(Optimizer):
    is_gradient_based = True
    is_vf_based = False
    is_sparse = False

    def __init__(self,
                 max_iter: int = 100,
                 desired_shape_extended: tuple = None,
                 callback: callable = None):
        """ABC init for scipy optimizer. Subclasses simply specify self.optimizer_method and self.bounds."""
        self.max_iter = max_iter
        self.desired_shape_extended = desired_shape_extended
        self.reconstruction_callback = callback
        self.current_loss = None

        assert self.reconstruction_callback is not None
        self.optimizer_method = None
        self.bounds = None

    def step(self, x: np.ndarray) -> List[np.ndarray]:
        """Perform a single step. Typecasting from np to tf and back needed to couple scipy optimizers with tf backprop."""
        self.ms.x.assign(
            x.reshape(self.desired_shape_extended).astype(np.float64))
        loss, grads = self.call_loss(self.ms)
        self.current_loss = loss
        self.reconstruction_callback(self.n_iter, loss, self.ms)
        self.n_iter += 1
        return [field.numpy().astype(np.float64).flatten() for field in [loss, grads[0]]]

    def optimize(self, ms: Microstructure, restart_from_niter: int = None) -> int:
        """Optimize."""
        self.n_iter = 0 if restart_from_niter is None else restart_from_niter
        sp_options = {
            'maxiter': self.max_iter - self.n_iter,
            'maxfun': self.max_iter - self.n_iter,
        }
        self.ms = ms
        self.opt_var = [self.ms.x]
        print("calculating initial solution")
        initial_solution = self.ms.x.numpy().astype(np.float64).flatten()
        print("minimizing")
        try:
            resdd = sopt.minimize(fun=self.step, x0=initial_solution, jac=True, tol=0,
                                method=self.optimizer_method, bounds=self.bounds, options=sp_options)
        except Exception as e:
            print(f"Optimization failed with error: {e}")
        print("minimization finished")
        return self.n_iter