"""
E-type 传输线模型 (公式详见 2.3 节)
特点: 无前导 zA*zC/(zA+zC)*L 项
"""

import numpy as np
from .base import TLMBaseModel
from ._safe_math import safe_sinh_cosh


class ETypeModel(TLMBaseModel):
    def compute_Z(self, f_arr, params):
        zA = self._compute_sub_z('zA', f_arr, params)
        zB = self._compute_sub_z('zB', f_arr, params)
        zC = self._compute_sub_z('zC', f_arr, params)
        zP = self._compute_sub_z('zP', f_arr, params)
        zQ = self._compute_sub_z('zQ', f_arr, params)
        R0 = params.get('R0', 0.0)
        L_val = params.get('L', 0.0)

        Z = np.zeros(len(f_arr), dtype=np.complex128)

        for i in range(len(f_arr)):
            z_sum = zA[i] + zC[i]
            beta = np.sqrt(z_sum / zB[i]) * L_val
            sh, ch = safe_sinh_cosh(beta)
            sqrt_term = np.sqrt(zB[i] * z_sum)

            if self.boundary == 'P-Q':
                num = sqrt_term * ch + (1/zQ[i]) * zB[i] * z_sum * sh
                den = (1 + zB[i]*z_sum/(zP[i]*zQ[i])) * sh + (1/zP[i] + 1/zQ[i]) * sqrt_term * ch
                Z[i] = num / den

            elif self.boundary == 'open-Q':
                num = sqrt_term * ch + (1/zQ[i]) * zB[i] * z_sum * sh
                den = sh + (1/zQ[i]) * sqrt_term * ch
                Z[i] = num / den

            elif self.boundary == 'open-open':
                Z[i] = sqrt_term * ch / sh

            else:
                raise ValueError(f"Unknown boundary: {self.boundary}")

            Z[i] += R0

        return Z
