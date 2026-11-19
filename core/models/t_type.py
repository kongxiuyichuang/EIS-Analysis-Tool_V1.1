"""
T-type 传输线模型 (公式详见 2.2 节)
"""

import numpy as np
from .base import TLMBaseModel
from ._safe_math import safe_sinh, safe_cosh, safe_sinh_cosh, safe_coth


class TTypeModel(TLMBaseModel):
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
            if self.zC_zero:
                alpha = np.sqrt(zA[i] / zB[i]) * L_val
                Z[i] = self._zC_zero_impedance(alpha, zA[i], zB[i], zP[i], zQ[i])
            else:
                beta = np.sqrt((zA[i] + zC[i]) / zB[i]) * L_val
                Z[i] = self._general_impedance(beta, zA[i], zB[i], zC[i], zP[i], zQ[i], L_val)
            Z[i] += R0

        return Z

    def _general_impedance(self, beta, zA, zB, zC, zP, zQ, L_val=0.0):
        sh, ch = safe_sinh_cosh(beta)
        z_sum = zA + zC
        leading = (zA * zC / z_sum) * L_val
        coeff = (zA**2 * np.sqrt(zB)) / (z_sum ** 1.5)
        sqrt_term = np.sqrt(zB) * z_sum
        cosh_m1 = ch - 1

        if self.boundary == 'P-Q':
            num = 2*cosh_m1 + (1/zP + 1/zQ) * sqrt_term * sh
            den = (1 + zB*z_sum/(zP*zQ)) * sh + (1/zP + 1/zQ) * sqrt_term * ch
            Z = leading + coeff * num / den

        elif self.boundary == 'P-P symmetric':
            num = 2 * (cosh_m1 + (1/zP) * sqrt_term * sh)
            den = (1 + zB*z_sum/(zP**2)) * sh + (2/zP) * sqrt_term * ch
            Z = leading + coeff * num / den

        elif self.boundary == 'open-Q':
            num = 2*cosh_m1 + (1/zQ) * sqrt_term * sh
            den = sh + (1/zQ) * sqrt_term * ch
            Z = leading + coeff * num / den

        elif self.boundary == 'open-open':
            Z = leading + 2 * coeff * cosh_m1 / sh

        else:
            raise ValueError(f"Unknown boundary: {self.boundary}")

        return Z

    def _zC_zero_impedance(self, alpha, zA, zB, zP, zQ):
        sh, ch = safe_sinh_cosh(alpha)
        sqrt_zA_zB = np.sqrt(zA * zB)
        cosh_m1 = ch - 1

        if self.boundary == 'P-Q':
            num = 2*cosh_m1 + (1/zP + 1/zQ) * sqrt_zA_zB * sh
            den = (1 + zA*zB/(zP*zQ)) * sh + (1/zP + 1/zQ) * sqrt_zA_zB * ch
            Z = sqrt_zA_zB * num / den

        elif self.boundary == 'P-P symmetric':
            num = 2 * (cosh_m1 + (1/zP) * sqrt_zA_zB * sh)
            den = (1 + zA*zB/(zP**2)) * sh + (2/zP) * sqrt_zA_zB * ch
            Z = sqrt_zA_zB * num / den

        elif self.boundary == 'open-Q':
            num = 2*cosh_m1 + (1/zQ) * sqrt_zA_zB * sh
            den = sh + (1/zQ) * sqrt_zA_zB * ch
            Z = sqrt_zA_zB * num / den

        elif self.boundary == 'open-open':
            Z = 2 * sqrt_zA_zB * cosh_m1 / sh

        else:
            raise ValueError(f"Unknown boundary: {self.boundary}")

        return Z
