"""
Universal-type 传输线模型（最一般形式）
公式详见 2.4 节，包含系数 A, B, C, D, E
"""

import numpy as np
from .base import TLMBaseModel
from ._safe_math import safe_sinh, safe_cosh


class UniversalTypeModel(TLMBaseModel):
    def compute_Z(self, f_arr, params):
        zA = self._compute_sub_z('zA', f_arr, params)
        zB = self._compute_sub_z('zB', f_arr, params)
        zC = self._compute_sub_z('zC', f_arr, params)
        zP = self._compute_sub_z('zP', f_arr, params)
        zQ = self._compute_sub_z('zQ', f_arr, params)
        ZV = self._compute_sub_z('ZV', f_arr, params)
        ZW = self._compute_sub_z('ZW', f_arr, params)
        ZX = self._compute_sub_z('ZX', f_arr, params)
        ZY = self._compute_sub_z('ZY', f_arr, params)
        R0 = params.get('R0', 0.0)
        L_val = params.get('L', 0.0)

        Z = np.zeros(len(f_arr), dtype=np.complex128)

        for i in range(len(f_arr)):
            beta = np.sqrt((zA[i] + zC[i]) / zB[i]) * L_val
            sh, ch = safe_sinh(beta), safe_cosh(beta)

            z_sum = zA[i] + zC[i]
            sqrt_term = np.sqrt(zB[i] * z_sum)

            p_zV = ZV[i] if np.isfinite(np.abs(ZV[i])) else 1e30
            p_zW = ZW[i] if np.isfinite(np.abs(ZW[i])) else 1e30
            p_zX = ZX[i] if np.isfinite(np.abs(ZX[i])) else 1e30
            p_zY = ZY[i] if np.isfinite(np.abs(ZY[i])) else 1e30

            A = ((p_zW*(p_zX+p_zY) + (p_zV+p_zW)*p_zY + p_zV*p_zW + p_zX*p_zY) * zA[i]**2
                 + 2*(p_zV*p_zW + p_zX*p_zY) * zA[i] * zC[i]
                 + (p_zV*(p_zX+p_zY) + (p_zV+p_zW)*p_zX + p_zV*p_zW + p_zX*p_zY) * zC[i]**2)

            B = sqrt_term * ((p_zW + p_zY) * zA[i]**2
                             + (p_zV*p_zW*(p_zX+p_zY) + (p_zV+p_zW)*p_zX*p_zY) * z_sum / zB[i]
                             + (p_zV + p_zX) * zC[i]**2)

            C = (-2 * p_zW * p_zY * zA[i]**2
                 + 2 * (p_zW*p_zX + p_zV*p_zY) * zA[i] * zC[i]
                 - 2 * p_zV * p_zX * zC[i]**2)

            D = (p_zV + p_zW + p_zX + p_zY) * sqrt_term

            E = zB[i] * z_sum + (p_zV + p_zW) * (p_zX + p_zY)

            leading = (zA[i] * zC[i] / z_sum) * L_val
            coeff = np.sqrt(zB[i]) / (z_sum ** 1.5)

            numerator = A * ch + B * sh + C
            denominator = D * ch + E * sh

            if np.abs(denominator) < 1e-30 * np.abs(numerator):
                denominator += 1e-30

            Z[i] = leading + coeff * numerator / denominator
            Z[i] += R0

        return Z
