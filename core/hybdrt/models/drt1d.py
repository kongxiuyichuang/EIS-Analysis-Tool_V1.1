# Workhorse class for distribution of relaxation times (DRT) fitting
# ----------------------------------------------------------------------
# Also enables distribution of phasances (DOP) fits
# Can be applied to EIS, chronopotentiometry, or joint-domain (hybrid) data


import inspect
import time
import warnings
import numpy as np
from numpy import ndarray
from matplotlib import pyplot as plt
from scipy import signal
from copy import deepcopy
import pickle
from typing import Optional, Tuple

from .. import utils, preprocessing as pp
from ..utils import stats
from ..matrices import mat1d, basis, phasance, ddt
from . import qphb, background
from .drtbase import DRTBase
from ..plotting import (
    plot_eis, plot_distribution, plot_chrono, shade_extrap_regions
)


class DRT(DRTBase):
    def __init__(self, **init_kw):
        super().__init__(**init_kw)

        self.background_gp = None

        self.v_baseline_scale = None

        self._qp_matrices = None

    @property
    def num_data(self):
        if self.fit_type.find('hybrid') >= 0:
            num_data = len(self.get_fit_times()) + 2 * len(self.get_fit_frequencies())
        elif self.fit_type.find('eis') >= 0:
            num_data = 2 * len(self.get_fit_frequencies())
        else:
            num_data = len(self.get_fit_times())
        return num_data

    @property
    def num_independent_data(self):
        if self.fit_type.find('hybrid') >= 0:
            num_data = len(self.get_fit_times()) + len(self.get_fit_frequencies())
        elif self.fit_type.find('eis') >= 0:
            num_data = len(self.get_fit_frequencies())
        else:
            num_data = len(self.get_fit_times())
        return num_data
    
    def _get_neg_allowed_indices(self, nonneg: bool, neg_allowed_tau_range: Optional[Tuple[float, float]] = None):
        if nonneg and neg_allowed_tau_range is not None:
            raise ValueError("If nonneg==True, neg_allowed_tau_range cannot be specified")
        
        if not nonneg and neg_allowed_tau_range is not None:
            # Find the parameter indices which correspond to the allowed tau range
            tau_min, tau_max = neg_allowed_tau_range
            index = np.where((self.basis_tau >= tau_min) & (self.basis_tau <= tau_max))[0]
            # Offset by special parameters
            return index + self.get_qp_mat_offset()
        
    def get_data_tau_limits(self):
        """Get tau limits of measured data.

        :return tuple: Tuple of tau_min, tau_max
        """
        return pp.get_tau_lim(self.get_fit_frequencies(True), self.get_fit_times(True), self.nonconsec_step_times)
        

    def _qphb_fit_core(self, times, i_signal, v_signal, frequencies, z, step_times=None, step_sizes=None, 
                       nonneg=True, neg_allowed_tau_range: Optional[Tuple[float, float]] = None, 
                       series_neg=False, scale_data=True, update_scale=False, solve_rp=False,
                       # chrono args
                       offset_steps=True, step_offset_size=None, discard_first_n: Optional[int] = None,
                       offset_baseline=True, v_baseline_deg: int = 0, v_baseline_sqrt: bool = False,
                       downsample=False, downsample_kw=None,
                       subtract_background=False, background_type='static', background_corr_power=None,
                       estimate_background_kw=None, smooth_inf_response=True,
                       # penalty settings
                       v_baseline_penalty=1e-6, ohmic_penalty=1e-6,
                       inductance_penalty=1e-6, capacitance_penalty=1e-6,
                       inductance_scale=1e-5, capacitance_scale=1e-3,
                       background_penalty=1,
                       penalty_type='integral',
                       remove_extremes=False, extreme_kw=None,
                       # error structure
                       init_weights_separately: bool = False,
                       chrono_error_structure='uniform', eis_error_structure=None,
                       remove_outliers=False, return_outlier_index=False, outlier_thresh=0.75,
                       chrono_vmm_epsilon=4, eis_vmm_epsilon=0.25, eis_reim_cor=0.25,
                       iw_l1_lambda_0=1e-4, iw_l2_lambda_0=1e-4,
                       # Hybrid settings
                       vz_offset=True, vz_offset_scale=1, vz_offset_eps=1,
                       eis_weight_factor=None, chrono_weight_factor=None,
                       hybrid_weight_factor_method=None,
                       # Prior hyperparameters
                       eff_hp=True, weight_factor=1,
                       # rp_scale=14, l2_lambda_0=None, l1_lambda_0=0.0, derivative_weights=None,
                       # iw_alpha=None, iw_beta=None,
                       # s_alpha=None, s_0=None,
                       # rho_alpha=None, rho_0=None,
                       # optimization control
                       xtol=1e-2, max_iter=50,
                       peak_locations=None,
                       progress_callback=None,
                       **kw):

        # Check inputs
        utils.validation.check_chrono_data(times, i_signal, v_signal)
        utils.validation.check_eis_data(frequencies, z)
        for err_struct in [chrono_error_structure, eis_error_structure]:
            utils.validation.check_error_structure(err_struct)
        utils.validation.check_penalty_type(penalty_type)

        if solve_rp and not scale_data and self.warn:
            warnings.warn('solve_rp is ignored if scale_data=False')
        # if (self.fit_dop or not nonneg) and not solve_rp and self.warn:
        #     warnings.warn('For best results, set solve_rp=True when performing DRT-DOP fits '
        #                   'or DRT fits without a non-negativity constraint')

        if series_neg and not nonneg:
            raise ValueError('Only one of series_neg and nonneg may be True')

        background_types = ['static', 'dynamic', 'scaled']
        if background_type not in background_types:
            raise ValueError(f"Invalid background_type {background_type}. Options: {background_types}")

        if remove_outliers and 'outlier_p' not in kw.keys():
            raise ValueError('If remove_outliers is True, the prior probability of outlier presence, outlier_p, '
                             'must be specified. A good starting value might be 0.01-0.05')
            
        self.v_baseline_deg = v_baseline_deg
        self.v_baseline_sqrt = v_baseline_sqrt

        # Copy data
        if times is not None:
            times = np.array(times)
            i_signal = np.array(i_signal)
            v_signal = np.array(v_signal)
            
            if discard_first_n is not None:
                dt_short = np.min(np.diff(times))
            
                _, (times, i_signal, v_signal) = pp.discard_first_n_chrono(times, i_signal, v_signal, discard_first_n, self.chrono_mode)
                
                # Determine t_sample after removing short samples
                t_sample = np.min(np.diff(times))
                
                if step_offset_size is None:
                    step_offset_size = - (dt_short + t_sample  * (discard_first_n - 1e-8))
                
        if frequencies is not None:
            frequencies = np.array(frequencies)
            z = np.array(z)

        # Find and remove extreme values (rough)
        if remove_extremes:
            if extreme_kw is None:
                extreme_kw = {'qr_size': 0.8, 'qr_thresh': 1.5}
            if times is not None:
                i_flag = pp.identify_extreme_values(i_signal, **extreme_kw)
                v_flag = pp.identify_extreme_values(v_signal, **extreme_kw)
                chrono_flag = i_flag | v_flag
                if np.sum(chrono_flag) > 0:
                    if self.warn:
                        warnings.warn(f'Identified extreme values in chrono data at the following '
                                      f'indices: {np.where(chrono_flag)[0].tolist()}. '
                                      f'These data points will be removed before fitting'
                                      )
                    times = times[~chrono_flag]
                    i_signal = i_signal[~chrono_flag]
                    v_signal = v_signal[~chrono_flag]
            if frequencies is not None:
                re_flag = pp.identify_extreme_values(z.real, **extreme_kw)
                im_flag = pp.identify_extreme_values(z.imag, **extreme_kw)
                eis_flag = re_flag | im_flag
                if np.sum(eis_flag) > 0:
                    if self.warn:
                        warnings.warn(f'Identified extreme values in EIS data at the following '
                                      f'indices: {np.where(eis_flag)[0].tolist()}. '
                                      f'These data points will be removed before fitting'
                                      )
                    frequencies = frequencies[~eis_flag]
                    z = z[~eis_flag]

        # Find and remove outliers (more precise)
        if remove_outliers:
            # Only need to supply kwargs that matter for initial weights estimation
            chrono_outlier_index, eis_outlier_index = self._qphb_fit_core(times, i_signal, v_signal, frequencies, z,
                                                                          step_times=step_times, step_sizes=step_sizes, 
                                                                          nonneg=nonneg, neg_allowed_tau_range=neg_allowed_tau_range,
                                                                          series_neg=series_neg, scale_data=scale_data,
                                                                          solve_rp=solve_rp, 
                                                                          offset_steps=offset_steps, step_offset_size=step_offset_size,
                                                                          discard_first_n=discard_first_n,
                                                                          offset_baseline=offset_baseline, v_baseline_deg=v_baseline_deg, v_baseline_sqrt=v_baseline_sqrt,
                                                                          downsample=downsample,
                                                                          downsample_kw=downsample_kw,
                                                                          subtract_background=False,
                                                                          smooth_inf_response=smooth_inf_response,
                                                                          v_baseline_penalty=v_baseline_penalty,
                                                                          ohmic_penalty=ohmic_penalty,
                                                                          inductance_penalty=inductance_penalty,
                                                                          capacitance_penalty=capacitance_penalty,
                                                                          background_penalty=background_penalty,
                                                                          inductance_scale=inductance_scale,
                                                                          capacitance_scale=capacitance_scale,
                                                                          penalty_type=penalty_type,
                                                                          chrono_error_structure=chrono_error_structure,
                                                                          eis_error_structure=eis_error_structure,
                                                                          remove_outliers=False,
                                                                          return_outlier_index=True,
                                                                          outlier_thresh=outlier_thresh,
                                                                          chrono_vmm_epsilon=chrono_vmm_epsilon,
                                                                          eis_vmm_epsilon=eis_vmm_epsilon,
                                                                          eis_reim_cor=eis_reim_cor, eff_hp=eff_hp,
                                                                          **kw)

            self.eis_outlier_index = eis_outlier_index
            self.chrono_outlier_index = chrono_outlier_index

            # Preserve step times determined before outlier removal
            step_times = self.step_times

            if times is not None:
                if np.sum(chrono_outlier_index) > 0:
                    if self.warn:
                        warnings.warn('Found outliers in chrono data at the following '
                                      f'indices: {np.where(chrono_outlier_index)[0].tolist()}. '
                                      f'These data points will be removed before fitting'
                                      )

                    # Store outliers for reference
                    self.chrono_outliers = (
                        times[chrono_outlier_index],
                        i_signal[chrono_outlier_index],
                        v_signal[chrono_outlier_index]
                    )

                    # Remove outliers from data to fit
                    times = times[~chrono_outlier_index]
                    i_signal = i_signal[~chrono_outlier_index]
                    v_signal = v_signal[~chrono_outlier_index]
                else:
                    self.chrono_outliers = None
            if frequencies is not None:
                if np.sum(eis_outlier_index) > 0:
                    if self.warn:
                        warnings.warn('Found outliers in EIS data at the following '
                                      f'indices: {np.where(eis_outlier_index)[0].tolist()}. '
                                      f'These data points will be removed before fitting'
                                      )

                    # Store outliers for reference
                    self.eis_outliers = (
                        frequencies[eis_outlier_index],
                        z[eis_outlier_index]
                    )

                    # Remove outliers from data to fit
                    frequencies = frequencies[~eis_outlier_index]
                    z = z[~eis_outlier_index]
                else:
                    self.eis_outliers = None

            # After identifying outliers, disable outlier error structure
            kw['outlier_p'] = None
        else:
            self.eis_outlier_index = None
            self.eis_outliers = None
            self.chrono_outlier_index = None
            self.chrono_outliers = None

        # Subtract chrono background
        if subtract_background and times is not None:
            if estimate_background_kw is None:
                estimate_background_kw = {}
            estimate_background_defaults = {
                'step_times': step_times,
                'nonneg': nonneg, 'series_neg': series_neg,
                'downsample': downsample, 'downsample_kw': downsample_kw
            }
            estimate_background_kw = dict(estimate_background_defaults, **estimate_background_kw)

            if estimate_background_kw.get('bkg_iter', 1) > 1:
                raise ValueError('When fitting with background subtraction, bkg_iter must be set to 1. If there'
                                 'are multiple background length scales, set kernel_size > 1')

            drt_bkg, bkg_gps, y_bkg = self.estimate_chrono_background(times, i_signal, v_signal, copy_self=True,
                                                                      **estimate_background_kw)
            if self.print_diagnostics:
                print('Finished initial background estimation')
            y_pred_bkg = drt_bkg.predict_response()

            if background_corr_power is None and background_type != 'static':
                bkg_std = np.std(y_bkg)
                v_std = np.std(y_pred_bkg)
                std_ratio = bkg_std / v_std
                # background_corr_power = 1.1 * np.exp(-5 * std_ratio)
                # background_corr_power = 1.25 * np.exp(-10 * std_ratio) + 0.25
                background_corr_power = np.log(0.02 / std_ratio + 1) + 0.25
                if self.print_diagnostics:
                    print('std ratio:', std_ratio)
                    print('background_corr_power:', background_corr_power)
                # background_corr_power = 1  # Default: penalty for dynamic background

            self.background_gp = bkg_gps[0]
            if background_type == 'static':
                if background_corr_power is not None:
                    rm_bkg = background.get_background_matrix(bkg_gps, drt_bkg.get_fit_times()[:, None],
                                                              y_drt=y_pred_bkg,
                                                              corr_power=background_corr_power)
                    y_resid = drt_bkg.raw_response_signal - y_pred_bkg
                    self.raw_response_background = rm_bkg @ y_resid
                else:
                    self.raw_response_background = y_bkg.copy()

                # Subtract the background from the signal to fit
                if self.chrono_mode == 'galv':
                    v_signal[drt_bkg.sample_index] -= self.raw_response_background
                else:
                    i_signal[drt_bkg.sample_index] = i_signal[drt_bkg.sample_index] - self.raw_response_background
        else:
            bkg_gps = None
            y_pred_bkg = None
            self.background_gp = None

        # Store series_neg
        self.series_neg = series_neg

        # Determine data type
        # Get num_eis. num_chrono must be determined after downsampling
        if times is None:
            data_type = 'eis'
            num_eis = len(frequencies)
        elif frequencies is None:
            data_type = 'chrono'
            num_eis = 0
        else:
            data_type = 'hybrid'
            num_eis = len(frequencies)

        if not self.fit_drt:
            if not (self.fit_dop or self.fit_ddt):
                raise ValueError('fit_drt=False requires fit_dop=True or fit_ddt=True')
            if times is not None:
                raise ValueError('fit_drt=False currently supports EIS data only (times must be None)')

        # Define special parameters included in quadratic programming parameter vector
        self.special_qp_params = {}

        if times is not None:
            self._add_special_qp_param('v_baseline', False, self.v_baseline_deg + 1 + int(self.v_baseline_sqrt))

        if vz_offset and data_type == 'hybrid':
            self._add_special_qp_param('vz_offset', False)

        if subtract_background and background_type == 'scaled':
            self._add_special_qp_param('background_scale', True)

        # DOP replaces R_inf and L
        if self.fit_ohmic:
            self._add_special_qp_param('R_inf', True)

        if self.fit_inductance:
            self._add_special_qp_param('inductance', True)

        if self.fit_capacitance:
            self._add_special_qp_param('C_inv', True)

        if self.fit_dop:
            if self.fixed_basis_nu is None:
                # self.basis_nu = np.linspace(-1, 1, 41)
                self.basis_nu = np.concatenate([np.linspace(-1, -0.4, 25), np.linspace(0.4, 1, 25)])
            else:
                self.basis_nu = self.fixed_basis_nu

            # Set nu_epsilon
            if self.nu_epsilon is None and self.nu_basis_type != 'delta':
                dnu = np.median(np.diff(np.sort(self.basis_nu)))
                self.nu_epsilon = 1 / dnu

            self._add_special_qp_param('x_dop', True, size=len(self.basis_nu))
        else:
            self.basis_nu = None

        if self.fit_ddt:
            if frequencies is None:
                raise ValueError('fit_ddt requires EIS data (frequencies and z)')

            if self.fit_ddt_p and (self.ddt_p_grid is not None or self.fit_ddt_2d):
                raise ValueError('fit_ddt_p (scalar exponent) is mutually exclusive with '
                                 'ddt_p_grid / fit_ddt_2d')

            if self.fixed_basis_tau_dd is None:
                # Default grid: 10 ppd, extended one decade beyond frequency-derived diffusion times.
                # In 2D mode a coarser grid is used to keep the problem size tractable.
                dd_ppd = 5 if (self.ddt_p_grid is not None or self.fit_ddt_2d) else 10
                self.basis_tau_dd = pp.get_basis_tau(frequencies, None, None, ppd=dd_ppd, extend_decades=1)
            else:
                self.basis_tau_dd = self.fixed_basis_tau_dd

            if self.tau_dd_epsilon is None and self.tau_dd_basis_type != 'delta':
                self.tau_dd_epsilon = 1 / np.mean(np.diff(np.log(self.basis_tau_dd)))

            # Anomalous-exponent grid: None -> 1D DDT (scalar p, default 0.5)
            if self.ddt_p_grid is not None:
                self.basis_p = np.asarray(self.ddt_p_grid, dtype=float)
            elif self.fit_ddt_2d:
                self.basis_p = ddt.DEFAULT_DDT_P_GRID.copy()
            else:
                self.basis_p = None

            if self.basis_p is not None:
                self.ddt_p_epsilon = 1 / np.mean(np.diff(self.basis_p))
            else:
                self.ddt_p_epsilon = None

            self._add_special_qp_param('x_ddt', True, size=self.num_ddt_params)
        else:
            self.basis_tau_dd = None
            self.basis_p = None

        # Get preprocessing hyperparameters. Won't know data factor until chrono data has been downsampled
        pp_hypers = qphb.get_default_hypers(eff_hp, self.fit_dop, self.nu_basis_type,
                                            fit_ddt=self.fit_ddt, ddt_kernel_type=self.ddt_kernel_type)

        # Validate provided kwargs
        for key in kw.keys():
            if key not in pp_hypers.keys():
                raise ValueError(f'Invalid keyword argument {key}')
        pp_hypers.update(kw)

        # Process data and calculate matrices for fit
        sample_data, matrices = self._prep_for_fit(times, i_signal, v_signal, frequencies, z,
                                                   step_times=step_times, step_sizes=step_sizes, downsample=downsample,
                                                   downsample_kw=downsample_kw, offset_steps=offset_steps, step_offset_size=step_offset_size, 
                                                   discard_first_n=discard_first_n,
                                                   smooth_inf_response=smooth_inf_response,
                                                   scale_data=scale_data, rp_scale=pp_hypers['rp_scale'],
                                                   penalty_type=penalty_type,
                                                   derivative_weights=pp_hypers['derivative_weights'])
        # Unpack processed data
        sample_times, sample_i, sample_v, response_baseline, z_scaled = sample_data
        # Unpack partial matrices
        rm_drt, induc_rv, inf_rv, cap_rv, rm_dop, rm_ddt, zm_drt, induc_zv, cap_zv, zm_dop, zm_ddt, base_penalty_matrices = matrices

        # Get num_chrono after downsampling
        if sample_times is not None:
            num_chrono = len(sample_times)
        else:
            num_chrono = 0
            
        def split_vector(x):
            return x[:num_chrono], x[num_chrono:]

        # Downsample static background
        if subtract_background and downsample and background_type == 'static' \
                and not estimate_background_kw.get('downsample', False):
            self.raw_response_background = self.raw_response_background[self.sample_index]

        # Get data factor after downsampling
        data_factor = qphb.get_data_factor_from_data(sample_times, self.step_times, frequencies)
        if self.print_diagnostics:
            print('data factor:', data_factor)

        # Get default hyperparameters and update with user-specified values
        qphb_hypers = qphb.get_default_hypers(eff_hp, self.fit_dop, self.nu_basis_type,
                                              fit_ddt=self.fit_ddt, ddt_kernel_type=self.ddt_kernel_type)
        qphb_hypers.update(kw)

        # Store fit kwargs for reference (after prep_for_fit creates self.fit_kwargs)
        self.fit_kwargs.update(qphb_hypers)
        self.fit_kwargs['nonneg'] = nonneg
        self.fit_kwargs['eff_hp'] = eff_hp
        self.fit_kwargs['penalty_type'] = penalty_type
        self.fit_kwargs['subtract_background'] = subtract_background
        self.fit_kwargs['background_type'] = background_type
        self.fit_kwargs['background_corr_power'] = background_corr_power
        self.fit_kwargs["neg_allowed_tau_range"] = neg_allowed_tau_range
        
        # Once prep_for_fit has created basis_tau, determine neg_allowed_indices
        neg_allowed_indices = self._get_neg_allowed_indices(nonneg, neg_allowed_tau_range)
        
        if self.print_diagnostics:
            print('lambda_0, iw_beta:', qphb_hypers['l2_lambda_0'], qphb_hypers['iw_beta'])

        # Format matrices for QP fit
        rm, zm, penalty_matrices = self._format_qp_matrices(rm_drt, inf_rv, induc_rv, cap_rv, rm_dop, rm_ddt,
                                                            zm_drt, induc_zv, cap_zv, zm_dop, zm_ddt,
                                                            base_penalty_matrices, v_baseline_penalty,
                                                            ohmic_penalty, inductance_penalty, capacitance_penalty,
                                                            vz_offset_scale, background_penalty, inductance_scale,
                                                            capacitance_scale, penalty_type,
                                                            qphb_hypers['derivative_weights'])

        if subtract_background and times is not None and background_type != 'static':
            rm_bkg = background.get_background_matrix(bkg_gps, sample_times[:, None], y_drt=y_pred_bkg,
                                                      corr_power=background_corr_power)
            if background_type == 'dynamic':
                rm_orig = rm.copy()
                rm = rm - rm_bkg @ rm
            else:
                rm_orig = None
        else:
            rm_bkg = None
            rm_orig = None

        # Construct hybrid response-impedance matrix
        if rm is None:
            rzm = zm.copy()
        elif zm is None:
            rzm = rm.copy()
        else:
            rzm = np.vstack((rm, zm))

        # Make a copy for vz_offset calculation
        if data_type == 'hybrid' and vz_offset:
            if subtract_background and background_type == 'dynamic':
                # The vz_offset is between z and the raw voltage, not the dynamic baseline transformed voltage
                rzm_vz = np.vstack((rm_orig, zm))
            else:
                rzm_vz = rzm.copy()
                # Remove v_baseline from rzm_vz - don't want to scale the baseline, only the delta
                vb_start, vb_end = self.get_special_indices("v_baseline")
                rzm_vz[:, vb_start:vb_end] = 0

            # VZ offset decays as we move away from overlap
            chrono_vz_strength, eis_vz_strength = self._get_vz_strength_vec(
                sample_times, frequencies, fit_times=sample_times, fit_frequencies=frequencies,
                vz_offset_eps=vz_offset_eps
            )
            eis_vz_strength = np.tile(eis_vz_strength, 2)
            vz_strength_vec = np.concatenate([chrono_vz_strength, eis_vz_strength])
        else:
            rzm_vz = None
            vz_strength_vec = 1

        # Offset voltage baseline
        if times is not None:
            if offset_baseline:
                self.scaled_response_offset = -response_baseline
            else:
                self.scaled_response_offset = 0
            # print('scaled_response_offset:', self.scaled_response_offset * self.response_signal_scale)
            rv = self.scaled_response_signal + self.scaled_response_offset

            if subtract_background and background_type == 'dynamic':
                rv_orig = rv.copy()
                rv = rv - rm_bkg @ rv
            else:
                rv_orig = None
        else:
            rv = None
            rv_orig = None

        # Flatten impedance vector
        if frequencies is not None:
            zv = np.concatenate([z_scaled.real, z_scaled.imag])
        else:
            zv = None

        # Construct hybrid response-impedance vector
        if times is None:
            rzv = zv.copy()
        elif frequencies is None:
            rzv = rv.copy()
        else:
            rzv = np.concatenate([rv, zv])

        # Construct lambda vectors
        l1_lambda_vector = np.zeros(rzm.shape[1])  # No penalty applied to v_baseline, R_inf, and inductance
        l1_lambda_vector[self.get_qp_mat_offset():] = qphb_hypers['l1_lambda_0']
        if self.fit_dop:
            dop_start, dop_end = self.dop_indices
            l1_lambda_vector[dop_start:dop_end] = qphb_hypers['dop_l1_lambda_0']
        if self.fit_ddt:
            ddt_start, ddt_end = self.ddt_indices
            l1_lambda_vector[ddt_start:ddt_end] = qphb_hypers['ddt_l1_lambda_0']

        # Initialize rho and s vectors at prior mode
        k_range = len(qphb_hypers['derivative_weights'])
        rho_vector = qphb_hypers['rho_0'].copy()
        s_vectors = [np.ones(rzm.shape[1]) * qphb_hypers['s_0'][k] for k in range(k_range)]

        if self.fit_dop:
            dop_rho_vector = qphb_hypers['dop_rho_0'].copy()
        else:
            dop_rho_vector = None

        if self.fit_ddt:
            ddt_rho_vector = qphb_hypers['ddt_rho_0'].copy()
        else:
            ddt_rho_vector = None

        # Update the Rp estimate by solving for the DRT coefficients. Important if fitting the DOP
        if scale_data and solve_rp:
            rp_est, dop_rescale_factor, ddt_rescale_factor = self._solve_data_scale(
                qphb_hypers, penalty_matrices, penalty_type, rho_vector,
                dop_rho_vector, s_vectors, rzv, rzm, nonneg, ddt_rho_vector=ddt_rho_vector)
            scale_factor = (qphb_hypers['rp_scale'] / rp_est)
            if self.print_diagnostics:
                print('Solution-based Rp estimate: {:.3f}'.format(rp_est * self.coefficient_scale))
                print('Data rescale factor:', scale_factor)

            # Update data and qphb parameters to reflect new scale
            for data_vec in [rv, zv, rzv, rv_orig]:
                if data_vec is not None:
                    data_vec *= scale_factor
            self.update_data_scale(scale_factor)

            # Update DOP scale to match DRT scale
            if self.fit_dop and self.normalize_dop:
                if self.print_diagnostics:
                    print('DOP rescale factor:', dop_rescale_factor)
                self.dop_scale_vector /= dop_rescale_factor
                dop_start, dop_end = self.dop_indices

                for mat in [rm, zm, rzm, rzm_vz, rm_orig]:
                    if mat is not None:
                        mat[:, dop_start:dop_end] /= dop_rescale_factor

                for mat_name in penalty_matrices.keys():
                    if mat_name[-3:] == 'dop':
                        penalty_matrices[mat_name][dop_start:dop_end, dop_start:dop_end] /= dop_rescale_factor

                self._qp_matrices = {
                    'rm': rm,
                    'zm': zm,
                    'penalty_matrices': penalty_matrices
                }

            # Update DDT scale to match DRT scale
            if self.fit_ddt and self.normalize_ddt and ddt_rescale_factor is not None:
                if self.print_diagnostics:
                    print('DDT rescale factor:', ddt_rescale_factor)
                self.ddt_scale_vector /= ddt_rescale_factor
                ddt_start, ddt_end = self.ddt_indices

                for mat in [rm, zm, rzm, rzm_vz, rm_orig]:
                    if mat is not None:
                        mat[:, ddt_start:ddt_end] /= ddt_rescale_factor

                for mat_name in penalty_matrices.keys():
                    if mat_name[-3:] == 'ddt':
                        penalty_matrices[mat_name][ddt_start:ddt_end, ddt_start:ddt_end] /= ddt_rescale_factor

                self._qp_matrices = {
                    'rm': rm,
                    'zm': zm,
                    'penalty_matrices': penalty_matrices
                }
        # elif scale_data and not solve_rp:
        #     pp.estimate_rp(times, step_times, step_sizes, response_signal, self.step_model, z)

        # Initialize x near zero
        x = np.zeros(rzm.shape[1]) + 1e-6

        # Construct matrices for variance estimation
        if times is not None:
            chrono_vmm = mat1d.construct_chrono_var_matrix(sample_times, self.nonconsec_step_times,
                                                           chrono_vmm_epsilon,
                                                           chrono_error_structure)
        else:
            chrono_vmm = None

        if frequencies is not None:
            eis_vmm = mat1d.construct_eis_var_matrix(frequencies, eis_vmm_epsilon, eis_reim_cor,
                                                     eis_error_structure)
        else:
            eis_vmm = None

        # Stack variance matrices
        if chrono_vmm is None:
            vmm = eis_vmm
        elif eis_vmm is None:
            vmm = chrono_vmm
        else:
            vmm = np.zeros((len(rzv), len(rzv)))
            vmm[:len(sample_times), :len(sample_times)] = chrono_vmm
            vmm[len(sample_times):, len(sample_times):] = eis_vmm

        # Initialize data weight (IMPORTANT)
        # ----------------------------------
        iw_hypers = qphb_hypers.copy()
        iw_hypers['l1_lambda_0'] = iw_l1_lambda_0
        iw_hypers['l2_lambda_0'] = iw_l2_lambda_0
        if 'dop_l2_lambda_0' in qphb_hypers.keys():
            dop_drt_ratio = qphb_hypers['dop_l2_lambda_0'] / qphb_hypers['l2_lambda_0']
            iw_hypers['dop_l2_lambda_0'] = dop_drt_ratio * iw_hypers['l2_lambda_0']
        if 'ddt_l2_lambda_0' in qphb_hypers.keys():
            ddt_drt_ratio = qphb_hypers['ddt_l2_lambda_0'] / qphb_hypers['l2_lambda_0']
            iw_hypers['ddt_l2_lambda_0'] = ddt_drt_ratio * iw_hypers['l2_lambda_0']
        
        # Initialize chrono and eis weights separately
        if init_weights_separately:
            if times is not None:
                chrono_est_weights, chrono_init_weights, x_overfit_chrono, chrono_outlier_t = \
                    qphb.initialize_weights(iw_hypers, penalty_matrices, penalty_type, rho_vector, dop_rho_vector,
                                            s_vectors,
                                            rv, rm, chrono_vmm, nonneg, self.special_qp_params,
                                            ddt_rho_vector=ddt_rho_vector)

                chrono_weight_scale = np.mean(chrono_est_weights ** -2) ** -0.5
            else:
                chrono_est_weights, chrono_init_weights, x_overfit_chrono, chrono_outlier_t = None, None, None, None
                chrono_weight_scale = None

            if frequencies is not None:
                eis_est_weights, eis_init_weights, x_overfit_eis, eis_outlier_t = \
                    qphb.initialize_weights(iw_hypers, penalty_matrices, penalty_type, rho_vector, dop_rho_vector,
                                            s_vectors,
                                            zv, zm, eis_vmm, nonneg, self.special_qp_params,
                                            ddt_rho_vector=ddt_rho_vector)

                eis_weight_scale = np.mean(eis_est_weights ** -2) ** -0.5
            else:
                eis_est_weights, eis_init_weights, x_overfit_eis, eis_outlier_t = None, None, None, None
                eis_weight_scale = None
        else:
            # Initialize weights together
            est_weights, init_weights, x_overfit, outlier_t = \
                    qphb.initialize_weights(iw_hypers, penalty_matrices, penalty_type, rho_vector, dop_rho_vector,
                                            s_vectors,
                                            rzv, rzm, vmm, nonneg, self.special_qp_params,
                                            ddt_rho_vector=ddt_rho_vector)
            
            # Separate EIS and chrono
            if num_eis == 0:
                chrono_est_weights = est_weights
                chrono_init_weights = init_weights
                x_overfit_chrono = x_overfit
                chrono_outlier_t = outlier_t
                chrono_weight_scale = np.mean(chrono_est_weights ** -2) ** -0.5
                eis_est_weights, eis_init_weights, x_overfit_eis, eis_outlier_t = None, None, None, None
                eis_weight_scale = None
            elif num_chrono == 0:
                eis_est_weights = est_weights
                eis_init_weights = init_weights
                x_overfit_eis = x_overfit
                eis_outlier_t = outlier_t
                eis_weight_scale = np.mean(eis_est_weights ** -2) ** -0.5
                chrono_est_weights, chrono_init_weights, x_overfit_chrono, chrono_outlier_t = None, None, None, None
                chrono_weight_scale = None
            else:
                chrono_est_weights, eis_est_weights =  split_vector(est_weights)
                chrono_init_weights, eis_init_weights =  split_vector(init_weights)
                x_overfit_chrono, x_overfit_eis = split_vector(x_overfit)
                chrono_outlier_t, eis_outlier_t = split_vector(outlier_t)
                chrono_weight_scale = np.mean(chrono_est_weights ** -2) ** -0.5
                eis_weight_scale = np.mean(eis_est_weights ** -2) ** -0.5
                
                
        # TEMP TEST
        # ---------
        # est_weights, init_weights, x_overfit, outlier_t = \
        #     qphb.initialize_weights(penalty_matrices, penalty_type, qphb_hypers['derivative_weights'], rho_vector,
        #                             s_vectors, rzv, rzm, vmm, nonneg, self.special_qp_params,
        #                             qphb_hypers['iw_alpha'], qphb_hypers['iw_beta'], qphb_hypers['outlier_p'])
        # 
        # basis_area = basis.get_basis_func_area(self.tau_basis_type, self.tau_epsilon, self.zga_params)
        # fig, ax = plt.subplots()
        # ax.plot(x_overfit)
        # rp = np.sum(np.abs(x_overfit[self.get_qp_mat_offset():])) * basis_area
        # rp = self.predict_r_p(absolute=True, x=x_overfit, raw=True)
        # # rp = 0.045 / self.impedance_scale
        # print('estimated rp from x_overfit:', rp * self.impedance_scale)
        # scale_factor = (qphb_hypers['rp_scale'] / rp)
        # scale_factor = 1
        # print('rescale factor:', scale_factor)
        # 
        # # Update data and qphb parameters to reflect new scale
        # for x_t in [x_overfit_eis, x_overfit_chrono]:
        #     if x_t is not None:
        #         x_t *= scale_factor
        # rzv *= scale_factor
        # est_weights /= scale_factor
        # init_weights /= scale_factor  # update for reference only
        # for weights in [eis_est_weights, eis_init_weights, eis_weight_scale,
        #                 chrono_est_weights, chrono_init_weights, chrono_weight_scale]:
        #     if weights is not None:
        #         weights /= scale_factor
        # self.update_data_scale(scale_factor)
        # ---------
        # END TEST

        # chrono_est_weights = est_weights[:len(rv)]
        # eis_est_weights = est_weights[len(rv):]
        # chrono_init_weights = init_weights[:len(rv)]
        # eis_init_weights = init_weights[len(rv):]

        # eis_est_weights = eis_est_weights_raw * eis_weight_factor
        # eis_init_weights = eis_init_weights_raw * eis_weight_factor

        # Get weight factors
        if data_type == 'hybrid':
            if eis_weight_factor is None or chrono_weight_factor is None:
                if eis_weight_factor is not None or chrono_weight_factor is not None:
                    warnings.warn("Both eis_weight_factor and chrono_weight_factor must be provided. If only one is provided, it will be ignored.")
                if hybrid_weight_factor_method == 'weight':
                    # Set based on ratio of EIS to chrono weights
                    ratio = (eis_weight_scale / chrono_weight_scale) ** 0.25
                    if eis_weight_factor is None:
                        eis_weight_factor = 1 / ratio
                        # eis_weight_factor = (chrono_weight_scale / eis_weight_scale) ** 0.5 * (num_chrono / num_eis) ** 0.25

                    if chrono_weight_factor is None:
                        # chrono_weight_factor = (eis_weight_scale / chrono_weight_scale) ** 0.25
                        # chrono_weight_factor = (eis_weight_scale / chrono_weight_scale) ** 0.5 * (num_eis / num_chrono) ** 0.25
                        chrono_weight_factor = ratio

                elif hybrid_weight_factor_method == 'rp':
                    rp_eis = pp.estimate_rp(None, None, None, None, None, self.z_fit)
                    rp_chrono = pp.estimate_rp(sample_times, self.step_times, self.step_sizes, self.raw_response_signal,
                                               self.step_model, None)
                    rp_tot = self.coefficient_scale * qphb_hypers['rp_scale']

                    if self.print_diagnostics:
                        print('rp_eis: {:.2e}'.format(rp_eis))
                        print('rp_chrono: {:.2e}'.format(rp_chrono))
                        print('rp_tot: {:.2e}'.format(rp_tot))
                        print('rp_chrono / rp_eis:', rp_chrono / rp_eis)
                    # ratio = (rp_chrono / rp_eis) ** 0.5

                    # rp_eis += rp_tot * 0.25
                    # rp_chrono += rp_tot * 0.25

                    # print('rp_tot:', rp_tot)
                    # print('rp_tot factor:', (rp_chrono * rp_eis) ** 0.5 / rp_tot)
                    # rp_tot_factor = (rp_chrono * rp_eis) ** 0.5 / rp_tot

                    if eis_weight_factor is None:
                        # eis_weight_factor = ratio ** -1 * rp_tot_factor ** 0.5
                        # eis_weight_factor = (rp_eis / rp_tot) ** 0.25
                        eis_weight_factor = rp_eis ** 0.75 / (rp_chrono ** 0.25 * rp_tot ** 0.5)
                        # eis_weight_factor = (rp_eis / rp_chrono) ** 0.5

                    if chrono_weight_factor is None:
                        # chrono_weight_factor = ratio * rp_tot_factor ** 0.5
                        # chrono_weight_factor = (rp_chrono / rp_tot) ** 0.25
                        chrono_weight_factor = rp_chrono ** 0.75 / (rp_eis ** 0.25 * rp_tot ** 0.5)
                        # chrono_weight_factor = (rp_chrono / rp_eis) ** 0.5
                elif hybrid_weight_factor_method is None:
                    eis_weight_factor = 1
                    chrono_weight_factor = 1
                else:
                    raise ValueError(f"Invalid hybrid_weight_factor_method argument {hybrid_weight_factor_method}. "
                                     f"Options: 'weight', 'rp', None")

            if self.print_diagnostics:
                print('num_chrono / num_eis:', num_chrono / num_eis)
                print('w_eis / w_chrono:', eis_weight_scale / chrono_weight_scale)
                print('eis weight factor:', eis_weight_factor)
                print('chrono weight factor:', chrono_weight_factor)

            est_weights = np.concatenate([chrono_est_weights, eis_est_weights])
            init_weights = np.concatenate([chrono_init_weights, eis_init_weights])
            outlier_t = np.concatenate([chrono_outlier_t, eis_outlier_t])
        elif data_type == 'eis':
            est_weights = eis_est_weights
            init_weights = eis_init_weights
            outlier_t = eis_outlier_t
        else:
            est_weights = chrono_est_weights
            init_weights = chrono_init_weights
            outlier_t = chrono_outlier_t

        # Return outliers if requested
        # ----------------------------
        if return_outlier_index:
            outlier_index = (1 - outlier_t) > outlier_thresh
            if times is None:
                eis_outlier_index = outlier_index
                chrono_outlier_index = None
            elif frequencies is None:
                eis_outlier_index = None
                chrono_outlier_index = outlier_index
            else:
                chrono_outlier_index = outlier_index[:num_chrono]
                eis_outlier_index = outlier_index[num_chrono:]

            # Consider any z data point with bad real OR imag value to be an outlier
            if eis_outlier_index is not None:
                eis_outlier_index = eis_outlier_index[:len(frequencies)] | eis_outlier_index[len(frequencies):]

            return chrono_outlier_index, eis_outlier_index

        weights = init_weights.copy()

        # Initialize outlier tvt
        if kw.get('outlier_p', None) is not None:
            outlier_tvt = qphb.outlier_tvt(vmm, outlier_t)
        else:
            outlier_tvt = None

        if self.print_diagnostics:
            if chrono_init_weights is not None:
                print('Est chrono weight:', np.mean(chrono_est_weights))
                print('Initial chrono weight:', np.mean(chrono_init_weights))
            if eis_init_weights is not None:
                print('Est EIS weight:', np.mean(eis_est_weights))
                print('Initial EIS weight:', np.mean(eis_init_weights))
        # ------------------------------

        # Initialize xmx_norms at 1
        xmx_norms = np.ones(k_range)
        dop_xmx_norms = np.ones(k_range)
        ddt_xmx_norms = np.ones(k_range)
        # xmx_norms = [10, 0.2, 0.1]

        # TEST: curvature constraints
        if peak_locations is not None:
            drt_curv_matrix = basis.construct_func_eval_matrix(np.log(self.basis_tau), None, self.tau_basis_type,
                                                               self.tau_epsilon, 2, self.zga_params)
            curv_matrix = np.zeros((len(self.basis_tau), len(x)))
            curv_matrix[:, self.get_qp_mat_offset():] = drt_curv_matrix
            peak_indices = np.array([utils.array.nearest_index(np.log(self.basis_tau), np.log(pl))
                                     for pl in peak_locations])
            curv_spread_func = lambda x, order, eps: np.exp(-(eps * np.abs(x)) ** (2 * order))
        else:
            curv_matrix = None
            peak_indices = None
            curv_spread_func = None

        self.qphb_history = []
        it = 0
        # fixed_prior = False

        while it < max_iter:

            if progress_callback is not None:
                progress_callback(it + 1, max_iter)

            x_in = x.copy()

            # Apply chrono/eis weight adjustment factors
            if data_type == 'hybrid':
                weights[:len(rv)] *= chrono_weight_factor
                weights[len(rv):] *= eis_weight_factor

            # # Set the outlier weights to nearly zero
            # if remove_outliers:
            #     weights[outlier_index] = 0

            # Apply overall weight scaling factor
            if it > 0:
                weights = weights * weight_factor
            # print(np.mean(weights))
            # weights = np.ones(len(rzv))

            # TEST: enforce curvature constraint
            if peak_locations is not None:  # and it > 5:
                curv = curv_matrix @ x_in
                peak_curv = curv[peak_indices]
                curv_limit = [2.5 * pc * curv_spread_func(np.log(self.basis_tau / pl), 1.5, 2)
                              for pc, pl in zip(peak_curv, peak_locations)]
                curv_limit = np.sum(curv_limit, axis=0)
                # curv_limit = 0.5 * (curv_limit + curv)

                curv_limit = np.zeros(len(self.basis_tau))
                for index in peak_indices:
                    curv_limit[max(0, index - 5):index + 5] = -100
                print(curv_limit)
                curvature_constraint = (-curv_matrix, -curv_limit)
            else:
                curvature_constraint = None

            # Update data scale as Rp estimate improves to maintain specified rp_scale
            if it > 1 and scale_data and update_scale:
                # Get scale factor
                # basis_area = basis.get_basis_func_area(self.tau_basis_type, self.tau_epsilon, self.zga_params)
                # rp = np.sum(np.abs(x[self.get_qp_mat_offset():])) * basis_area
                rp = self.predict_r_p(absolute=True, x=x, raw=True)
                scale_factor = (qphb_hypers['rp_scale'] / rp) ** 0.5  # Damp the scale factor to avoid oscillations
                if self.print_diagnostics:
                    print('Iter {} scale factor: {:.3f}'.format(it, scale_factor))
                # Update data and qphb parameters to reflect new scale
                for x_t in [x_in, x_overfit_eis, x_overfit_chrono]:
                    if x_t is not None:
                        x_t *= scale_factor
                rzv *= scale_factor
                if rv_orig is not None:
                    rv_orig *= scale_factor
                xmx_norms *= scale_factor ** 0.5  # shouldn't this be scale_factor ** 2?
                if self.fit_dop:
                    dop_xmx_norms *= scale_factor ** 0.5
                if self.fit_ddt:
                    ddt_xmx_norms *= scale_factor ** 0.5
                est_weights /= scale_factor
                init_weights /= scale_factor  # update for reference only
                weights /= scale_factor
                # Update data scale attributes
                self.update_data_scale(scale_factor)

            # Perform actual QPHB operation
            x, s_vectors, rho_vector, dop_rho_vector, ddt_rho_vector, weights, outlier_t, outlier_tvt, cvx_result, converged = \
                qphb.iterate_qphb(x_in, s_vectors, rho_vector, dop_rho_vector, rzv, weights, est_weights, outlier_tvt,
                                  rzm, vmm, penalty_matrices, penalty_type, l1_lambda_vector, qphb_hypers,
                                  eff_hp, xmx_norms, dop_xmx_norms, None, None, curvature_constraint, nonneg,
                                  self.special_qp_params, xtol, 1, self.qphb_history, neg_allowed_indices=neg_allowed_indices,
                                  ddt_rho_vector=ddt_rho_vector, ddt_xmx_norms=ddt_xmx_norms)

            # Normalize to ordinary ridge solution
            if it == 0:
                if self.fit_drt:
                    # Only include DRT penalty in XMX norms
                    x_drt = x[self.get_qp_mat_offset():]
                    pm_drt_list = [penalty_matrices[f'm{k}'][self.get_qp_mat_offset():, self.get_qp_mat_offset():]
                                   for k in range(k_range)]
                    xmx_norms = np.array([x_drt.T @ pm_drt_list[k] @ x_drt for k in range(k_range)])
                    if self.print_diagnostics:
                        print('xmx', xmx_norms)

                if self.fit_dop:
                    dop_start, dop_end = self.dop_indices
                    x_dop = x[dop_start:dop_end]
                    pm_dop_list = [penalty_matrices[f'm{k}'][dop_start:dop_end, dop_start:dop_end]
                                   for k in range(k_range)]
                    dop_xmx_norms = np.array([x_dop.T @ pm_dop_list[k] @ x_dop for k in range(k_range)])
                    if self.print_diagnostics:
                        print('dop_xmx', dop_xmx_norms)

                if self.fit_ddt:
                    ddt_start, ddt_end = self.ddt_indices
                    x_ddt = x[ddt_start:ddt_end]
                    pm_ddt_list = [penalty_matrices[f'm{k}'][ddt_start:ddt_end, ddt_start:ddt_end]
                                   for k in range(k_range)]
                    ddt_xmx_norms = np.array([x_ddt.T @ pm_ddt_list[k] @ x_ddt for k in range(k_range)])
                    if self.print_diagnostics:
                        print('ddt_xmx', ddt_xmx_norms)

            # Optimize the scalar DDT exponent p in the QPHB outer loop
            # (coordinate descent: for candidate p, re-solve the convex QP with
            # the current weights/penalties and minimize its primal objective)
            if self.fit_ddt and self.fit_ddt_p and frequencies is not None and it >= 1:
                from scipy.optimize import minimize_scalar

                ddt_start, ddt_end = self.ddt_indices
                l2_matrix = qphb.calculate_qp_l2_matrix(qphb_hypers, rho_vector, dop_rho_vector,
                                                        penalty_matrices, s_vectors, penalty_type,
                                                        self.special_qp_params, nonlin=False,
                                                        ddt_rho_vector=ddt_rho_vector)
                wm = np.diag(weights)

                def _build_ddt_matrix(pc):
                    if self.tau_dd_basis_type == 'delta':
                        return ddt.get_ddt_impedance_matrix_components(
                            frequencies, self.basis_tau_dd, self.ddt_kernel_type, p=pc)
                    else:
                        return ddt._construct_ddt_block_gaussian_gh(
                            2 * np.pi * frequencies, self.basis_tau_dd,
                            self.ddt_kernel_type, pc, self.tau_dd_epsilon)

                def _qp_objective(pc):
                    zm_u = _build_ddt_matrix(pc)
                    s_c = 1 / np.max(np.abs(zm_u), axis=0)
                    zsc = zm_u * s_c
                    rzm_c = rzm.copy()
                    rzm_c[num_chrono:num_chrono + num_eis, ddt_start:ddt_end] = zsc.real
                    rzm_c[num_chrono + num_eis:num_chrono + 2 * num_eis, ddt_start:ddt_end] = zsc.imag
                    cvx = qphb.solve_convex_opt(wm @ rzv, wm @ rzm_c, l2_matrix, l1_lambda_vector,
                                                nonneg, self.special_qp_params,
                                                neg_allowed_indices=neg_allowed_indices)
                    return float(cvx['primal objective'])

                res = minimize_scalar(_qp_objective, bounds=self.ddt_p_bounds, method='bounded',
                                      options={'xatol': 1e-3})
                p_new = float(res.x)
                if np.isfinite(p_new) and abs(p_new - self.ddt_p) > 1e-5:
                    self.ddt_p = p_new
                    zm_u = _build_ddt_matrix(self.ddt_p)
                    s_new = 1 / np.max(np.abs(zm_u), axis=0)
                    self.ddt_scale_vector = s_new
                    zsc = zm_u * s_new
                    rzm[num_chrono:num_chrono + num_eis, ddt_start:ddt_end] = zsc.real
                    rzm[num_chrono + num_eis:num_chrono + 2 * num_eis, ddt_start:ddt_end] = zsc.imag
                    if self.print_diagnostics:
                        print('Iter {} updated ddt_p = {:.4f}'.format(it, self.ddt_p))

            # Update background estimate
            if times is not None and subtract_background and background_type == 'scaled':
                # Use the residuals from the current iteration to estimate the background
                y_hat = rzm @ x
                y_err_chrono = (rzv - y_hat)[:num_chrono]
                y_bkg = rm_bkg @ y_err_chrono
                rzm[:num_chrono, self.special_qp_params['background_scale']['index']] = y_bkg

            # Update vz_offset column
            if data_type == 'hybrid' and vz_offset:
                # Update the response matrix with the current predicted y vector
                # vz_offset offsets chrono and eis predictions
                y_hat = rzm_vz @ x
                vz_sep = y_hat.copy()
                vz_sep[len(rv):] *= -1  # vz_offset > 0 means EIS Rp smaller chrono Rp
                rzm[:, self.special_qp_params['vz_offset']['index']] = vz_sep * vz_strength_vec
                # if rm_orig is not None:
                #     rm_orig[:, self.special_qp_params['vz_offset']['index']] = vz_sep[:len(rv)]

            if converged:
                break
            elif it == max_iter - 1 and self.warn:
                warnings.warn(f'Solution did not converge within {max_iter} iterations. This is usually not an issue.')

            it += 1

        # Re-apply weight scaling factor, since weights come out of iterate_qphb
        weights = weights * weight_factor
        
        # # Set the outlier weights to nearly zero (final update)
        # if remove_outliers:
        #     weights[outlier_index] = 1e-10

        # Store the scaled weights for reference
        scaled_weights = weights.copy()
        if data_type == 'hybrid':
            # weights from iterate_qphb does not have weight factors applied
            scaled_weights[:len(rv)] *= chrono_weight_factor
            scaled_weights[len(rv):] *= eis_weight_factor

        # Store QPHB diagnostic parameters
        # TODO: should scaled_weights or weights be used here? weights seems to make sense...
        p_matrix, q_vector = qphb.calculate_pq(rzm, rzv, penalty_matrices, penalty_type, qphb_hypers, l1_lambda_vector,
                                               rho_vector, dop_rho_vector, s_vectors, scaled_weights,
                                               self.special_qp_params, ddt_rho_vector=ddt_rho_vector)

        # post_lp = qphb.evaluate_posterior_lp(x, penalty_matrices, penalty_type, qphb_hypers, l1_lambda_vector,
        #                                      rho_vector, s_vectors, weights, rzm, rzv, xmx_norms)

        # Calculate the estimated background
        if subtract_background and times is not None:
            if background_type == 'dynamic':
                resid = rv_orig - rm_orig @ x
                self.raw_response_background = (rm_bkg @ resid) * self.response_signal_scale
            elif background_type == 'scaled':
                rzm_resid = rzm.copy()
                x_bkg_index = self.special_qp_params['background_scale']['index']
                rzm_resid[:, x_bkg_index] = 0
                resid = (rzv - rzm_resid @ x)[:num_chrono]
                self.raw_response_background = (rm_bkg @ resid) * self.response_signal_scale * x[x_bkg_index]
            else:
                # Add the background back to the raw data
                self.raw_response_signal = self.raw_response_signal + self.raw_response_background
        elif times is not None:
            self.raw_response_background = np.zeros(len(sample_times))
        else:
            self.raw_response_background = None

        if self.print_diagnostics:
            w_mat = np.diag(scaled_weights)
            wtw = w_mat.T @ w_mat
            rss = rzv.T @ wtw @ rzv - 2 * rzv.T @ wtw @ rzm @ x + x.T @ rzm.T @ wtw @ rzm @ x
            wtd_resid = w_mat @ (rzv - rzm @ x)
            print('rss:', rss)
            print('Mean wtd. resid:', np.mean(np.abs(wtd_resid)))

        self.qphb_params = {'est_weights': est_weights.copy(),
                            'init_weights': init_weights.copy(),
                            'weights': scaled_weights.copy(),  # scaled weights
                            'true_weights': weights.copy(),  # unscaled weights
                            'data_factor': data_factor,
                            'chrono_weight_factor': chrono_weight_factor,
                            'eis_weight_factor': eis_weight_factor,
                            'xmx_norms': xmx_norms.copy(),
                            'dop_xmx_norms': dop_xmx_norms,
                            'ddt_xmx_norms': ddt_xmx_norms,
                            'x_overfit_chrono': x_overfit_chrono,
                            'x_overfit_eis': x_overfit_eis,
                            'p_matrix': p_matrix,
                            'q_vector': q_vector,
                            'rho_vector': rho_vector,
                            'dop_rho_vector': dop_rho_vector,
                            'ddt_rho_vector': ddt_rho_vector,
                            's_vectors': s_vectors,
                            'outlier_t': outlier_t,
                            'vmm': vmm,
                            'l1_lambda_vector': l1_lambda_vector,
                            # 'posterior_lp': post_lp,
                            'rm': rzm,
                            'rv': rzv,
                            'penalty_matrices': penalty_matrices,
                            'hypers': qphb_hypers,
                            'num_eis': num_eis,
                            'num_chrono': num_chrono,
                            'rm_bkg': rm_bkg,
                            'rm_orig': rm_orig,
                            'rv_orig': rv_orig,
                            'vz_strength_vec': vz_strength_vec,
                            # 'outlier_index': outlier_index,
                            # 'wtd_resid': wtd_resid
                            }

        if self.print_diagnostics:
            print('rho:', rho_vector)
            if self.fit_dop:
                print('dop_rho:', dop_rho_vector)

        # Store final cvxopt result
        self.cvx_result = cvx_result

        # Get sigma vector from (unscaled) weights
        sigma_vec = (weights ** -1)
        if data_type == 'hybrid':
            v_sigma = sigma_vec[:len(rv)] * self.response_signal_scale
            z_sigma = utils.eis.concat_vector_to_complex(sigma_vec[len(rv):]) * self.impedance_scale
        elif data_type == 'eis':
            z_sigma = utils.eis.concat_vector_to_complex(sigma_vec) * self.impedance_scale
            v_sigma = None
        else:
            v_sigma = sigma_vec * self.response_signal_scale
            z_sigma = None

        # Extract model parameters
        x_out = np.array(list(cvx_result['x']))
        self.fit_parameters = self.extract_qphb_parameters(x_out)
        self.fit_parameters['v_sigma_tot'] = v_sigma
        self.fit_parameters['v_sigma_res'] = None
        self.fit_parameters['z_sigma_tot'] = z_sigma
        self.fit_parameters['vz_offset_eps'] = vz_offset_eps
        self.fit_parameters['p_matrix'] = p_matrix
        self.fit_parameters['q_vector'] = q_vector

        self.fit_type = f'qphb_{data_type}'

    def get_offset_pq(self, weights=None, filter_weights=False, unscale=False):
        # Get necessary matrices/vectors
        rzm = self.qphb_params['rm'].copy()
        rzv = self.qphb_params['rv'].copy()
        penalty_matrices = deepcopy(self.qphb_params['penalty_matrices'])
        penalty_type = self.fit_kwargs['penalty_type']
        qphb_hypers = self.qphb_params['hypers']
        l1_lambda_vector = self.qphb_params['l1_lambda_vector'].copy()
        rho_vector = self.qphb_params['rho_vector']
        dop_rho_vector = self.qphb_params['dop_rho_vector']
        ddt_rho_vector = self.qphb_params.get('ddt_rho_vector', None)
        s_vectors = self.qphb_params['s_vectors'].copy()

        if weights is None:
            scaled_weights = self.qphb_params['weights']
            # print(np.mean(scaled_weights))

            # TODO: compare and determine how to simplify weights. Seems like mean of all weights is safest...

            if filter_weights:
                num_chrono = self.num_chrono
                chrono_weights = scaled_weights[:num_chrono]
                eis_weights = scaled_weights[num_chrono:]
                # chrono_weights = ndimage.median_filter(chrono_weights, size=5, mode='reflect')
                # eis_weights = ndimage.median_filter(eis_weights, size=5, mode='reflect')
                chrono_weights = np.ones_like(chrono_weights) * np.mean(chrono_weights)
                eis_weights = np.ones_like(eis_weights) * np.mean(eis_weights)
                scaled_weights = np.concatenate([chrono_weights, eis_weights])
            else:
                scaled_weights = np.ones_like(scaled_weights) * np.mean(scaled_weights)
        else:
            scaled_weights = weights

        if unscale:
            # rzv = rzv * self.coefficient_scale
            scaled_weights = scaled_weights / self.coefficient_scale
            for key in list(penalty_matrices.keys()):
                mat = penalty_matrices[key]
                penalty_matrices[key] = mat / self.coefficient_scale ** 2

        # Get raw parameter values
        x_raw = np.array(list(self.cvx_result['x']))

        # For each offset parameter, add its influence to the data vector
        del_index = []
        for name in ['v_baseline', 'vz_offset']:
            if name in self.special_qp_params.keys():
                for ii in range(self.special_qp_params[name]["size"]):
                    index = self.special_qp_params[name]['index'] + ii
                    val = x_raw[index]
                    rzv -= rzm[:, index] * val

                    del_index.append(index)

        if len(del_index) > 0:
            # After accounting for offsets, delete the corresponding matrix/vector entries
            # Response matrix - delete columns
            rzm = np.delete(rzm, del_index, axis=1)

            # Penalty matrices - delete rows and columns
            for key in list(penalty_matrices.keys()):
                mat = penalty_matrices[key]
                penalty_matrices[key] = np.delete(np.delete(mat, del_index, axis=0), del_index, axis=1)

            # Vectors - delete entries
            l1_lambda_vector = np.delete(l1_lambda_vector, del_index)
            for i, sv in enumerate(s_vectors):
                s_vectors[i] = np.delete(sv, del_index)

            # Remove entries from special_qp dict
            special_qp = deepcopy(self.special_qp_params)
            for name in ['v_baseline', 'vz_offset']:
                if name in special_qp.keys():
                    del special_qp[name]

            # Shift indices of remaining special params
            for key in list(special_qp.keys()):
                index = special_qp[key]['index']
                shift = np.sum([1 if di < index else 0 for di in del_index])
                special_qp[key]['index'] = index - shift

            p_matrix, q_vector = qphb.calculate_pq(rzm, rzv, penalty_matrices, penalty_type, qphb_hypers,
                                                   l1_lambda_vector,
                                                   rho_vector, dop_rho_vector, s_vectors, scaled_weights,
                                                   special_qp, ddt_rho_vector=ddt_rho_vector)
        else:
            # No offsets to remove
            p_matrix = self.fit_parameters['p_matrix'].copy()
            q_vector = self.fit_parameters['q_vector'].copy()

        return p_matrix, q_vector

    def fit_chrono(self, times, i_signal, v_signal, step_times=None, step_sizes=None,
                   nonneg=True, neg_allowed_tau_range: Optional[Tuple[float, float]] = None,
                   scale_data=True, update_scale=False,
                   offset_baseline=True, offset_steps=True, step_offset_size=None, discard_first_n: Optional[int] = None,
                   subtract_background=False, estimate_background_kw=None,
                   downsample=False, downsample_kw=None, smooth_inf_response=True,
                   error_structure='uniform', vmm_epsilon=4,
                   **kwargs):

        self._qphb_fit_core(times, i_signal, v_signal, None, None, step_times=step_times, step_sizes=step_sizes, 
                            nonneg=nonneg, neg_allowed_tau_range=neg_allowed_tau_range,
                            scale_data=scale_data, update_scale=update_scale, offset_steps=offset_steps, step_offset_size=step_offset_size,
                            discard_first_n=discard_first_n,
                            offset_baseline=offset_baseline, downsample=downsample, downsample_kw=downsample_kw,
                            subtract_background=subtract_background, estimate_background_kw=estimate_background_kw,
                            smooth_inf_response=smooth_inf_response, chrono_error_structure=error_structure,
                            chrono_vmm_epsilon=vmm_epsilon, **kwargs)

    def fit_eis(self, frequencies, z, nonneg=True, neg_allowed_tau_range: Optional[Tuple[float, float]] = None,
                scale_data=True, update_scale=False,
                error_structure=None, vmm_epsilon=0.25, vmm_reim_cor=0.25, **kwargs):
        """
        Perform a conventional DRT fit of EIS data.
        :param ndarray frequencies: array of frequencies
        :param ndarray z: complex array of impedance values
        :param bool nonneg: if True, constrain the DRT to be non-negative. If False, allow negative DRT values
        :param bool scale_data: if True, scale the impedance prior to fitting. If set to False, the model tuning will
        not work as intended and may yield unexpected results.
        :param bool update_scale: if True, update the data scale during solution iterations. Setting this to False
        provides more stable performance, but setting this to True can be helpful if your data is truncated or contains
        outliers
        :param str error_structure: error structure to use for the data. If None, a flexible error structure based on
        residual filtering is used. If 'uniform', a uniform error structure is assumed
        :param float vmm_epsilon: inverse scale parameter for flexible error structure filter. Larger values will allow
        greater local variations in error scale
        :param float vmm_reim_cor: correlation factor between real and imaginary error structures. A value of 1 forces
        the reak and imaginary error structures to be the same, while a value of 0 corresponds to completely independent
        real and imaginary error structures
        :param kwargs: additional keyword args passed to _qphb_fit_core
        :return:
        """

        self._qphb_fit_core(None, None, None, frequencies, z, nonneg=nonneg, neg_allowed_tau_range=neg_allowed_tau_range, scale_data=scale_data,
                            update_scale=update_scale, eis_error_structure=error_structure, eis_vmm_epsilon=vmm_epsilon,
                            eis_reim_cor=vmm_reim_cor, **kwargs)
            

    def fit_hybrid(self, times, i_signal, v_signal, frequencies, z, step_times=None, step_sizes=None,
                   nonneg=True, neg_allowed_tau_range: Optional[Tuple[float, float]] = None, scale_data=True, update_scale=False,
                   # chrono parameters
                   offset_steps=True, step_offset_size=None, discard_first_n: Optional[int] = None,
                   offset_baseline=True, subtract_background=False, estimate_background_kw=None,
                   downsample=False, downsample_kw=None, smooth_inf_response=True,
                   # vz offset
                   vz_offset=True, vz_offset_scale=1, vz_offset_eps=1,
                   # Error structure
                   chrono_error_structure='uniform', eis_error_structure=None,
                   chrono_vmm_epsilon=4, eis_vmm_epsilon=0.25, eis_reim_cor=0.25,
                   eis_weight_factor=None, chrono_weight_factor=None, **kwargs):


        self._qphb_fit_core(times, i_signal, v_signal, frequencies, z, step_times=step_times, step_sizes=step_sizes, 
                            nonneg=nonneg, neg_allowed_tau_range=neg_allowed_tau_range,
                            scale_data=scale_data, update_scale=update_scale, offset_steps=offset_steps, step_offset_size=step_offset_size,
                            discard_first_n=discard_first_n,
                            offset_baseline=offset_baseline, downsample=downsample, downsample_kw=downsample_kw,
                            subtract_background=subtract_background, estimate_background_kw=estimate_background_kw,
                            smooth_inf_response=smooth_inf_response, chrono_error_structure=chrono_error_structure,
                            eis_error_structure=eis_error_structure, chrono_vmm_epsilon=chrono_vmm_epsilon,
                            eis_vmm_epsilon=eis_vmm_epsilon, eis_reim_cor=eis_reim_cor, vz_offset=vz_offset,
                            vz_offset_scale=vz_offset_scale, vz_offset_eps=vz_offset_eps,
                            eis_weight_factor=eis_weight_factor, chrono_weight_factor=chrono_weight_factor, **kwargs)

    def _continue_from_init(self, qphb_hypers, x_init, rv, rm, vmm, rho_vector, dop_rho_vector, s_vectors, outlier_t,
                            penalty_matrices, xmx_norms, dop_xmx_norms,
                            est_weights, weights, l1_lambda_vector,
                            nonneg=True, update_scale=False, weight_factor=1,
                            eis_weight_factor=None, chrono_weight_factor=None,
                            penalty_type='integral', eff_hp=True, xtol=1e-2, max_iter=10, min_iter=2,
                            ddt_rho_vector=None, ddt_xmx_norms=None, progress_callback=None, **kw):

        # Update hyperparameters with user-specified values
        qphb_hypers = qphb_hypers.copy()
        qphb_hypers.update(kw)
        # print(qphb_hypers)
        # print(rv[-10:])
        # print(self.coefficient_scale)

        if eis_weight_factor is None:
            eis_weight_factor = self.qphb_params['chrono_weight_factor']
        if chrono_weight_factor is None:
            chrono_weight_factor = self.qphb_params['chrono_weight_factor']

        x = x_init.copy()
        continue_history = []
        it = 0

        if 'vz_offset' in self.special_qp_params.keys():
            # Make a copy for vz_offset calculation
            rzm_vz = rm.copy()
            # Remove v_baseline from rzm_vz - don't want to scale the baseline, only the sample response
            vb_start, vb_end = self.get_special_indices("v_baseline")
            rzm_vz[:, vb_start:vb_end] = 0
            vz_strength_vec = self.qphb_params['vz_strength_vec']
        else:
            rzm_vz = None
            vz_strength_vec = 1

        # Initialize outlier_tvt
        if qphb_hypers.get('outlier_p', None) is not None:
            outlier_tvt = qphb.outlier_tvt(vmm, outlier_t)
        else:
            outlier_tvt = None

        while it < max_iter:

            if progress_callback is not None:
                progress_callback(it + 1, max_iter)

            x_in = x.copy()

            # Apply chrono/eis weight adjustment factors
            if self.fit_type.find('hybrid') > -1:
                weights[:self.qphb_params['num_chrono']] *= chrono_weight_factor
                weights[self.qphb_params['num_chrono']:] *= eis_weight_factor

            weights = weights * weight_factor

            # Update data scale as Rp estimate improves to maintain specified rp_scale
            if it > 1 and update_scale:
                # Get scale factor
                rp = self.predict_r_p(absolute=True, x=x, raw=True)
                scale_factor = (qphb_hypers['rp_scale'] / rp) ** 0.5  # Damp the scale factor to avoid oscillations
                if self.print_diagnostics:
                    print('Iter {} scale factor: {:.3f}'.format(it, scale_factor))

                # Update data and qphb parameters to reflect new scale
                x_in *= scale_factor
                rv *= scale_factor
                xmx_norms *= scale_factor ** 0.5
                if self.fit_dop:
                    dop_xmx_norms *= scale_factor ** 0.5
                if self.fit_ddt:
                    ddt_xmx_norms *= scale_factor ** 0.5
                est_weights /= scale_factor
                weights /= scale_factor
                # Update data scale attributes
                self.update_data_scale(scale_factor)

            # Perform actual QPHB operation
            x, s_vectors, rho_vector, dop_rho_vector, ddt_rho_vector, weights, outlier_t, outlier_tvt, cvx_result, converged = \
                qphb.iterate_qphb(x_in, s_vectors, rho_vector, dop_rho_vector, rv, weights, est_weights, outlier_tvt,
                                  rm,
                                  vmm, penalty_matrices, penalty_type, l1_lambda_vector, qphb_hypers, eff_hp, xmx_norms,
                                  dop_xmx_norms, None, None, None, nonneg, self.special_qp_params, xtol, 1,
                                  continue_history, 
                                  neg_allowed_indices=self._get_neg_allowed_indices(self.fit_kwargs["nonneg"], self.fit_kwargs["neg_allowed_tau_range"]),
                                  ddt_rho_vector=ddt_rho_vector, ddt_xmx_norms=ddt_xmx_norms)

            # Update vz_offset column
            if self.fit_type.find('hybrid') > -1 and 'vz_offset' in self.special_qp_params.keys():
                # Update the response matrix with the current predicted y vector
                # vz_offset offsets chrono and eis predictions
                y_hat = rzm_vz @ x
                vz_sep = y_hat.copy()
                vz_sep[self.qphb_params['num_chrono']:] *= -1  # vz_offset > 0 means EIS Rp smaller than chrono Rp
                rm[:, self.special_qp_params['vz_offset']['index']] = vz_sep * vz_strength_vec

            if converged and it >= min_iter - 1:
                break
            elif it == max_iter - 1 and self.warn:
                warnings.warn(f'Solution did not converge within {max_iter} iterations')

            it += 1

        return continue_history

    @property
    def num_drt_params(self):
        if not self.fit_drt or self.basis_tau is None:
            return 0
        return len(self.basis_tau) * (1 + int(self.series_neg))

    def get_drt_params(self, x, sign=1):
        if x is not None:
            if isinstance(x, dict):
                # fit_parameter dict provided. Get DRT params only
                x = x["x"]
            elif len(x) > self.num_drt_params:
                x = self.extract_qphb_parameters(x)['x']
        else:
            x = self.fit_parameters['x']

        if self.series_neg:
            # Get the coefficients corresponding to the requested sign
            if sign == 1:
                return x[:len(self.basis_tau)]
            elif sign == -1:
                return -x[len(self.basis_tau):]
            elif sign == 0:
                # Net distribution (pos + neg)
                return x[:len(self.basis_tau)] - x[len(self.basis_tau):]
            else:
                raise ValueError(f'Invalid sign {sign}. Options: -1, 0, 1')

        return x

    @property
    def default_dist_sign(self):
        if self.series_neg:
            return 0
        else:
            return 1

    @property
    def dop_indices(self):
        if self.fit_dop:
            start_index = self.special_qp_params['x_dop']['index']
            end_index = start_index + len(self.basis_nu)
        else:
            start_index, end_index = None, None

        return start_index, end_index

    def get_special_indices(self, key):
        start_index = self.special_qp_params[key]['index']
        end_index = start_index + self.special_qp_params[key].get('size', 1)
        return start_index, end_index

    def get_dop_params(self, x=None):
        if x is not None:
            if len(x) > len(self.basis_nu):
                x = self.extract_qphb_parameters(x)['x_dop']
        else:
            x = self.fit_parameters['x_dop']

        return x

    @property
    def ddt_indices(self):
        if self.fit_ddt:
            start_index = self.special_qp_params['x_ddt']['index']
            end_index = start_index + self.num_ddt_params
        else:
            start_index, end_index = None, None

        return start_index, end_index

    @property
    def num_ddt_params(self):
        if self.basis_tau_dd is None:
            return 0
        n_p = 1 if self.basis_p is None else len(self.basis_p)
        return len(self.basis_tau_dd) * n_p

    def get_ddt_params(self, x=None):
        if x is not None:
            if len(x) > len(self.basis_tau_dd):
                x = self.extract_qphb_parameters(x)['x_ddt']
        else:
            x = self.fit_parameters['x_ddt']

        return x
    
    def get_drt_norm(self, normalize: bool, normalize_by: Optional[float] = None, 
                     x: Optional[ndarray] = None, absolute=False):
        if normalize_by is not None:
            normalize = True
            
        if normalize:
            if normalize_by is None:
                normalize_by = self.predict_r_p(x=x, absolute=absolute)
        else:
            normalize_by = 1
            
        return normalize_by
    
    def predict_distribution(self, tau=None, ppd=20, x=None, order=0, sign=1, 
                             normalize=False, normalize_by: Optional[float] = None, 
                             abs_norm: bool = False):
        warnings.warn("predict_distribution is deprecated and will be removed in the future. Please use predict_drt instead", DeprecationWarning)
        return self.predict_drt(tau=tau, ppd=ppd, x=x, order=order, sign=sign, 
                                normalize=normalize, normalize_by=normalize_by, abs_norm=abs_norm)

    def predict_drt(self, tau=None, ppd=20, x=None, order=0, sign=1, 
                             normalize=False, normalize_by: Optional[float] = None, 
                             abs_norm: bool = False):
        """ 
        Predict distribution as function of tau
        :param ndarray tau: tau values at which to evaluate the distribution
        :return: array of distribution density values
        """
        if not self.fit_drt or self.basis_tau is None:
            # Pure DOP / DDT mode: no DRT distribution
            return np.array([])

        # If tau is not provided, go one decade beyond self.basis_tau with finer spacing
        if tau is None:
            tau = self.get_tau_eval(ppd)

        # Construct basis matrix
        basis_matrix = basis.construct_func_eval_matrix(np.log(self.basis_tau), np.log(tau),
                                                        self.tau_basis_type, epsilon=self.tau_epsilon,
                                                        order=order, zga_params=self.zga_params)

        x = self.get_drt_params(x, sign)

        normalize_by = self.get_drt_norm(normalize, normalize_by, x=x, absolute=abs_norm)

        return basis_matrix @ x / normalize_by

    def estimate_distribution_cov(self, tau=None, ppd=20, p_matrix=None, sign=1, order=0, 
                                  normalize=False, normalize_by: Optional[float] = None,
                                  var_floor=0.0, tau_data_limits=None, extend_var=False):
        """
        DRT parameter covariance matrix
        :param ndarray tau: tau values at which to evaluate the distribution
        :return: array of distribution density values
        """
        if not self.fit_drt or self.basis_tau is None:
            # Pure DOP / DDT mode: no DRT covariance
            return None

        # If tau is not provided, go one decade beyond self.basis_tau with finer spacing
        if tau is None:
            tau = self.get_tau_eval(ppd)

        # Construct basis matrix
        basis_matrix = basis.construct_func_eval_matrix(np.log(self.basis_tau), np.log(tau),
                                                        self.tau_basis_type, epsilon=self.tau_epsilon,
                                                        order=order, zga_params=self.zga_params)

        # Get normalization - square for covariance
        normalize_by = self.get_drt_norm(normalize, normalize_by)
        normalize_by **= 2

        # Get parameter covariance
        x_cov = self.estimate_param_cov(p_matrix)

        if x_cov is not None:
            # Limit to DRT parameters
            x_cov = x_cov[self.get_qp_mat_offset():, self.get_qp_mat_offset():]

            # Limit to requested sign
            if self.series_neg:
                tau_len = len(self.basis_tau)
                if sign == 1:
                    x_cov = x_cov[:tau_len, :tau_len]
                elif sign == -1:
                    x_cov = x_cov[tau_len:, tau_len:]
                elif sign == 0:
                    x_cov_pos = x_cov[:tau_len, :tau_len]
                    x_cov_neg = x_cov[tau_len:, tau_len:]
                    x_cov_pn = x_cov[:tau_len, tau_len:]
                    x_cov_np = x_cov[tau_len:, :tau_len]
                    x_cov = x_cov_pos + x_cov_neg - (x_cov_pn + x_cov_np)

            # Distribution covariance given by matrix product
            dist_cov = basis_matrix @ x_cov @ basis_matrix.T

            # Normalize
            dist_cov = dist_cov / normalize_by

            # Add variance beyond measurement bounds
            # if extrapolation_var_scale > 0:
            #     add_var = np.zeros(len(tau))
            #     if tau_data_limits is None:
            #         tau_data_limits = pp.get_tau_lim(self.get_fit_frequencies(True), self.get_fit_times(True),
            #                                          self.step_times)
            #     t_start, t_end = tau_data_limits
            #     add_var[tau > t_end] = np.log(tau[tau > t_end] / t_end) ** 2
            #     add_var[tau < t_start] = (-np.log(tau[tau < t_start] / t_start)) ** 2
            #     add_var *= extrapolation_var_scale
            #     dist_cov += np.diag(add_var)

            # Extend variance beyond measurement bounds
            if extend_var:
                if tau_data_limits is None:
                    if self.basis_tau is not None and self.get_fit_frequencies(True) is None and self.get_fit_times(True) is None:
                        # No data stored in instance. Most likely extracted from DRTMD
                        # Assuming that basis_tau is extended by 1 decade, go in 1 decade from each side
                        tau_data_limits = (self.basis_tau[9], self.basis_tau[-10])
                    else:    
                        tau_data_limits = pp.get_tau_lim(self.get_fit_frequencies(True), self.get_fit_times(True),
                                                        self.step_times)
                t_left, t_right = tau_data_limits
                left_index = utils.array.nearest_index(tau, t_left) + 1
                right_index = utils.array.nearest_index(tau, t_right)
                var = np.diag(dist_cov).copy()
                # Set the variance outside the measurement bounds to the variance at the corresponding bound
                var[:left_index] = np.maximum(var[:left_index], var[left_index])
                var[right_index:] = np.maximum(var[right_index:], var[right_index])
                dist_cov[np.diag_indices(dist_cov.shape[0])] = var

            # Set variance floor
            if var_floor > 0:
                dist_var = np.diag(dist_cov).copy()
                dist_var[dist_var < var_floor] = var_floor
                np.fill_diagonal(dist_cov, dist_var)

            return dist_cov
        else:
            # Error in precision matrix inversion
            return None

    def estimate_dop_cov(self, nu=None, p_matrix=None, normalize=False, 
                         normalize_tau=None, normalize_quantiles=(0.25, 0.75), 
                         var_floor=0.0, order=0,
                         delta_density=False):
        # If tau is not provided, go one decade beyond self.basis_tau with finer spacing
        if nu is None:
            nu = self.basis_nu

        # Construct basis matrix
        basis_matrix = basis.construct_func_eval_matrix(self.basis_nu, nu,
                                                        self.nu_basis_type, epsilon=self.nu_epsilon,
                                                        order=order, zga_params=None)

        # Normalize
        normalize_by = self.get_dop_norm(nu, normalize, normalize_tau, normalize_quantiles)
        normalize_by **= 2

        # Get parameter covariance
        x_cov = self.estimate_param_cov(p_matrix)

        if x_cov is not None:
            # Limit to DOP parameters
            dop_start, dop_end = self.dop_indices
            x_cov = x_cov[dop_start:dop_end, dop_start:dop_end]

            # Normalize delta function magnitude to grid spacing
            if delta_density and self.nu_basis_type == 'delta':
                dnu = self.get_nu_basis_spacing()
                x_cov = x_cov / dnu

            # Distribution covariance given by matrix product
            dist_cov = basis_matrix @ x_cov @ basis_matrix.T

            # Normalize
            dist_cov = dist_cov / normalize_by

            # Set variance floor
            if var_floor > 0:
                dist_var = np.diag(dist_cov).copy()
                dist_var[dist_var < var_floor] = var_floor
                np.fill_diagonal(dist_cov, dist_var)

            return dist_cov
        else:
            # Error in precision matrix inversion
            return None
        
    def predict_distribution_ci(self, tau=None, ppd=20, x=None, order=0, sign=1, 
                                normalize=False, normalize_by: Optional[float] = None,
                                quantiles=[0.025, 0.975]):
        warnings.warn("predict_distribution_ci is deprecated. Please use predict_drt_ci instead", DeprecationWarning)
        return self.predict_drt_ci(tau=tau, ppd=ppd, x=x, order=order, sign=sign,
                                   normalize=normalize, normalize_by=normalize_by,
                                   quantiles=quantiles)
        

    def predict_drt_ci(self, tau=None, ppd=20, x=None, order=0, sign=1, 
                                normalize=False, normalize_by: Optional[float] = None,
                                quantiles=[0.025, 0.975]):
        # Get distribution std
        dist_cov = self.estimate_distribution_cov(tau=tau, ppd=ppd, order=order, sign=sign, 
                                                  normalize=normalize, normalize_by=normalize_by)
        if dist_cov is not None:
            dist_sigma = np.diag(dist_cov) ** 0.5

            # Distribution mean
            dist_mu = self.predict_drt(tau=tau, ppd=ppd, x=x, order=order, sign=sign, 
                                                normalize=normalize, normalize_by=normalize_by)

            # Determine number of std devs to obtain quantiles
            s_lo, s_hi = stats.std_normal_quantile(quantiles)

            dist_lo = dist_mu + s_lo * dist_sigma
            dist_hi = dist_mu + s_hi * dist_sigma

            return dist_lo, dist_hi
        else:
            # Error in covariance calculation
            return None, None

    def predict_dop_ci(self, nu=None, x=None, normalize=False, normalize_tau=None, quantiles=[0.025, 0.975],
                       order=0, normalize_quantiles=(0.25, 0.75), delta_density=False, include_ideal=True):
        # Get distribution std
        dist_cov = self.estimate_dop_cov(nu, order=order, normalize=normalize, 
                                         normalize_tau=normalize_tau,
                                         normalize_quantiles=normalize_quantiles,
                                         delta_density=delta_density)
        if dist_cov is not None:
            dist_sigma = np.diag(dist_cov) ** 0.5

            # Distribution mean
            dist_mu = self.predict_dop(nu=nu, x=x, normalize=normalize,
                                       normalize_tau=normalize_tau, order=order,
                                       normalize_quantiles=normalize_quantiles,
                                       delta_density=delta_density, include_ideal=include_ideal)

            # Determine number of std devs to obtain quantiles
            s_lo, s_hi = stats.std_normal_quantile(quantiles)

            dist_lo = dist_mu + s_lo * dist_sigma
            dist_hi = dist_mu + s_hi * dist_sigma

            return dist_lo, dist_hi
        else:
            # Error in covariance calculation
            return None, None

    def get_nu_basis_spacing(self):
        # For each grid point, get the minimum distance to next basis location
        if self.fixed_basis_nu is not None:
            basis_nu = self.fixed_basis_nu
        else:
            basis_nu = self.basis_nu

        dnu = np.diff(np.sort(basis_nu))
        dnu = np.minimum(dnu[1:], dnu[:-1])
        median_dnu = np.median(dnu)
        dnu = np.append(np.insert(dnu, 0, median_dnu), median_dnu)
        return dnu

    def predict_dop(self, nu=None, x=None, normalize=False, normalize_tau=None, order=0, return_nu=False,
                    normalize_quantiles=(0, 1), delta_density=False, include_ideal=True):
        if nu is None:
            nu = np.linspace(-1, 1, 1001)
            # Ensure basis_nu is included
            nu = np.unique(np.concatenate([self.basis_nu, nu]))
            # Ensure pure inductance, resistance, and capacitance are included
            nu = np.unique(np.concatenate([nu, np.array([-1, 0, 1])]))
        else:
            nu = np.sort(nu)

        # Construct basis matrix
        basis_matrix = basis.construct_func_eval_matrix(self.basis_nu, nu,
                                                        self.nu_basis_type, epsilon=self.nu_epsilon,
                                                        order=order, zga_params=None)

        x = self.get_dop_params(x=x)

        if delta_density:
            dnu = self.get_nu_basis_spacing()
            if self.nu_basis_type == 'delta':
                x = x / dnu

        dop = basis_matrix @ x


        # TODO: revisit normalization
        normalize_by = self.get_dop_norm(nu, normalize, normalize_tau, normalize_quantiles)

        # print(normalize_by)

        dop /= normalize_by
        
        # Add pure inductance, resistance, and capacitance
        if include_ideal:
            ohmic_index = np.where(nu == 0)
            if len(ohmic_index) == 1:
                r_inf = self.fit_parameters['R_inf']
                if delta_density:
                    r_inf = r_inf / dnu[utils.array.nearest_index(self.basis_nu, 0)]
                if normalize:
                    # Since ideal elements are delta functions, they should not be 
                    # scaled by the non-ideal basis function area
                    r_inf = r_inf / (normalize_by[ohmic_index] * self.nu_basis_area)
                    # Since ideal elements are delta functions, they should not be 
                    # scaled by the non-ideal basis function area
                dop[ohmic_index] += r_inf
    
    
            induc_index = np.where(nu == 1)
            if len(induc_index) == 1:
                induc = self.fit_parameters['inductance']
                if delta_density:
                    induc = induc / dnu[utils.array.nearest_index(self.basis_nu, 1)]
                if normalize:
                    # Since ideal elements are delta functions, they should not be 
                    # scaled by the non-ideal basis function area
                    induc = induc / (normalize_by[induc_index] * self.nu_basis_area)
                dop[induc_index] += induc
    
            cap_index = np.where(nu == -1)
            if len(cap_index) == 1:
                c_inv = self.fit_parameters['C_inv']
                if delta_density:
                    c_inv = c_inv / dnu[utils.array.nearest_index(self.basis_nu, -1)]
                if normalize:
                    # Since ideal elements are delta functions, they should not be 
                    # scaled by the non-ideal basis function area
                    c_inv = c_inv / (normalize_by[cap_index] * self.nu_basis_area)
                dop[cap_index] += c_inv

        if return_nu:
            return nu, dop
        else:
            return dop
        
    def get_dop_norm(self, nu, normalize: bool = False, normalize_tau: Optional[tuple] = None, 
                     normalize_quantiles: tuple = (0, 1)):
        if normalize:
            if normalize_tau is None:
                data_tau_lim = pp.get_tau_lim(self.get_fit_frequencies(), self.get_fit_times(), self.step_times)
                normalize_tau = np.array(data_tau_lim)
            
            normalize_by = phasance.phasor_scale_vector(nu, normalize_tau, normalize_quantiles)
            normalize_by /= self.nu_basis_area
        else:
            normalize_by = 1
            
        return normalize_by

    def get_tau_dd_eval(self, ppd=20):
        """
        Get diffusion time grid for DDT evaluation and plotting.
        """
        if self.basis_tau_dd is None:
            raise ValueError('basis_tau_dd must be set before predicting the DDT')

        log_tau_min = np.min(np.log10(self.basis_tau_dd)) - 1
        log_tau_max = np.max(np.log10(self.basis_tau_dd)) + 1
        tau = np.logspace(log_tau_min, log_tau_max, int((log_tau_max - log_tau_min) * ppd) + 1)

        return tau

    def get_ddt_norm(self, normalize: bool = False, normalize_by: Optional[float] = None,
                     x: Optional[ndarray] = None, abs_norm: bool = False):
        """
        Get normalization for the DDT distribution.

        When normalize is True, the distribution is scaled such that
        int q(tau_dd, p) dln(tau_dd) dp = 1 (in 2D mode) or
        int q(tau_dd) dln(tau_dd) = 1 (in 1D mode).
        """
        if normalize_by is not None:
            normalize = True

        if normalize:
            if normalize_by is None:
                if x is None:
                    x = self.get_ddt_params()
                ddt_area = ddt.get_ddt_basis_area(self.tau_dd_basis_type, self.tau_dd_epsilon,
                                                  self.basis_tau_dd, p_grid=self.basis_p,
                                                  p_epsilon=self.ddt_p_epsilon)
                if abs_norm:
                    normalize_by = np.sum(np.abs(x)) * ddt_area
                else:
                    normalize_by = np.sum(x) * ddt_area
        else:
            normalize_by = 1

        return normalize_by

    def predict_ddt(self, tau_dd=None, ppd=20, x=None, order=0, normalize=False,
                    normalize_by: Optional[float] = None, abs_norm: bool = False,
                    return_tau_dd: bool = False, marginalize=None, return_p=False):
        """
        Predict the distribution of diffusion times as a function of tau_dd
        (and, in 2D mode, of the anomalous exponent p).

        :param ndarray tau_dd: diffusion times at which to evaluate the distribution
        :param int ppd: points per decade for default evaluation grid
        :param x: DDT parameters (either a dict of fit parameters, a raw QPHB
            parameter vector, or None to use the stored fit parameters)
        :param int order: derivative order of the basis functions to evaluate
        :param bool normalize: if True, normalize so that int q dln(tau_dd) = 1
        :param bool return_tau_dd: if True, also return the tau_dd evaluation grid
        :param str marginalize: in 2D mode, 'tau' returns the p-marginal
            q(tau_dd) and 'p' returns the tau-marginal q(p); None returns the
            full 2D distribution q(tau_dd, p)
        :param bool return_p: if True (2D mode), also return the p grid
        :return: array of distribution density values
        """
        if tau_dd is None:
            tau_dd = self.get_tau_dd_eval(ppd)

        x = self.get_ddt_params(x)

        normalize_by = self.get_ddt_norm(normalize, normalize_by, x=x, abs_norm=abs_norm)

        if self.basis_p is None:
            # 1D mode
            if order == 0 and self.tau_dd_basis_type == 'delta':
                # The delta basis is only defined at the basis points. Interpolate
                # the coefficients in log-tau_dd space so the returned distribution
                # is continuous (e.g. for plotting and integration).
                dist = np.interp(np.log(tau_dd), np.log(self.basis_tau_dd), x) / normalize_by
            else:
                basis_matrix = basis.construct_func_eval_matrix(np.log(self.basis_tau_dd), np.log(tau_dd),
                                                                self.tau_dd_basis_type, epsilon=self.tau_dd_epsilon,
                                                                order=order)
                dist = basis_matrix @ x / normalize_by

            if return_tau_dd:
                return tau_dd, dist
            else:
                return dist

        # 2D mode
        n_tau = len(self.basis_tau_dd)
        n_p = len(self.basis_p)
        X = x.reshape(n_tau, n_p)  # columns indexed by p

        if order == 0 and self.tau_dd_basis_type == 'delta':
            # Delta basis: interpolate each p-slice in log-tau_dd space
            dist = np.empty((len(tau_dd), n_p))
            for j in range(n_p):
                dist[:, j] = np.interp(np.log(tau_dd), np.log(self.basis_tau_dd), X[:, j])
        else:
            # 2D Gaussian basis: separable evaluation dist = B_tau X B_p^T
            B_tau = basis.construct_func_eval_matrix(np.log(self.basis_tau_dd), np.log(tau_dd),
                                                     self.tau_dd_basis_type, epsilon=self.tau_dd_epsilon,
                                                     order=order)
            B_p = basis.construct_func_eval_matrix(self.basis_p, self.basis_p,
                                                   'gaussian', epsilon=self.ddt_p_epsilon,
                                                   order=order)
            dist = B_tau @ X @ B_p.T
        dist /= normalize_by

        if marginalize == 'tau':
            dist = dist.sum(axis=1) * np.mean(np.diff(self.basis_p))
            if return_tau_dd:
                return tau_dd, dist
            return dist
        elif marginalize == 'p':
            dist = dist.sum(axis=0) * np.mean(np.diff(np.log(tau_dd)))
            if return_p:
                return self.basis_p, dist
            return dist

        if return_p:
            if return_tau_dd:
                return tau_dd, self.basis_p, dist
            return self.basis_p, dist
        if return_tau_dd:
            return tau_dd, dist
        return dist

    def estimate_ddt_cov(self, tau_dd=None, ppd=20, p_matrix=None, normalize=False,
                         var_floor=0.0, order=0):
        """
        Estimate the covariance of the DDT distribution.

        Only implemented for 1D mode (p = 0.5); returns None in 2D mode.
        """
        if self.basis_p is not None:
            return None

        if tau_dd is None:
            tau_dd = self.get_tau_dd_eval(ppd)

        basis_matrix = basis.construct_func_eval_matrix(np.log(self.basis_tau_dd), np.log(tau_dd),
                                                        self.tau_dd_basis_type, epsilon=self.tau_dd_epsilon,
                                                        order=order)

        normalize_by = self.get_ddt_norm(normalize)
        normalize_by **= 2

        x_cov = self.estimate_param_cov(p_matrix)

        if x_cov is not None:
            ddt_start, ddt_end = self.ddt_indices
            x_cov = x_cov[ddt_start:ddt_end, ddt_start:ddt_end]
            dist_cov = basis_matrix @ x_cov @ basis_matrix.T
            dist_cov = dist_cov / normalize_by

            if var_floor > 0:
                dist_var = np.diag(dist_cov).copy()
                dist_var[dist_var < var_floor] = var_floor
                np.fill_diagonal(dist_cov, dist_var)

            return dist_cov
        else:
            return None

    def predict_ddt_ci(self, tau_dd=None, ppd=20, x=None, quantiles=[0.025, 0.975],
                       normalize=False, order=0):
        """
        Predict credible intervals of the DDT distribution.

        Only implemented for 1D mode (p = 0.5); returns None in 2D mode.
        """
        if self.basis_p is not None:
            return None

        dist_cov = self.estimate_ddt_cov(tau_dd=tau_dd, ppd=ppd, normalize=normalize, order=order)

        if dist_cov is None:
            return None

        if tau_dd is None:
            tau_dd = self.get_tau_dd_eval(ppd)

        dist_mu = self.predict_ddt(tau_dd=tau_dd, ppd=ppd, x=x, normalize=normalize, order=order)
        dist_std = np.sqrt(np.diag(dist_cov))

        from scipy.stats import norm
        dist_lo = dist_mu + norm.ppf(quantiles[0]) * dist_std
        dist_hi = dist_mu + norm.ppf(quantiles[1]) * dist_std

        return dist_lo, dist_hi

    def integrate_ddt(self, tau_dd_min, tau_dd_max, ppd=10, **predict_kw):
        """
        Integrate the DDT distribution over the given diffusion time range.

        In 2D mode the result is the integral over log(tau_dd) of the p-marginal
        q(tau_dd).
        """
        num_decades = np.log10(tau_dd_max) - np.log10(tau_dd_min)
        tau_dd = np.logspace(np.log10(tau_dd_min), np.log10(tau_dd_max), int(num_decades * ppd) + 1)
        predict_kw['marginalize'] = 'tau'
        q = self.predict_ddt(tau_dd, **predict_kw)
        return np.trapezoid(q, x=np.log(tau_dd))

    def predict_response(self, times=None, input_signal=None, step_times=None, step_sizes=None, op_mode=None,
                         offset_steps=None, step_offset_size=None,
                         include_dop=True, include_drt=True,
                        include_ohmic=True, include_cap=True,
                         smooth_inf_response=None, x=None, include_vz_offset=True, subtract_background=True,
                         y_bkg=None, v_baseline=None):
        # If chrono_mode is not provided, use fitted chrono_mode
        if op_mode is None:
            op_mode = self.chrono_mode
        utils.validation.check_ctrl_mode(op_mode)

        # If times is not provided, use self.t_fit
        if times is None:
            use_fit_times = True
            times = self.get_fit_times()
        else:
            use_fit_times = False

        # If kwargs not provided, use same values used in fitting
        if offset_steps is None:
            offset_steps = self.fit_kwargs['offset_steps']
        if step_offset_size is None:
            step_offset_size = self.fit_kwargs["step_offset_size"]
        if smooth_inf_response is None:
            smooth_inf_response = self.fit_kwargs['smooth_inf_response']

        # Get prediction matrix and vectors
        rm_drt, induc_rv, inf_rv, cap_rv, rm_dop = self._prep_chrono_prediction_matrix(times, input_signal,
                                                                               step_times, step_sizes, op_mode,
                                                                               offset_steps, step_offset_size, smooth_inf_response)
        # Response matrices from _prep_response_prediction_matrix will be scaled. Rescale to data scale
        # rm *= self.input_signal_scale
        # induc_rv *= self.input_signal_scale
        # inf_rv *= self.input_signal_scale

        # Get parameters
        if x is not None:
            if isinstance(x, dict):
                fit_parameters = x
            else:
                fit_parameters = self.extract_qphb_parameters(x)
        else:
            fit_parameters = self.fit_parameters

        x_drt = fit_parameters['x']
        x_dop = fit_parameters.get('x_dop', None)
        r_inf = fit_parameters.get('R_inf', 0)
        induc = fit_parameters.get('inductance', 0)
        c_inv = fit_parameters.get('C_inv', 0)

        response = np.zeros(len(times))
        
        if v_baseline is None:
            # if use_fit_times:
            # Only include the baseline if the prediction times match the fitted times
            v_baseline = self.predict_v_baseline(times)
            # else:
            #     v_baseline = 0
            
        if include_drt:
            response += rm_drt @ x_drt

        if include_ohmic:
            response += inf_rv * r_inf
        
        if include_cap:
            response += c_inv * cap_rv

        if x_dop is not None and include_dop:
            response += rm_dop @ x_dop

        # response = rm_drt @ x_drt + inf_rv * r_inf + induc * induc_rv + c_inv * cap_rv

        # if x_dop is not None:
        #     response += rm_dop @ x_dop

        # if not subtract_background:
        #     if not np.array_equal(times, self.get_fit_times()):
        #         raise ValueError('Background can only be included if prediction times are same as fit times')
        #     response += self.raw_response_background

        if include_vz_offset:
            # # Need to back out the offset to get the fit parameter, apply offset to fit parameter by offset
            # v_baseline_param = (v_baseline + self.scaled_response_offset * self.response_signal_scale)
            # v_baseline_offset = v_baseline_param * (1 + self.fit_parameters.get('vz_offset', 0)) \
            #                     - self.scaled_response_offset * self.response_signal_scale
            # Apply vz_offset before adding baseline
            vz_strength_vec, _ = self._get_vz_strength_vec(
                times, vz_offset_eps=self.fit_parameters.get('vz_offset_eps', None)
            )
            response *= (1 + fit_parameters.get('vz_offset', 0) * vz_strength_vec)

        response += v_baseline

        if not subtract_background:
            if y_bkg is None:
                y_bkg = self.predict_chrono_background(times)
            if len(times) != len(y_bkg):
                raise ValueError('Length of background does not match length of times')
            response += y_bkg

        return response
    
    def predict_v_baseline(self, times, x_vb: Optional[ndarray] = None):
        if "v_baseline" in self.fit_parameters.keys():
            vb_mat = background.get_baseline_matrix(times, self.v_baseline_deg, normalize=False, sqrt=self.v_baseline_sqrt)
            if x_vb is None:
                x_vb = self.fit_parameters["v_baseline"]
            return vb_mat @ x_vb
        else:
            return np.zeros_like(times)
        

    def predict_chrono_background(self, times):
        if self.background_gp is None:
            return np.zeros(len(times))
            # raise ValueError('Chrono background was not fitted. To fit the chrono background, '
            #                  'call fit_chrono or fit_hybrid with subtract_background=True')

        if np.array_equal(times, self.get_fit_times()):
            # If times are fit times, use precalculated background
            return self.raw_response_background
        else:
            # Predict using fitted GP
            if self.fit_kwargs['background_type'] == 'static':
                return self.background_gp.predict(times[:, None])
            else:
                # Need to account for correlation between DRT and background
                y_pred = self.predict_response(times)
                rm_bkg = background.get_background_matrix([self.background_gp], times[:, None],
                                                          y_drt=y_pred,
                                                          corr_power=self.fit_kwargs['background_corr_power'])
                # Get residuals for training data
                y_resid = self.raw_response_signal - self.predict_response()

                return rm_bkg @ y_resid

    def predict_z(self, frequencies, include_vz_offset=True, x=None,
                  include_dop=True, include_ddt=True, include_drt=True, include_inductance=True,
                  include_ohmic=True, include_cap=True
                  ):
        # Get matrix
        zm, zm_dop, zm_ddt = self._prep_impedance_prediction_matrix(frequencies)

        # Get parameters
        if x is not None:
            if isinstance(x, dict):
                fit_parameters = x
            else:
                fit_parameters = self.extract_qphb_parameters(x)
        else:
            fit_parameters = self.fit_parameters

        x_drt = fit_parameters['x']
        x_dop = fit_parameters.get('x_dop', None)
        x_ddt = fit_parameters.get('x_ddt', None)
        r_inf = fit_parameters.get('R_inf', 0)
        induc = fit_parameters.get('inductance', 0)
        c_inv = fit_parameters.get('C_inv', 0)

        z = np.zeros(len(frequencies), dtype=complex)

        if include_drt:
            z += zm @ x_drt

        if include_ohmic:
            z += r_inf
        if include_inductance:
            z += induc * 2j * np.pi * frequencies
        if include_cap:
            z+= c_inv * (2j * np.pi * frequencies) ** -1

        if x_dop is not None and include_dop:
            z += zm_dop @ x_dop

        if x_ddt is not None and include_ddt:
            z += zm_ddt @ x_ddt

        if include_vz_offset:
            _, vz_strength_vec = self._get_vz_strength_vec(None, frequencies,
                                                           vz_offset_eps=self.fit_parameters.get('vz_offset_eps', None))
            z *= (1 - fit_parameters.get('vz_offset', 0) * vz_strength_vec)

        return z

    def predict_sigma(self, measurement):
        if measurement == 'chrono':
            key = 'v_sigma_tot'
        elif measurement == 'eis':
            key = 'z_sigma_tot'

        return self.fit_parameters.get(key, None)

    def predict_r_p(self, sign=None, absolute=False, x=None, raw=False):
        if not self.fit_drt or self.basis_tau is None:
            # Pure DOP / DDT mode: no DRT block, so no relaxation polarization resistance
            return 0.0

        basis_area = basis.get_basis_func_area(self.tau_basis_type, self.tau_epsilon, self.zga_params)

        if sign is None:
            if self.series_neg:
                sign = 0
            else:
                sign = 1

        if raw:
            if len(x) > self.num_drt_params:
                x = x[self.get_qp_mat_offset():]
        else:
            x = self.get_drt_params(x, sign)

        if absolute:
            sum_x = np.sum(np.abs(x))
        else:
            sum_x = np.sum(x)
        return sum_x * basis_area

    def predict_r_inf(self):
        r_inf = self.fit_parameters.get('R_inf', 0)

        if self.fit_dop and self.nu_basis_type == 'delta':
            zero_index = np.where(self.basis_nu == 0)
            if len(zero_index) == 1:
                r_inf += np.sum(self.fit_parameters['x_dop'][zero_index])

        return r_inf

    def predict_r_tot(self):
        return self.predict_r_inf() + self.predict_r_p()

    def integrate_distribution(self, tau_min, tau_max, ppd=10, **predict_kw):
        warnings.warn("integrate_distribution is deprecated. Please use integrate_drt instead", DeprecationWarning)
        return self.integrate_drt(tau_min, tau_max, ppd=ppd, **predict_kw)
    
    def integrate_drt(self, tau_min, tau_max, ppd=10, **predict_kw):
        num_decades = np.log10(tau_max) - np.log10(tau_min)
        tau = np.logspace(np.log10(tau_min), np.log10(tau_max), int(num_decades * ppd) + 1)
        gamma = self.predict_drt(tau, **predict_kw)
        return np.trapezoid(gamma, x=np.log(tau))

    def split_r_p(self, tau_splits, **predict_kw):
        tau_splits = sorted(tau_splits)
        if 'tau' not in predict_kw.keys():
            ppd = predict_kw.get('ppd', 20)
            tau = self.get_tau_eval(ppd)
        else:
            tau = predict_kw.pop('tau')

        gamma = self.predict_drt(tau, **predict_kw)

        split_index = [utils.array.nearest_index(tau, ts) for ts in tau_splits]
        start_index = np.array([0] + split_index)
        end_index = np.array(split_index + [len(tau)]) + 1

        split_area = np.array([np.trapezoid(gamma[i:j], x=np.log(tau[i:j])) for i, j in zip(start_index, end_index)])

        return split_area

    # def evaluate_residuals(self):
    #     # Get fitted response
    #     y_hat = self.predict_response(**self.fit_kwargs)
    #
    #     return y_hat - self.raw_response_signal

    # ----------------------------------------------------
    # Fit quantification
    # ----------------------------------------------------
    def evaluate_chi_sq(self, frequencies=None, z=None, x=None, weights=None, **predict_kw):
        # Get fit frequencies
        if frequencies is None:
            frequencies = self.get_fit_frequencies()

        if z is None:
            z = self.z_fit

        # Get weights
        if weights is not None:
            if weights == 'modulus':
                weights = 1 / np.abs(z)
            elif np.shape(weights) != np.shape(z):
                raise ValueError('Weights must have same shape as frequencies')

        # Get model impedance
        z_hat = self.predict_z(frequencies, x=x, **predict_kw)

        # Calculate residuals
        return np.sum((weights * np.abs(z - z_hat)) ** 2)

    # ----------------------------------------------------
    # Peak finding
    # ----------------------------------------------------
    def find_peaks(self, 
        tau: Optional[ndarray] = None, 
        x: Optional[ndarray] = None, 
        normalize: bool = True, 
        ppd: int = 10, 
        prominence: Optional[float] = None, 
        height: Optional[float] = None, 
        sign: int = 1,
        return_info: bool = False,
        method: str = 'thresh', 
        prob_thresh: float = 0.25, 
        p_matrix: Optional[ndarray] = None, 
        fxx_var_floor: float = 1e-5, 
        extend_var: bool = True,
        num_peaks: int = None,
        **kw):
        """Find peaks in the DRT.

        :param Optional[ndarray] tau: tau grid for peak finding. 
            If not provided, defaults to a grid with 10 points per 
            decade over the basis_tau range with 1 decade of extension 
            at the boundaries.
        :param Optional[ndarray] x: Array of fit parameters. By default, 
            uses the fitted parameters of this DRT instance.
        :param bool normalize: If True, normalize the DRT to 
            polarization resistance for consistent results. 
            Defaults to True.
        :param int ppd: Points per decade for tau grid. Ignored if tau 
            is provided. Defaults to 10.
        :param Optional[float] prominence: Minimum prominence of 
            detected peaks (see scipy.signal.find_peaks). 
            By default, will be automatically determined.
        :param Optional[float] height: Minimum height of 
            detected peaks (see scipy.signal.find_peaks). 
            By default, will be automatically determined
        :param int sign: If 1, find positive and/or negative peaks in 
            the net distribution. If 0, find peaks in the positive
            and negative components separately. Defaults to 1.
        :param bool return_info: If True, return extra information 
            (see return parameters). Defaults to False.
        :param str method: Peak-finding method. If 'thresh', find all 
            peaks in the curvature of the DRT that exceed the height and 
            prominence thresholds. If 'prob', a very weak threshold
            is applied during initial peak finding, but the peaks are 
            then filtered based on their apparent plausibility using
            the credible interval of the DRT. Defaults to 'thresh'.
        :param float prob_thresh: Probability threshold for 'prob' 
            method. Defaults to 0.25.
        :param Optional[ndarray] p_matrix: Custom precision matrix for 
            credible interval calculation (only used for method='prob'). 
            By default, use the fitted precision matrix from this DRT 
            instance.
        :param float fxx_var_floor: Minimum curvature variance for 
            'prob' method. Defaults to 1e-5.
        :param bool extend_var: If True, extend the curvature variance 
            at the boundaries for stability (only for method='prob'). 
            Defaults to True.
        :param int num_peaks: Number of peaks to find if method == 'prob'.
            If provided, prob_thresh is ignored.
        :return ndarray peak_tau: Detected peak locations (tau)
        :return ndarray tau: Tau grid used  for peak finding. 
            Only returned if return_info=True.
        :return ndarray peak_indices: Array of peak indices.
            Only returned if return_info=True.
        :return tuple peak_info: Peak information from scipy.signal.find_peaks.
            Only returned if return_info=True.
        """
        method_options = ['thresh', 'prob']
        if method not in method_options:
            raise ValueError(f'Invalid method {method}. Options: {method_options}')

        # If tau is not provided, go one decade beyond basis_tau with standard spacing
        # Finer spacing can cause many minor peaks to appear in 2nd derivative
        if tau is None:
            tau = self.get_tau_eval(ppd)

        # fx = self.predict_distribution(tau=tau, x=x, order=1)
        fxx = self.predict_drt(tau=tau, x=x, order=2, sign=sign, normalize=normalize)

        # if normalize:
        # fx = fx / self.predict_r_p()
        # fxx = fxx / self.predict_r_p()

        # peak_indices = peaks.find_peaks(fx, fxx, fx_kw, fxx_kw)
        # if fxx_kw is None:
        #     fxx_kw = {}

        # f_mix = 0.5 * (f - fxx)

        # if prominence is None:
        #     prominence = -np.min(fxx) * 2.5e-2
        #     print('prom:', prominence)

        if prominence is None:
            # prominence = np.median(np.abs(fxx)) + 0.05 * np.percentile(-fxx[fxx > -np.inf], 95)
            if method == 'thresh':
                prominence = 0.05 * np.std(fxx[~np.isinf(fxx)]) + 5e-3
            else:
                prominence = 5e-3
            # print(prominence)

        if height is None:
            if method == 'thresh':
                height = 0
            else:
                height = 1e-3

        # if self.series_neg:
        #     # Find positive and negative peaks separately
        #     peak_index_list = []
        #     peak_info_list = []
        #     for sign in [-1, 1]:
        #         f = self.predict_distribution(tau=tau, x=x, order=0, sign=sign, normalize=normalize)
        #         fxx = self.predict_distribution(tau=tau, x=x, order=2, sign=sign, normalize=normalize)
        #         peak_index, peak_info = signal.find_peaks(-sign * fxx, height=height, prominence=prominence, **kw)
        #         # Limit to peaks that are positive in the direction of the current sign
        #         pos_index = sign * f[peak_index] > 0
        #         # print(pos_index)
        #         peak_index = peak_index[pos_index]
        #         peak_info = {k: v[pos_index] for k, v in peak_info.items()}
        #
        #         peak_index_list.append(peak_index)
        #         peak_info_list.append(peak_info)
        #     # Concatenate pos and neg peaks
        #     peak_indices = np.concatenate(peak_index_list)
        #     peak_info = {k: np.concatenate([pi[k] for pi in peak_info_list]) for k in peak_info.keys()}
        #
        #     # Sort ascending
        #     sort_index = np.argsort(peak_indices)
        #     peak_indices = peak_indices[sort_index]
        #     peak_info = {k: v[sort_index] for k, v in peak_info.items()}
        # else:

        if self.fit_kwargs['nonneg'] and sign != 0:
            # This captures both the standard nonneg case and the series_neg case
            # peak_indices = peaks.find_peaks_simple(fxx, order=2, height=0, prominence=prominence, **kw)
            peak_indices, peak_info = signal.find_peaks(-sign * fxx, height=height, prominence=prominence, **kw)
            # peak_indices = peaks.find_peaks_compound(fx, fxx, **kw)
        else:
            f = self.predict_drt(tau=tau, x=x, order=0, sign=sign, normalize=normalize)
            # Find positive and negative peaks separately
            peak_index_list = []
            peak_info_list = []
            for peak_sign in [-1, 1]:
                peak_index, peak_info = signal.find_peaks(-peak_sign * fxx, height=height, prominence=prominence, **kw)
                # Limit to peaks that are positive in the direction of the current sign
                pos_index = peak_sign * f[peak_index] > 0
                # print(pos_index)
                peak_index = peak_index[pos_index]
                peak_info = {k: v[pos_index] for k, v in peak_info.items()}

                peak_index_list.append(peak_index)
                peak_info_list.append(peak_info)
            # Concatenate pos and neg peaks
            peak_indices = np.concatenate(peak_index_list)
            peak_info = {k: np.concatenate([pi[k] for pi in peak_info_list]) for k in peak_info.keys()}

            # Sort ascending
            sort_index = np.argsort(peak_indices)
            peak_indices = peak_indices[sort_index]
            peak_info = {k: v[sort_index] for k, v in peak_info.items()}

        if method == 'prob':
            # Use prominence or height, whichever is smaller
            min_prom = np.minimum(peak_info['prominences'], peak_info['peak_heights'])

            # Get curvature std
            fxx_cov = self.estimate_distribution_cov(tau, p_matrix=p_matrix, order=2, sign=sign, normalize=normalize,
                                                     var_floor=fxx_var_floor,
                                                     extend_var=extend_var
                                                     )
            fxx_sigma = np.diag(fxx_cov) ** 0.5

            # Evaluate peak confidence
            peak_prob = 1 - 2 * stats.cdf_normal(0, min_prom, fxx_sigma[peak_indices])
            # print(peak_prob)
            
            if num_peaks is not None:
                prob_sort = np.sort(peak_prob)[::-1]
                prob_thresh = prob_sort[min(num_peaks - 1, len(peak_prob) - 1)]
            
            # TODO: incorporate probability that peak height is greater than zero
            # Keep peaks that exceed probability threshold
            peak_indices = peak_indices[peak_prob >= prob_thresh]

            # peak_prob = 1 - 2 * stats.cdf_normal(0, min_prom, 1)
            
            # Add prob to info dict
            peak_info['probs'] = peak_prob
            

        if return_info:
            return tau[peak_indices], tau, peak_indices, peak_info
        else:
            return tau[peak_indices]

    def mark_peaks(self, ax, x=None, sign=1, peak_tau=None, find_peaks_kw=None, scale_prefix=None, area=None,
                   normalize=False, normalize_by: Optional[float] = None, y_offset: float = 0.0,
                   **plot_kw):
        if find_peaks_kw is None:
            find_peaks_kw = {}

        # Find peaks
        if peak_tau is None:
            peak_tau = self.find_peaks(x=x, sign=sign, **find_peaks_kw)

        gamma_peaks = self.predict_drt(peak_tau, 
                                                normalize=normalize, normalize_by=normalize_by, 
                                                x=x, sign=sign)

        if area is not None:
            gamma_peaks *= area

        if scale_prefix is None:
            scale_prefix = utils.scale.get_scale_prefix(gamma_peaks)
        scale_factor = utils.scale.get_factor_from_prefix(scale_prefix)

        ax.scatter(peak_tau, gamma_peaks / scale_factor + y_offset, **plot_kw)

    def estimate_param_cov(self, p_matrix=None):
        """
        Estimate parameter covariance matrix
        :return: covariance matrix
        """
        if p_matrix is None:
            p_matrix = self.fit_parameters.get('p_matrix', None)

        if p_matrix is not None:
            try:
                p_inv = np.linalg.inv(p_matrix)
                # Invert DOP scaling
                if self.fit_dop:
                    dop_start, dop_end = self.dop_indices
                    dop_scale_mat = np.tile(self.dop_scale_vector, (len(p_inv), 1))
                    p_inv[:, dop_start:dop_end] *= dop_scale_mat
                    p_inv[dop_start:dop_end, :] *= dop_scale_mat.T
                return p_inv * self.coefficient_scale ** 2
            except np.linalg.LinAlgError:
                warnings.warn('Singular P matrix - could not obtain covariance estimate')
                return None
        else:
            raise Exception('Parameter covariance estimation is only available for qphb fits')
        
    def fisher_matrix(self, weighted: bool = True):
        rm = self.qphb_params['rm']
        if weighted:
            w = self.qphb_params['weights']
            rm = np.diag(w) @ rm
        return rm.T @ rm

    def generate_map_samples(self, max_iter=2, shift_frac=0.05, shift_scale=1.5, random_seed=None):
        # Generate samples if no existing samples OR sampling arguments have changed
        # kwargs = {'max_iter': max_iter, 'x_interval': x_interval, 'delta_vals': delta_vals}
        kwargs = {'max_iter': max_iter, 'shift_frac': shift_frac,
                  'shift_scale': shift_scale, 'random_seed': random_seed}
        if self.map_samples is None or not utils.array.check_equality(kwargs, self.map_sample_kw):

            # Generate MAP samples
            # Estimate covariance matrix for coefficients
            cov = self.estimate_param_cov()
            sigma_x = np.diag(cov) ** 0.5
            sigma_x /= self.coefficient_scale

            # Get estimated coefficients from regular QPHB
            x_hat = self.qphb_history[-1]['x']

            # Get args for optimization func
            arg_list = inspect.getfullargspec(qphb.optimize_lp_semi_fixed).args

            # Set kwargs for optimize func
            kwargs = dict(fixed_prior=True, special_qp_params=self.special_qp_params,
                          xtol=1e-3, max_iter=max_iter, history=None)
            kwargs.update(self.fit_kwargs)
            kwargs.update(self.qphb_params)

            # Get ordered arg values
            arg_vals = [kwargs.get(arg, None) for arg in arg_list[2:]]

            # Set floor for parameter values
            if self.fit_kwargs['nonneg']:
                x_floor = np.zeros(len(x_hat))
                for v in self.special_qp_params.values():
                    if not v['nonneg']:
                        x_floor[v['index']] = -np.inf
            else:
                x_floor = np.ones(len(x_hat)) * -np.inf

            # Perform initial optimization with fixed prior and R_inf fixed - ensures that lp_hat is correct
            # (sometimes lp increases significantly with 1-2 iterations of optimize_lp_semi_fixed)
            fixed_x_index = [self.special_qp_params['R_inf']['index']]
            x_in = x_hat.copy()
            x_init, lp_hat = qphb.optimize_lp_semi_fixed(x_in, fixed_x_index, *arg_vals)
            # print('delta x:', x_init - x_hat)
            print('delta lp:', lp_hat - self.qphb_params['posterior_lp'])
            x_hat = x_init

            # Determine number of parameters to shift in each sample
            shift_num = int(len(x_hat) * shift_frac)
            print('shift_num:', shift_num)

            # Set up random state
            rng = np.random.default_rng(seed=random_seed)

            # lp_array = np.empty(num_samples)
            # x_array = np.empty((num_samples, len(x_hat)))

            all_indices = np.arange(0, len(x_hat), dtype=int)
            possible_indices = all_indices.copy()

            # Define selection probability to disfavor selection of adjacent indices
            prob_func = basis.get_basis_func('gaussian')
            prob_eps = 0.25 * self.tau_epsilon
            prob_tau = np.concatenate(
                (
                    np.ones(self.get_qp_mat_offset()) * self.basis_tau[0] * 1e-2,
                    self.basis_tau
                )
            )

            def rng_prob(index):
                prob = 1 - prob_func(np.log(prob_tau / prob_tau[index]), prob_eps)
                return prob

            # index_sep = 10

            print('Sampling...')
            # for i in range(num_samples):
            #     # print(i)
            #     if len(possible_indices) == 0:
            #         possible_indices = all_indices.copy()
            #     # Select parameters to shift
            #     primary_index = rng.choice(possible_indices, 1, replace=False)
            #     possible_indices = np.delete(possible_indices, np.where(possible_indices == primary_index))
            #     select_prob = rng_prob(primary_index)
            #     select_prob /= np.sum(select_prob)
            #     # print(select_prob)
            #
            #     shift_index = [primary_index[0]]
            #     while len(shift_index) < shift_num:
            #         new_index = rng.choice(all_indices, 1, replace=False, p=select_prob)
            #         shift_index.append(new_index[0])
            #         # print(new_index, shift_index)
            #         select_prob *= rng_prob(new_index)
            #         select_prob /= np.sum(select_prob)
            #         # print(select_prob)
            #
            #     shift_index = np.array(sorted(shift_index))
            #     # print(shift_index)
            #
            #     # Get shift sizes
            #     shift_sigma = sigma_x[shift_index]
            #     delta_x = rng.normal(0, shift_sigma * shift_scale)
            #
            #     #
            #     x_in = x_hat.copy()
            #     x_in[shift_index] += delta_x
            #     x_in = np.maximum(x_in, x_floor)
            #
            #
            #     # Optimize with shifted parameters fixed in place
            #     x_array[i], lp_array[i] = qphb.optimize_lp_semi_fixed(x_in, shift_index, *arg_vals)
            #
            #     print(shift_index, delta_x, lp_array[i])

            #     if i == int(num_samples / 4):
            #         print('Sampling 25% complete')
            #     elif i == int(num_samples / 2):
            #         print('Sampling 50% complete')
            #     elif i == int(num_samples * 0.75):
            #         print('Sampling 75% complete')

            lp_array = []
            x_array = []

            for primary_index in range(self.get_qp_mat_offset(), len(x_hat)):
                # # Add a copy of the optimal solution rather than sampling 0
                # lp_array.append(lp_hat)
                # x_array.append(x_hat)

                for delta in [-3, -2, -1, 1, 2, 3]:
                    primary_delta = delta * sigma_x[primary_index] * shift_scale
                    # Check if shift is within bounds; if not, skip
                    if x_hat[primary_index] >= x_floor[primary_index]:
                        shift_index = []
                        if shift_num > 0:
                            # Select additional indices to shift
                            select_prob = rng_prob(primary_index)
                            select_prob /= np.sum(select_prob)
                            while len(shift_index) < shift_num:
                                new_index = rng.choice(all_indices, 1, replace=False, p=select_prob)
                                shift_index.append(new_index[0])
                                # print(new_index, shift_index)
                                select_prob *= rng_prob(new_index)
                                select_prob /= np.sum(select_prob)
                                # print(select_prob)

                            # shift_index = np.array(sorted(shift_index))
                            # print(shift_index)

                            # Get shift sizes
                            shift_sigma = sigma_x[shift_index]
                            delta_x = rng.normal(0, shift_sigma)

                            #
                            x_in = x_hat.copy()
                            x_in[shift_index] += delta_x

                        # Shift the primary index
                        x_in[primary_index] += primary_delta
                        x_in = np.maximum(x_in, x_floor)

                        # Optimize with shifted parameters fixed in place
                        shift_index.append(primary_index)
                        shift_index = sorted(np.array(shift_index))
                        x_out, lp_out = qphb.optimize_lp_semi_fixed(x_in, shift_index, *arg_vals)

                        x_array.append(x_out)
                        lp_array.append(lp_out)

                if primary_index == int(len(x_hat) / 4):
                    print('Sampling 25% complete')
                elif primary_index == int(len(x_hat) / 2):
                    print('Sampling 50% complete')
                elif primary_index == int(len(x_hat) * 0.75):
                    print('Sampling 75% complete')

            print('Sampling 100% complete')

            x_array = np.array(x_array)
            lp_array = np.array(lp_array)

            # delta_array = np.concatenate((-np.array(delta_vals), np.array(delta_vals)))

            # for fixed_x_index in range(self.get_qp_mat_offset(), len(x_hat), x_interval):
            #     if fixed_x_index % 10 == 0:
            #         print(f'Sampling parameter {fixed_x_index}')
            #
            #     lp_out = np.zeros(len(delta_array) + 1)
            #     x_out = np.empty((len(delta_array) + 1, len(x_hat)))
            #
            #     for i, delta in enumerate(delta_array):
            #         # Fix the parameter at delta std devs from its optimal value
            #         x_in = x_hat.copy()
            #         x_in[fixed_x_index] += delta * sigma_x[fixed_x_index]
            #
            #         # Optimize with fixed parameter
            #         x_out[i], lp_out[i] = qphb.optimize_lp_semi_fixed(x_in, fixed_x_index, *arg_vals)
            #
            #     # Insert values for original solution
            #     lp_out[-1] = lp_hat
            #     x_out[-1] = x_hat.copy()
            #
            #     lp_array.append(lp_out)
            #     x_array.append(x_out)

            # Concatenate log-prob values and coef vectors
            # lp_array = np.concatenate(lp_array)
            # x_array = np.vstack(x_array)
            x_array *= self.coefficient_scale

            # Store samples
            self.map_samples = {'lp': lp_array, 'x': x_array}
            # self.map_sample_kw = {'max_iter': max_iter, 'x_interval': x_interval, 'delta_vals': delta_vals}
            self.map_sample_kw = {'max_iter': max_iter, 'shift_frac': shift_frac,
                                  'shift_scale': shift_scale, 'random_seed': random_seed}

    # def estimate_ci(self, method, quantiles=[0.025, 0.975], **sample_kw):
    #     if self.fit_type.find('qphb') > -1:
    #         if method == 'cov':
    #             # Estimate covariance matrix for coefficients
    #             cov = self.estimate_param_cov()
    #             if cov is not None:
    #                 sigma_x = np.diag(cov) ** 0.5
    #
    #                 x_hat = np.empty(len(sigma_x))
    #                 for key, info in self.special_qp_params.items():
    #                     if key == 'dop':
    #                         print('get dop')
    #                         dop_start, dop_end = self.dop_indices
    #                         x_hat[dop_start:dop_end] = self.fit_parameters['x_dop']
    #                     else:
    #                         start = info['index']
    #                         end = start + info.get('size', 1)
    #                         x_hat[start:end] = self.fit_parameters[key]
    #                 x_hat[self.get_qp_mat_offset():] = self.fit_parameters['x']
    #
    #                 # Determine number of std devs to obtain quantiles
    #                 s_quant = stats.std_normal_quantile(quantiles)
    #                 s_lo = s_quant[0]
    #                 s_hi = s_quant[1]
    #
    #                 x_lo = x_hat + s_lo * sigma_x
    #                 x_hi = x_hat + s_hi * sigma_x
    #             else:
    #                 x_lo, x_hi = None, None
    #         elif method == 'map_sample':
    #             # Use QPHB algorithm to sample MAP solutions with different parameters fixed
    #             self.generate_map_samples(**sample_kw)
    #
    #             x_array = self.map_samples['x']
    #             lp_array = self.map_samples['lp']
    #             lp_hat = self.qphb_params['posterior_lp']
    #             x_lo, x_hi = utils.weighted_quantile_2d(x_array, quantiles, np.exp(lp_array - lp_hat), axis=0)
    #
    #     return x_lo, x_hi

    def get_parameter_quantile(self, quantile):
        if self.map_samples is not None:
            x_array = self.map_samples['x']
            lp_array = self.map_samples['lp']
            lp_hat = self.qphb_params['posterior_lp']
            if np.shape(quantile) == ():
                quantile = [quantile]
            x_out = utils.weighted_quantile_2d(x_array, quantile, np.exp(lp_array - lp_hat), axis=0)
            return x_out
        else:
            raise Exception('Map samples must be generated before estimating parameter quantiles')

    def estimate_posterior_mean(self, **sample_kw):
        self.generate_map_samples(**sample_kw)

        x_array = self.map_samples['x']
        lp_array = self.map_samples['lp']
        lp_hat = self.qphb_params['posterior_lp']
        x_mean = np.average(x_array, axis=0, weights=np.exp(lp_array - lp_hat))

        return x_mean

    # def get_lml(self):
    #     """
    #     Calculate log marginal likelihood (evidence) assuming fixed hyperparameters (s, rho)
    #     :return:
    #     """
    #     cov = np.linalg.inv(self.qphb_params['p_matrix'])
    #     det_sign, log_det = np.linalg.slogdet(cov)

    def evaluate_rss(self, weights=None, x=None, normalize=False):
        if weights is None:
            weights = self.qphb_params['est_weights']
        elif weights == "uniform":
            # Uniform weights within each domain
            n_chrono = self.qphb_params["num_chrono"]
            weights = np.empty(len(self.qphb_params["est_weights"]))
            weights[:n_chrono] = np.mean(self.qphb_params["est_weights"][:n_chrono])
            weights[n_chrono:] = np.mean(self.qphb_params["est_weights"][n_chrono:])
        elif np.isscalar(weights):
            weights = np.ones_like(self.qphb_params['est_weights']) * weights
        
            
        if x is None:
            x = self.qphb_history[-1]['x']  # Get parameters in scaled space with special params included

        rss = qphb.evaluate_rss(x, self.qphb_params['rm'], self.qphb_params['rv'], weights)

        # Normalize to number of data points
        if normalize:
            rss /= self.num_data

        return rss

    def evaluate_llh(self, weights=None, x=None, marginalize_weights=True, alpha_0=2, beta_0=1,
                     subtract_background=True, normalize: bool = False):
        if isinstance(weights, ndarray):
            if not np.shape(weights) == self.qphb_params['est_weights'].shape:
                raise ValueError('Expected weights array of shape {}, but received shape {}'.format(self.qphb_params['est_weights'].shape, np.shape(weights)))
        elif weights is None:
            weights = self.qphb_params['est_weights']
        elif weights == "uniform":
            # Uniform weights within each domain
            n_chrono = self.qphb_params["num_chrono"]
            weights = np.empty(len(self.qphb_params["est_weights"]))
            weights[:n_chrono] = np.mean(self.qphb_params["est_weights"][:n_chrono])
            weights[n_chrono:] = np.mean(self.qphb_params["est_weights"][n_chrono:])
        elif np.isscalar(weights):
            weights = np.ones_like(self.qphb_params['est_weights']) * weights

        if x is None:
            x = self.qphb_history[-1]['x']  # Get parameters in scaled space with special params included
            
        if not subtract_background and self.fit_kwargs['subtract_background']:
            # Treat any background corrections as errors.
            # Use the original chrono matrix and vector 
            # instead of the background-corrected instances
            rzm = self.qphb_params['rm'].copy()
            rzm[:self.num_chrono] = self.qphb_params['rm_orig']
            rzv = self.qphb_params['rv'].copy()
            rzv[:self.num_chrono] = self.qphb_params['rv_orig']
        else:
            rzm = self.qphb_params['rm']
            rzv = self.qphb_params['rv']

        llh = qphb.evaluate_llh(x, rzm, rzv, weights,
                                marginalize_weights=marginalize_weights, alpha_0=alpha_0, beta_0=beta_0
                                )
        
        # Normalize to number of data points
        if normalize:
            llh /= self.num_data

        return llh

    def evaluate_bic(self, x=None, **llh_kw):
        # Get log-likelihood
        llh = self.evaluate_llh(x=x, **llh_kw)

        # Determine number of parameters from the number of fitted coefficients
        num_params = self.get_qp_mat_offset() + len(self.basis_tau)

        return stats.bic(num_params, self.num_independent_data, llh)

    def evaluate_lml(self, history_entry=None, weights=None, update_hypers=None):
        # Extract basic info from fit
        penalty_matrices = self.qphb_params['penalty_matrices']
        penalty_type = self.fit_kwargs['penalty_type']
        qphb_hypers = self.qphb_params['hypers'].copy()
        l1_lambda_vector = self.qphb_params['l1_lambda_vector']
        rm = self.qphb_params['rm']
        rv = self.qphb_params['rv']

        # Update qphb_hypers
        if update_hypers is None:
            update_hypers = {}
        qphb_hypers.update(update_hypers)

        # If history entry not provided, use final state from optimization
        if history_entry is None:
            history_entry = self.qphb_history[-1]

        x_hat = history_entry['x']
        rho_vector = history_entry['rho_vector']
        s_vectors = history_entry['s_vectors']
        dop_rho_vector = history_entry['dop_rho_vector']
        ddt_rho_vector = history_entry.get('ddt_rho_vector', None)

        if weights is None:
            weights = self.qphb_params['est_weights']

        lml = qphb.evaluate_lml(x_hat, penalty_matrices, penalty_type, qphb_hypers, 
                                l1_lambda_vector, rho_vector, dop_rho_vector,
                                s_vectors, weights, rm, rv, self.special_qp_params,
                                ddt_rho_vector=ddt_rho_vector)

        return lml

    # Plotting
    # --------------------------------------------
    def get_chrono_transforms(self, include_outliers=True):
        times = self.get_fit_times()

        # Add outlier times
        if include_outliers and self.chrono_outliers is not None:
            t_out = self.chrono_outliers[0]
            times = np.unique(np.concatenate([times, t_out]))

        trans_functions = utils.chrono.get_time_transforms(times, self.nonconsec_step_times)

        return trans_functions

    def plot_chrono_fit(self, ax=None, transform_time=False, linear_time_axis=None,
                        display_linear_ticks=None, linear_tick_kw=None,
                        plot_data=True, data_kw=None, data_label='',
                        scale_prefix=None, x=None, subtract_background=True, y_bkg=None,
                        predict_kw=None, c='k', tight_layout=True, show_outliers=False, outlier_kw=None, **kw):
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))
        else:
            fig = ax.get_figure()

        if display_linear_ticks is None:
            if transform_time and not linear_time_axis:
                display_linear_ticks = True
            else:
                display_linear_ticks = False

        if linear_time_axis is None:
            if transform_time and not display_linear_ticks:
                linear_time_axis = True
            else:
                linear_time_axis = False

        # Set default data plotting kwargs if not provided
        if data_kw is None:
            data_kw = dict(s=10, alpha=0.5)
            
        if predict_kw is None:
            predict_kw = {}

        # Get fit times
        if predict_kw.get("times", None) is None:
            times = self.get_fit_times()
        else:
            times = predict_kw["times"]

        # # Transform time to visualize each step on a log scale
        # if transform_time:
        #     # Transforms are not defined for times outside input times.
        #     # Must pad times to prevent issues with transforms if making secondary x-axis
        #     x_plot, trans_functions = get_transformed_plot_time(times, self.step_times, linear_time_axis)
        # else:
        #     x_plot = times
        # Get time transform functions
        if transform_time:
            trans_functions = self.get_chrono_transforms(show_outliers)
        else:
            trans_functions = None

        # Get fitted response
        
        y_hat = self.predict_response(x=x, subtract_background=subtract_background, y_bkg=None, **predict_kw)

        # # Add the fitted background to the predicted response
        # if not subtract_background:
        #     y_hat += self.raw_response_background

        # Get appropriate scale
        if scale_prefix is None:
            if plot_data:
                scale_prefix = utils.scale.get_common_scale_prefix([y_hat, self.raw_response_signal])
            else:
                scale_prefix = utils.scale.get_scale_prefix(y_hat)
        scale_factor = utils.scale.get_factor_from_prefix(scale_prefix)

        # Plot data
        if plot_data:
            y_meas = self.raw_response_signal
            if subtract_background:
                # Add the removed background back to the data
                if y_bkg is not None:
                    y_meas = y_meas - y_bkg
                else:
                    y_meas = y_meas - self.raw_response_background

            plot_tuple = utils.chrono.signals_to_tuple(self.get_fit_times(), None, y_meas, self.chrono_mode)
            plot_chrono(plot_tuple, chrono_mode=self.chrono_mode, step_times=self.nonconsec_step_times,
                        axes=ax, transform_time=transform_time, trans_functions=trans_functions,
                        linear_time_axis=False, scale_prefix=scale_prefix,
                        label=data_label, tight_layout=tight_layout, **data_kw)

        # Plot outliers
        if show_outliers and self.chrono_outliers is not None:
            if outlier_kw is None:
                outlier_kw = {'c': 'r', 'label': 'Outliers'}
            _, y_out = utils.chrono.get_input_and_response(self.chrono_outliers[1], self.chrono_outliers[2],
                                                           self.chrono_mode)
            plot_tuple = utils.chrono.signals_to_tuple(self.chrono_outliers[0], None, y_out, self.chrono_mode)
            plot_chrono(plot_tuple, chrono_mode=self.chrono_mode, step_times=self.nonconsec_step_times,
                        axes=ax, transform_time=transform_time, trans_functions=trans_functions,
                        linear_time_axis=False, scale_prefix=scale_prefix,
                        tight_layout=tight_layout, **outlier_kw)

        # Plot fitted response
        plot_tuple = utils.chrono.signals_to_tuple(times, None, y_hat, self.chrono_mode)
        plot_chrono(plot_tuple, chrono_mode=self.chrono_mode, step_times=self.nonconsec_step_times,
                    axes=ax, plot_func='plot', transform_time=transform_time, trans_functions=trans_functions,
                    linear_time_axis=linear_time_axis,
                    display_linear_ticks=display_linear_ticks, linear_tick_kw=linear_tick_kw,
                    scale_prefix=scale_prefix,
                    c=c, tight_layout=tight_layout, **kw)
        # ax.plot(x_plot, y_hat / scale_factor, c=c, **kw)

        # # Add linear time axis
        # if linear_time_axis and transform_time:
        #     axt = add_linear_time_axis(ax, times, self.step_times, trans_functions)
        # else:
        #     axt = None

        # Labels
        # if transform_time:
        #     ax.set_xlabel('$f(t)$')
        # else:
        #     ax.set_xlabel('$t$ (s)')
        #
        # if self.chrono_mode == 'galv':
        #     ax.set_ylabel(f'$v$ ({scale_prefix}V)')
        # elif self.chrono_mode == 'pot':
        #     ax.set_ylabel(f'$i$ ({scale_prefix}A)')

        # if tight_layout:
        #     fig.tight_layout()

        # if return_second_ax:
        #     return ax, axt
        # else:
        return ax

    def plot_chrono_residuals(self, plot_sigma=True, ax=None, x=None, transform_time=False,
                              linear_time_axis=False, display_linear_ticks=True, linear_tick_kw=None,
                              subtract_background=True, y_bkg=None,
                              predict_kw=None,
                              s=10, alpha=0.5, scale_prefix=None, show_outliers=False, outlier_kw=None,
                              tight_layout=True, **kw):

        # Check x_axis string
        # x_axis_options = ['t', 'f(t)', 'index']
        # if x_axis not in x_axis_options:
        #     raise ValueError(f'Invalid x_axis option {x_axis}. Options: {x_axis_options}')

        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))
        else:
            fig = ax.get_figure()

        if display_linear_ticks is None:
            if transform_time and not linear_time_axis:
                display_linear_ticks = True
            else:
                display_linear_ticks = False

        if linear_time_axis is None:
            if transform_time and not display_linear_ticks:
                linear_time_axis = True
            else:
                linear_time_axis = False

        # Get fit times
        times = self.get_fit_times()

        # Get time transforms
        if transform_time:
            trans_functions = self.get_chrono_transforms(show_outliers)
        else:
            trans_functions = None

        # Get outlier times
        # if show_outliers and self.chrono_outliers is not None:
        #     t_out = self.chrono_outliers[0]
        #     # Get sorted indices for plotting vs. index
        #     all_times, t_sort_index = np.unique(np.concatenate([times, t_out]), return_inverse=True)
        #     t_index = t_sort_index[:len(times)]
        #     t_out_index = t_sort_index[len(times):]
        # else:
        #     t_index = np.arange(len(times))
        #     t_out_index = None
        #     t_out = None

        # Get x values for plotting
        # trans_functions = None
        # if x_axis == 'f(t)':
        #     # Transform time to visualize each step on a log scale
        #     x_plot, trans_functions = get_transformed_plot_time(times, self.nonconsec_step_times)
        #     ax.set_xlabel('$f(t)$')
        # elif x_axis == 'index':
        #     # Uniform spacing
        #     x_plot = t_index
        #     ax.set_xlabel('Sample index')
        # else:
        #     x_plot = times
        #     ax.set_xlabel('$t$ (s)')

        # Get model response
        if predict_kw is None:
            predict_kw = {}
        y_hat = self.predict_response(x=x, subtract_background=subtract_background, y_bkg=y_bkg, **predict_kw)

        # Get measured response
        y_meas = self.raw_response_signal

        # Remove estimated background from data if specified
        if subtract_background:
            if y_bkg is not None:
                y_meas = y_meas - y_bkg
            else:
                y_meas = y_meas - self.raw_response_background

        # Calculate residuals
        y_err = y_meas - y_hat

        # Get appropriate scale
        if scale_prefix is None:
            scale_prefix, scale_factor = utils.scale.get_scale_prefix_and_factor(y_err)
        else:
            scale_factor = utils.scale.get_factor_from_prefix(scale_prefix)

        # Plot outliers
        if show_outliers and self.chrono_outliers is not None:
            t_out = self.chrono_outliers[0]
            _, y_out = utils.chrono.get_input_and_response(self.chrono_outliers[1], self.chrono_outliers[2],
                                                           self.chrono_mode)
            y_err_out = self.predict_response(times=t_out, x=x, subtract_background=True, **predict_kw) - y_out

            # Get x values for plotting
            # if x_axis == 'f(t)':
            #     x_out = trans_functions[1](t_out)
            # elif x_axis == 'index':
            #     x_out = t_out_index
            # else:
            #     x_out = t_out

            # Plot residuals
            if outlier_kw is None:
                outlier_kw = {'s': s, 'alpha': alpha, 'c': 'r', 'label': 'Outliers'}
            # ax.scatter(x_out, y_err_out / scale_factor, **outlier_kw)

            plot_tuple = utils.chrono.signals_to_tuple(t_out, None, y_err_out, self.chrono_mode)
            plot_chrono(plot_tuple, chrono_mode=self.chrono_mode, step_times=self.nonconsec_step_times,
                        axes=ax, transform_time=transform_time, trans_functions=trans_functions,
                        linear_time_axis=False,
                        display_linear_ticks=False,
                        scale_prefix=scale_prefix,
                        tight_layout=tight_layout, **outlier_kw)

        # Plot residuals
        plot_tuple = utils.chrono.signals_to_tuple(times, None, y_err, self.chrono_mode)
        plot_chrono(plot_tuple, chrono_mode=self.chrono_mode, step_times=self.nonconsec_step_times,
                    axes=ax, transform_time=transform_time, trans_functions=trans_functions,
                    linear_time_axis=linear_time_axis,
                    display_linear_ticks=display_linear_ticks, linear_tick_kw=linear_tick_kw,
                    scale_prefix=scale_prefix,
                    tight_layout=tight_layout, alpha=alpha, **kw)

        # ax.scatter(x_plot, y_err / scale_factor, s=s, alpha=alpha, **kw)

        # Indicate zero
        ax.axhline(0, c='k', lw=1, zorder=-10)

        # Show error structure
        if plot_sigma:
            sigma = self.predict_sigma('chrono') / scale_factor
            if sigma is not None:
                if transform_time:
                    x_plot = trans_functions[1](times)
                else:
                    x_plot = times
                ax.fill_between(x_plot, -3 * sigma, 3 * sigma, color='k', lw=0, alpha=0.15)

        # # Add linear time axis
        # if linear_time_axis and x_axis == 'f(t)':
        #     axt = add_linear_time_axis(ax, times, self.step_times, trans_functions)

        # Labels
        if self.chrono_mode == 'galv':
            ax.set_ylabel(fr'$v - \hat{{v}}$ ({scale_prefix}V)')
        elif self.chrono_mode == 'pot':
            ax.set_ylabel(fr'$i - \hat{{i}}$ ({scale_prefix}A)')

        fig.tight_layout()

        return ax

    def plot_chrono_correction(self, ax=None, transform_time=False, linear_time_axis=True,
                               display_linear_ticks=False, linear_tick_kw=None,
                               scale_prefix=None, show_background=True, bkg_twin_ax=False,
                               tight_layout=True, raw_kw=None, corrected_kw=None, background_kw=None):

        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 2.75))
        else:
            fig = ax.get_figure()

        # Get fit times
        times = self.get_fit_times()

        y_meas = self.raw_response_signal
        y_bkg = self.raw_response_background
        
        # Add the polynomial baseline
        y_bkg_plot = y_bkg + self.predict_v_baseline(self.get_fit_times())
        
        if show_background:
            # We want to show the background and data on the same axes
            # We have to remove the 0-degree offset from the background to be subtracted
            y_bkg = y_bkg_plot - self.fit_parameters["v_baseline"][0]
        else:
            # The background won't be shown, so we can subtract the 0-degree offset from the data
            y_bkg = y_bkg_plot

        # Get appropriate scale
        if scale_prefix is None:
            scale_prefix = utils.scale.get_common_scale_prefix([y_meas, y_meas - y_bkg])
        # scale_factor = utils.scale.get_factor_from_prefix(scale_prefix)

        # Plot raw data
        plot_tuple = utils.chrono.signals_to_tuple(times, None, y_meas, self.chrono_mode)
        if raw_kw is None:
            raw_kw = {'label': 'Raw', 'plot_func': 'plot'}
        plot_chrono(plot_tuple, chrono_mode=self.chrono_mode, step_times=self.nonconsec_step_times,
                    axes=ax, transform_time=transform_time,
                    linear_time_axis=False, scale_prefix=scale_prefix,
                    tight_layout=False, **raw_kw)

        # Plot the background
        if show_background:
            if bkg_twin_ax:
                ax_bkg = ax.twinx()
                bkg_scale_prefix = None
                # # Why did I not include the baseline voltage here?
                # y_bkg_plot = y_bkg
            else:
                ax_bkg = ax
                bkg_scale_prefix = scale_prefix
                

            plot_tuple = utils.chrono.signals_to_tuple(times, None, y_bkg_plot,
                                                       self.chrono_mode)
            if background_kw is None:
                background_kw = {'label': 'Background', 'plot_func': 'plot', 'c': 'r', 'alpha': 0.75}

            plot_chrono(plot_tuple, chrono_mode=self.chrono_mode, step_times=self.nonconsec_step_times,
                        axes=ax_bkg, transform_time=transform_time,
                        linear_time_axis=False, scale_prefix=bkg_scale_prefix,
                        tight_layout=False, **background_kw)

            if bkg_twin_ax:
                units = ax_bkg.get_ylabel().split(' ')[1]
                ax_bkg.set_ylabel(f'Background {units}')
        else:
            ax_bkg = None

        # Plot the corrected signal
        plot_tuple = utils.chrono.signals_to_tuple(times, None, y_meas - y_bkg, self.chrono_mode)
        if corrected_kw is None:
            corrected_kw = {'label': 'Corrected', 'plot_func': 'plot', 'c': 'k', 'alpha': 0.75}
        plot_chrono(plot_tuple, chrono_mode=self.chrono_mode, step_times=self.nonconsec_step_times,
                    axes=ax, transform_time=transform_time,
                    linear_time_axis=linear_time_axis,
                    display_linear_ticks=display_linear_ticks, linear_tick_kw=linear_tick_kw,
                    scale_prefix=scale_prefix,
                    tight_layout=tight_layout, **corrected_kw)

        if show_background and bkg_twin_ax:
            handles, labels = ax.get_legend_handles_labels()
            h_bkg, l_bkg = ax_bkg.get_legend_handles_labels()
            leg = ax.legend(handles=handles + h_bkg, labels=labels + l_bkg)
            leg.set_zorder(10)
        else:
            ax.legend()

        return ax

    def plot_eis_fit(self, frequencies=None, axes=None, plot_type='nyquist', plot_data=True, data_kw=None,
                     bode_cols=None, bode_rep="cartesian", data_label='', scale_prefix=None, area=None, normalize=False,
                     x=None, predict_kw={}, c='k', tight_layout=True, show_outliers=False, outlier_kw=None, 
                     y_offset: float = 0.0,
                     **kw):

        # Set default data plotting kwargs if not provided
        if data_kw is None:
            data_kw = dict(s=10, alpha=0.5)

        # Get model impedance
        if frequencies is None:
            frequencies = self.get_fit_frequencies()
        z_hat = self.predict_z(frequencies, x=x, **predict_kw)

        # Get rp for normalization
        if normalize:
            normalize_rp = self.predict_r_p()
            scale_prefix = ''
        else:
            normalize_rp = None

        # Get scale prefix
        if scale_prefix is None:
            if area is None:
                area_mult = 1
            else:
                area_mult = area
            z_hat_concat = utils.eis.complex_vector_to_concat(z_hat) * area_mult
            if plot_data:
                z_data_concat = utils.eis.complex_vector_to_concat(self.z_fit) * area_mult
                scale_prefix = utils.scale.get_common_scale_prefix([z_data_concat, z_hat_concat])
            else:
                scale_prefix = utils.scale.get_scale_prefix(z_hat_concat)

        # Plot data if requested
        if plot_data:
            axes = plot_eis((self.get_fit_frequencies(), self.z_fit - 1j * y_offset), plot_type, axes=axes, scale_prefix=scale_prefix,
                            label=data_label, bode_rep=bode_rep,
                            bode_cols=bode_cols, area=area, normalize=normalize, normalize_rp=normalize_rp,
                            tight_layout=False, **data_kw)

        # Plot outliers if requested
        if show_outliers and self.eis_outliers is not None:
            if outlier_kw is None:
                outlier_kw = {'c': 'r', 'label': 'Outliers'}
            axes = plot_eis((self.eis_outliers[0], self.eis_outliers[1]  - 1j * y_offset), plot_type, axes=axes, scale_prefix=scale_prefix, bode_rep=bode_rep,
                            bode_cols=bode_cols, area=area, normalize=normalize, normalize_rp=normalize_rp,
                            tight_layout=False, **outlier_kw)

        # Plot fit
        axes = plot_eis((frequencies, z_hat  - 1j * y_offset), plot_type, axes=axes, plot_func='plot', c=c, scale_prefix=scale_prefix, bode_rep=bode_rep,
                        bode_cols=bode_cols, area=area, normalize=normalize, normalize_rp=normalize_rp,
                        tight_layout=tight_layout, **kw)

        # fig = np.atleast_1d(axes)[0].get_figure()
        # fig.tight_layout()

        return axes

    def plot_eis_residuals(self, plot_sigma=True, axes=None, scale_prefix=None, x=None,
                           predict_kw={}, part='both', show_outliers=False, outlier_kw=None,
                           s=10, alpha=0.5, **kw):

        if part == 'both':
            bode_cols = ['real', 'imag']
        elif part == 'real':
            bode_cols = ['real']
        elif part == 'imag':
            bode_cols = ['imag']
        else:
            raise ValueError(f"Invalid part argument {part}. Options: 'both', 'real', 'imag'")

        if axes is None:
            fig, axes = plt.subplots(1, len(bode_cols), figsize=(3 * len(bode_cols), 2.75))
        else:
            fig = np.atleast_1d(axes)[0].get_figure()
        axes = np.atleast_1d(axes)

        # Get fit frequencies
        f_fit = self.get_fit_frequencies()

        # Get model impedance
        y_hat = self.predict_z(f_fit, x=x, **predict_kw)

        # Calculate residuals
        y_err = self.z_fit - y_hat

        # Get scale prefix
        if scale_prefix is None:
            scale_prefix = utils.scale.get_scale_prefix(np.concatenate([y_err.real, y_err.imag]))

        # Plot residuals
        plot_eis((f_fit, y_err), axes=axes, plot_type='bode', bode_cols=bode_cols,
                 s=s, alpha=alpha, label='Residuals', scale_prefix=scale_prefix, **kw)

        # Plot outliers
        if show_outliers and self.eis_outliers is not None:
            f_out, z_out = self.eis_outliers

            # Calculate residuals
            y_err_out = self.predict_z(f_out, x=x, **predict_kw) - z_out

            # Plot residuals
            if outlier_kw is None:
                outlier_kw = {'c': 'r', 'label': 'Outliers'}
            plot_eis((f_out, y_err_out), axes=axes, plot_type='bode', bode_cols=bode_cols,
                     s=s, alpha=alpha, scale_prefix=scale_prefix, **outlier_kw)

        # Indicate zero
        for ax in axes:
            ax.axhline(0, c='k', lw=1, zorder=-10)

        # Plot error structure
        if plot_sigma:
            sigma = self.predict_sigma('eis')
            if sigma is not None:
                scale_factor = scale_factor = utils.scale.get_factor_from_prefix(scale_prefix)
                if 'real' in bode_cols:
                    axes[bode_cols.index('real')].fill_between(f_fit, -3 * sigma.real / scale_factor,
                                                                3 * sigma.real / scale_factor,
                                                                color='k', lw=0, alpha=0.15, zorder=-10,
                                                                label=r'$\pm 3 \sigma$')
                if 'imag' in bode_cols:
                    axes[bode_cols.index('imag')].fill_between(f_fit, -3 * sigma.imag / scale_factor,
                                                                3 * sigma.imag / scale_factor,
                                                                color='k', lw=0, alpha=0.15, zorder=-10,
                                                                label=r'$\pm 3 \sigma$')

            axes[-1].legend()

        # Update axis labels
        if 'real' in bode_cols:
            axes[bode_cols.index('real')].set_ylabel(fr'$Z^{{\prime}} - \hat{{Z}}^{{\prime}}$ ({scale_prefix}$\Omega$)')
        if 'imag' in bode_cols:
            axes[bode_cols.index('imag')].set_ylabel(
                fr'$-(Z^{{\prime\prime}} - \hat{{Z}}^{{\prime\prime}})$ ({scale_prefix}$\Omega$)')

        fig.tight_layout()

        return axes
    
    def plot_distribution(self, tau=None, ppd=20, x=None, ax=None, scale_prefix=None, plot_bounds=False,
                          shade_extrap: bool=False, shade_kw: Optional[dict] = None,
                          normalize=False, normalize_by=None, sign=None,
                          plot_ci=False, ci_kw=None, ci_quantiles=[0.025, 0.975],  # sample_kw={},
                          area=None, order=0, mark_peaks=False, mark_peaks_kw=None, tight_layout=True,
                          return_line=False, freq_axis=False, y_offset: float = 0.,
                          set_xlim: bool = False,
                          **kw):
        warnings.warn("plot_distribution is deprecated and will be removed in the future. Please use plot_drt instead", DeprecationWarning)
        return self.plot_drt(tau=tau, ppd=ppd, x=x, ax=ax, scale_prefix=scale_prefix, plot_bounds=plot_bounds,
                             shade_extrap=shade_extrap, shade_kw=shade_kw, normalize=normalize, normalize_by=normalize_by,
                             sign=sign, plot_ci=plot_ci, ci_kw=ci_kw, ci_quantiles=ci_quantiles,
                             area=area, order=order, mark_peaks=mark_peaks, mark_peaks_kw=mark_peaks_kw, tight_layout=tight_layout,
                             return_line=return_line, freq_axis=freq_axis, y_offset=y_offset,
                             set_xlim=set_xlim, **kw)

    def plot_drt(self, tau=None, ppd=20, x=None, ax=None, scale_prefix=None, plot_bounds=False,
                          shade_extrap: bool=False, shade_kw: Optional[dict] = None,
                          normalize=False, normalize_by=None, sign=None,
                          plot_ci=False, ci_kw=None, ci_quantiles=[0.025, 0.975],  # sample_kw={},
                          area=None, order=0, mark_peaks=False, mark_peaks_kw=None, tight_layout=True,
                          return_line=False, freq_axis=False, y_offset: float = 0.,
                          set_xlim: bool = False,
                          **kw):
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))
        else:
            fig = ax.get_figure()

        if tau is None:
            # If tau is not provided, go one decade beyond self.basis_tau with finer spacing
            tau = self.get_tau_eval(ppd)

        if sign is None:
            if self.series_neg:
                sign = [-1, 1]
            else:
                sign = [1]
        signs = np.atleast_1d(sign)

        # if sample_kw == {} and self.map_sample_kw is not None:
        #     sample_kw = self.map_sample_kw

        # Normalize
        if normalize_by is not None:
            normalize = True
        
        if normalize:
            if normalize_by is None:
                normalize_by = self.predict_r_p(x=x, absolute=True)
            scale_prefix = ''
            # print('r_p:', r_p)

        # Get common scale prefix if plotting multiple signs
        if self.series_neg and len(signs) > 1 and scale_prefix is None:
            gamma_pos = self.predict_drt(tau, x=x, order=order, sign=1)
            gamma_neg = self.predict_drt(tau, x=x, order=order, sign=-1)
            scale_prefix = utils.scale.get_common_scale_prefix([gamma_pos, gamma_neg])

        lines = []
        for sign in signs:
            # Calculate MAP distribution at evaluation points
            gamma = self.predict_drt(tau, x=x, order=order, sign=sign)
            # if line == 'mode':
            #     # Calculate MAP distribution at evaluation points
            #     gamma = self.predict_distribution(tau, x=x, order=order, sign=sign)
            # elif line == 'mean':
            #     # Estimate posterior mean from MAP samples
            #     x_mean = self.estimate_posterior_mean(**sample_kw)
            #     gamma = self.predict_distribution(tau, x=x_mean[self.get_qp_mat_offset():], order=order, sign=sign)
            # else:
            #     raise ValueError(f"Invalid line argument {line}. Options: 'mode', 'mean'")

            ax, info = plot_distribution(tau, gamma, ax, area, scale_prefix, 
                                         normalize_by=normalize_by, freq_axis=freq_axis,
                                         return_info=True, y_offset=y_offset, **kw)
            line, scale_prefix, scale_factor = info
            lines.append(line)

            if plot_ci:
                if self.fit_type.find('qphb') > -1:
                    gamma_lo, gamma_hi = self.predict_drt_ci(
                        tau, ppd, x=x, order=order, sign=sign, 
                        normalize=normalize, normalize_by=normalize_by,
                        quantiles=ci_quantiles
                    )
                    
                    if area is not None:
                        for g in (gamma_lo, gamma_hi):
                            g *= area
                            
                    # if normalize:
                    #     for g in (gamma_lo, gamma_hi):
                    #         g /= r_p

                    # Enforce sign constraint in CI
                    if gamma_lo is not None:
                        if self.series_neg:
                            if sign == 1:
                                gamma_lo = np.maximum(gamma_lo, 0)
                            elif sign == -1:
                                gamma_hi = np.minimum(gamma_hi, 0)
                        elif self.fit_kwargs['nonneg'] and order == 0:
                            gamma_lo = np.maximum(gamma_lo, 0)

                        if ci_kw is None:
                            ci_kw = {}
                        ci_defaults = dict(color=line[0].get_color(), lw=0.5, alpha=0.25, zorder=-10)
                        ci_defaults.update(ci_kw)
                        ci_kw_sign = ci_defaults

                        ax.fill_between(tau, gamma_lo / scale_factor + y_offset, gamma_hi / scale_factor + y_offset, **ci_kw_sign)

            if mark_peaks:
                if mark_peaks_kw is None:
                    mark_peaks_kw = {}
                self.mark_peaks(ax, x=x, sign=sign, scale_prefix=scale_prefix, area=area, 
                                normalize=normalize, normalize_by=normalize_by, y_offset=y_offset,
                                **mark_peaks_kw)

        if normalize:
            y_label = r'$\gamma \, / \, R_p$'
        else:
            if area is not None:
                y_units = r'$\Omega \cdot \mathrm{cm}^2$'
            else:
                y_units = r'$\Omega$'
            y_label = fr'$\gamma$ ({scale_prefix}{y_units})'
        ax.set_ylabel(y_label)

        # Indicate measurement range
        if plot_bounds:
            # Get min and max tau across all measurement types
            if self.fit_type.find('eis') > -1 or self.fit_type.find('hybrid') > -1:
                eis_tau_min = 1 / (2 * np.pi * np.max(self.get_fit_frequencies()))
                eis_tau_max = 1 / (2 * np.pi * np.min(self.get_fit_frequencies()))
            else:
                eis_tau_min = np.inf
                eis_tau_max = -np.inf

            if self.fit_type.find('chrono') > -1 or self.fit_type.find('hybrid') > -1:
                time_deltas = pp.get_time_since_step(self.get_fit_times(), self.step_times)
                chrono_tau_min = np.min(time_deltas)
                chrono_tau_max = np.max(time_deltas)
            else:
                chrono_tau_min = np.inf
                chrono_tau_max = -np.inf

            tau_min = min(eis_tau_min, chrono_tau_min)
            tau_max = max(eis_tau_max, chrono_tau_max)

            # Indicate bounds with vertical lines
            ax.axvline(tau_min, ls=':', c='k', alpha=0.6, lw=1.5, zorder=-10)
            ax.axvline(tau_max, ls=':', c='k', alpha=0.6, lw=1.5, zorder=-10)
            
        if shade_extrap:
            if shade_kw is None:
                shade_kw = {}
            shade_extrap_regions(ax, *self.get_data_tau_limits(), **shade_kw)
            
        if set_xlim:
            ax.set_xlim(np.min(tau), np.max(tau))

        if tight_layout:
            fig.tight_layout()

        if return_line:
            return ax, lines
        else:
            return ax

    def plot_dop(self, nu=None, x=None, ax=None, scale_prefix=None, normalize=False, normalize_tau=None,
                 invert_nu=True, phase=True, area=None,
                 plot_ci=False, ci_kw=None, ci_quantiles=[0.025, 0.975], order=0,
                 delta_density=False, include_ideal=True,
                 tight_layout=True, return_line=False, normalize_quantiles=(0, 1), 
                 y_offset: float = 0.0,
                 **kw):

        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))
        else:
            fig = ax.get_figure()

        nu, dop = self.predict_dop(nu=nu, x=x, normalize=normalize, normalize_tau=normalize_tau,
                                   order=order, return_nu=True, normalize_quantiles=normalize_quantiles,
                                   delta_density=delta_density, include_ideal=include_ideal)

        # Invert nu for more intuitive visualization
        if invert_nu:
            nu_plot = -nu
            x_label_sign = '-'
        else:
            nu_plot = nu
            x_label_sign = ''

        # Convert nu to phase for easier interpretation
        if phase:
            nu_plot = nu_plot * 90 #np.pi / 2
            x_label = fr'${x_label_sign}\theta$ ($^\circ$)'
        else:
            x_label = fr'${x_label_sign}\nu$'

        # Get scale factor
        if scale_prefix is None:
            scale_prefix = utils.scale.get_scale_prefix(dop)
        scale_factor = utils.scale.get_factor_from_prefix(scale_prefix)
        
        if area is not None:
            # Multiply by area
            scale_factor = scale_factor / area

        line = ax.plot(nu_plot, dop / scale_factor + y_offset, **kw)

        # Plot CI
        if plot_ci:
            if self.fit_type.find('qphb') > -1:
                dop_lo, dop_hi = self.predict_dop_ci(nu=nu, x=x, normalize=normalize, normalize_tau=normalize_tau,
                                                     quantiles=ci_quantiles, order=order,
                                                     normalize_quantiles=normalize_quantiles,
                                                     delta_density=delta_density, include_ideal=include_ideal)

                if dop_lo is not None:
                    # Enforce sign constraint in CI
                    if order == 0:
                        dop_lo = np.maximum(dop_lo, 0)

                    if ci_kw is None:
                        ci_kw = {}
                    ci_defaults = dict(color=line[0].get_color(), lw=0.5, alpha=0.2, zorder=-10)
                    ci_defaults.update(ci_kw)
                    ci_kw = ci_defaults

                    ax.fill_between(nu_plot, dop_lo / scale_factor + y_offset, dop_hi / scale_factor + y_offset, **ci_kw)

        if invert_nu:
            ax.set_xlabel(x_label)
        else:
            ax.set_xlabel(x_label)

        if area is not None:
            area_units = r'$\cdot \mathrm{cm}^2$'
        else:
            area_units = ''
            
        if normalize:
            ax.set_ylabel(fr'$\tilde{{\rho}}$ ({scale_prefix}$\Omega${area_units})')
        else:
            ax.set_ylabel(fr'$\rho$ ({scale_prefix}$\Omega \cdot \mathrm{{s}}^\nu${area_units})')

        if tight_layout:
            fig.tight_layout()

        if return_line:
            return ax, line
        else:
            return ax

    def plot_ddt(self, tau_dd=None, ppd=20, x=None, ax=None, scale_prefix=None, normalize=False,
                 plot_ci=False, ci_kw=None, ci_quantiles=[0.025, 0.975], order=0,
                 tight_layout=True, return_line=False, y_offset: float = 0., **kw):
        """
        Plot the distribution of diffusion times (DDT).

        In 2D mode (anomalous-exponent grid) a heat map of q(tau_D, p) is drawn
        together with the p-marginal q(tau_D).

        :param ndarray tau_dd: diffusion times at which to evaluate the DDT
        :param int ppd: points per decade for the default evaluation grid
        :param x: DDT parameters (dict, raw vector, or None)
        :param ax: matplotlib axis to plot on
        :param bool normalize: if True, normalize so that int q dln(tau_dd) = 1
        :param bool plot_ci: if True, plot credible intervals (1D mode only)
        :return: matplotlib axis (or axis, line if return_line is True)
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))
        else:
            fig = ax.get_figure()

        if tau_dd is None:
            tau_dd = self.get_tau_dd_eval(ppd)

        if self.basis_p is not None:
            # 2D mode: heat map of q(tau_D, p) + p-marginal
            tau_dd, p_grid, q2d = self.predict_ddt(tau_dd, ppd=ppd, x=x, order=order,
                                                   normalize=normalize, return_p=True,
                                                   return_tau_dd=True)
            q2d = np.asarray(q2d).T  # shape (n_p, n_tau_eval)

            if scale_prefix is None:
                scale_prefix = utils.scale.get_scale_prefix(q2d)
            scale_factor = utils.scale.get_factor_from_prefix(scale_prefix)

            X, Y = np.meshgrid(tau_dd, p_grid)
            pc = ax.pcolormesh(X, Y, q2d / scale_factor, shading='auto')

            # p-marginal (line kwargs such as c/lw are applied to the line)
            line_kw = {kk: kw.pop(kk) for kk in list(kw) if kk in ('c', 'color', 'lw', 'ls', 'label')}
            line_color = line_kw.pop('color', line_kw.pop('c', 'k'))
            q_tau = self.predict_ddt(tau_dd, ppd=ppd, x=x, order=order,
                                     normalize=normalize, marginalize='tau')
            ax.plot(tau_dd, q_tau / scale_factor + y_offset, color=line_color, lw=1, **line_kw)

            fig.colorbar(pc, ax=ax, label=fr'$q$ ({scale_prefix}$\Omega$)')
            ax.set_xscale('log')
            ax.set_xlabel(r'$\tau_D$ (s)')
            ax.set_ylabel(r'$p$')
            ax.set_title('DDT')

            if tight_layout:
                fig.tight_layout()
            return ax

        q = self.predict_ddt(tau_dd, ppd=ppd, x=x, order=order, normalize=normalize)

        if scale_prefix is None:
            scale_prefix = utils.scale.get_scale_prefix(q)
        scale_factor = utils.scale.get_factor_from_prefix(scale_prefix)

        line = ax.plot(tau_dd, q / scale_factor + y_offset, **kw)

        if plot_ci:
            if self.fit_type.find('qphb') > -1:
                q_lo, q_hi = self.predict_ddt_ci(tau_dd=tau_dd, ppd=ppd, x=x,
                                                 quantiles=ci_quantiles, order=order, normalize=normalize)
                if q_lo is not None:
                    q_lo = np.maximum(q_lo, 0)
                    if ci_kw is None:
                        ci_kw = {}
                    ci_defaults = dict(color=line[0].get_color(), lw=0.5, alpha=0.2, zorder=-10)
                    ci_defaults.update(ci_kw)
                    ci_kw = ci_defaults
                    ax.fill_between(tau_dd, q_lo / scale_factor + y_offset, q_hi / scale_factor + y_offset, **ci_kw)

        ax.set_xscale('log')
        ax.set_xlabel(r'$\tau_D$ (s)')
        if normalize:
            ax.set_ylabel(fr'$\tilde{{q}}$ ({scale_prefix})')
        else:
            ax.set_ylabel(fr'$q$ ({scale_prefix}$\Omega$)')

        if tight_layout:
            fig.tight_layout()

        if return_line:
            return ax, line
        else:
            return ax

    def plot_results(self, axes=None, x=None, show_outliers=False, outlier_kw=None,
                     drt_kw=None, eis_fit_kw=None, eis_resid_kw=None,
                     chrono_fit_kw=None, chrono_resid_kw=None):

        ddt_ax = None

        # Define axes
        if self.fit_type.find('eis') > -1 and self.fit_ddt:
            if axes is None:
                fig, axes = plt.subplots(2, 3, figsize=(9, 5))
            else:
                fig = axes.ravel()[0].get_figure()
            drt_ax = axes[0, 0]
            eis_fit_ax = axes[0, 1]
            ddt_ax = axes[0, 2]
            eis_resid_axes = axes[1, :2]
            axes[1, 2].axis('off')
            chrono_fit_ax = None
            chrono_resid_ax = None
        elif self.fit_type.find('eis') > -1:
            if axes is None:
                fig, axes = plt.subplots(2, 2, figsize=(6, 5))
            else:
                fig = axes.ravel()[0].get_figure()
            drt_ax = axes[0, 0]
            eis_fit_ax = axes[0, 1]
            eis_resid_axes = axes[1]
            chrono_fit_ax = None
            chrono_resid_ax = None
        elif self.fit_type.find('chrono') > -1:
            if axes is None:
                fig, axes = plt.subplots(1, 3, figsize=(9, 2.5))
            else:
                fig = axes.ravel()[0].get_figure()
            drt_ax = axes[0]
            chrono_fit_ax = axes[1]
            chrono_resid_ax = axes[2]
            eis_fit_ax = None
            eis_resid_axes = None
        else:
            if axes is None:
                fig, axes = plt.subplots(2, 3, figsize=(9, 5.))
            else:
                fig = axes.ravel()[0].get_figure()
            drt_ax = axes[0, 0]
            eis_fit_ax = axes[0, 1]
            chrono_fit_ax = axes[0, 2]
            eis_resid_axes = axes[1, :2]
            chrono_resid_ax = axes[1, 2]

        # Plot distribution
        if drt_kw is None and self.series_neg:
            for sign, c in zip([1, -1], ['b', 'r']):
                self.plot_drt(x=x, sign=sign, ax=drt_ax, plot_ci=True, c=c)
        else:
            if drt_kw is None:
                drt_kw = {}
            
            self.plot_drt(x=x, ax=drt_ax, **dict(plot_ci=True, c='k', shade_extrap=True, set_xlim=True) | drt_kw)
            
        drt_ax.set_title('DRT')

        # Plot DDT
        if ddt_ax is not None:
            self.plot_ddt(x=x, ax=ddt_ax, plot_ci=True, c='k')
            ddt_ax.set_title('DDT')

        # Plot EIS fit and residuals
        if eis_fit_ax is not None:
            if eis_fit_kw is None:
                eis_fit_kw = {}
            if eis_resid_kw is None:
                eis_resid_kw = {}

            self.plot_eis_fit(axes=eis_fit_ax, plot_type='nyquist', x=x,
                              show_outliers=show_outliers, outlier_kw=outlier_kw,
                              **eis_fit_kw)
            eis_fit_ax.set_title('EIS Fit')

            self.plot_eis_residuals(axes=eis_resid_axes, x=x,
                                    show_outliers=show_outliers, outlier_kw=outlier_kw,
                                    **eis_resid_kw)
            eis_resid_axes[0].set_title(r'$Z^\prime$ Residuals')
            eis_resid_axes[1].set_title(r'$Z^{\prime\prime}$ Residuals')
            eis_resid_axes[0].legend()
            eis_resid_axes[1].get_legend().remove()

        # Plot chrono fit and residuals
        if chrono_fit_ax is not None:
            if chrono_fit_kw is None:
                chrono_fit_kw = {'linear_time_axis': False}
            if chrono_resid_kw is None:
                chrono_resid_kw = {'linear_time_axis': False}

            self.plot_chrono_fit(ax=chrono_fit_ax, x=x,
                                 show_outliers=show_outliers, outlier_kw=outlier_kw,
                                 **chrono_fit_kw)
            chrono_fit_ax.set_title('Chrono Fit')

            self.plot_chrono_residuals(ax=chrono_resid_ax, x=x,
                                       show_outliers=show_outliers, outlier_kw=outlier_kw,
                                       **chrono_resid_kw)
            chrono_resid_ax.set_title('Chrono Residuals')

        fig.tight_layout()

        return axes

    # Preprocessing (matrix calculations, scaling)
    # --------------------------------------------
    def _solve_data_scale(self, hypers, penalty_matrices, penalty_type, rho_vector, dop_rho_vector,
                          s_vectors, rzv, rzm, nonneg, ddt_rho_vector=None):
        if not self.fit_drt:
            # Pure DOP / DDT mode: no DRT block to estimate Rp from
            return 1.0, None, None

        # Perform a quick Elastic Net solution to estimate Rp
        x_rp = qphb.estimate_x_rp(hypers, penalty_matrices, penalty_type, rho_vector, dop_rho_vector,
                                  s_vectors, rzv, rzm, nonneg, self.special_qp_params, l2_lambda_0=1e-4,
                                  l1_lambda_0=1e-3, ddt_rho_vector=ddt_rho_vector)
        rp_est = self.predict_r_p(absolute=True, x=x_rp, raw=True)

        if self.fit_dop:
            dop_start, dop_end = self.dop_indices
            x_drt_max = np.max(np.abs(x_rp[self.get_qp_mat_offset():]))
            x_dop_max = np.max(np.abs(x_rp[dop_start:dop_end]))
            dop_rescale_factor = x_drt_max / x_dop_max
        else:
            dop_rescale_factor = None

        if self.fit_ddt:
            ddt_start, ddt_end = self.ddt_indices
            x_drt_max = np.max(np.abs(x_rp[self.get_qp_mat_offset():]))
            x_ddt_max = np.max(np.abs(x_rp[ddt_start:ddt_end]))
            ddt_rescale_factor = x_drt_max / x_ddt_max
        else:
            ddt_rescale_factor = None

        return rp_est, dop_rescale_factor, ddt_rescale_factor

    def _prep_for_fit(self,
                      # Chrono data
                      times, i_signal, v_signal,
                      # EIS data
                      frequencies, z,
                      # Chrono options
                      step_times, step_sizes, downsample, downsample_kw, offset_steps, step_offset_size, discard_first_n, smooth_inf_response,
                      # Scaling
                      scale_data, rp_scale,
                      penalty_type, derivative_weights):

        start_time = time.time()

        # Checks
        utils.validation.check_penalty_type(penalty_type)
        utils.validation.check_eis_data(frequencies, z)
        utils.validation.check_chrono_data(times, i_signal, v_signal)

        # Store fit kwargs that may be relevant for prediction and plotting
        self.fit_kwargs = {'smooth_inf_response': smooth_inf_response, 'offset_steps': offset_steps, "step_offset_size": step_offset_size}

        # Clear map samples
        self.map_samples = None
        self.map_sample_kw = None

        # If chrono data provided, get input signal step information
        sample_times, sample_i, sample_v, step_times, step_sizes, tau_rise = self.process_chrono_signals(
            times, i_signal, v_signal, step_times, step_sizes, offset_steps, step_offset_size, discard_first_n, downsample, downsample_kw
        )

        # Set basis_tau - must have chrono step information
        if self.fit_drt:
            if self.fixed_basis_tau is not None:
                self.basis_tau = self.fixed_basis_tau
            else:
                # Default: 10 ppd basis grid. Extend basis tau one decade beyond data on each end
                self.basis_tau = pp.get_basis_tau(frequencies, times, step_times, tau_grid=self.tau_supergrid, 
                                                  extend_decades=self.extend_basis_decades)
        else:
            # Pure DOP / DDT mode: no DRT block
            self.basis_tau = None

        # If epsilon is not set, apply default value
        if self.tau_epsilon is None and self.fit_drt:
            if self.tau_basis_type in ('gaussian', 'zga'):
                # Determine epsilon from basis_tau spacing
                dlntau = np.mean(np.diff(np.log(self.basis_tau)))
                self.tau_epsilon = 1 / dlntau
            elif self.tau_basis_type == 'Cole-Cole':
                # Default for Cole-Cole
                self.tau_epsilon = 0.95

        # If using ZGA basis function, set parameters
        if self.tau_basis_type == 'zga' and self.zga_params is None:
            self.set_zga_params()

        # Set t_fit again to ensure matrix recalc status is updated
        self.t_fit = sample_times

        if sample_times is not None:
            # Get matrix and vectors for chrono fit
            rm_drt, inf_rv, induc_rv, cap_rv, rm_dop = \
                self._prep_chrono_fit_matrix(sample_times, step_times, step_sizes,
                                             tau_rise, smooth_inf_response)

            if self.series_neg:
                rm_drt = np.hstack((rm_drt, -rm_drt))
        else:
            # No chrono data
            self.t_fit = []
            rm_drt, inf_rv, induc_rv, cap_rv, rm_dop = None, None, None, None, None

        # DDT has no time-domain kernel (EIS only)
        rm_ddt = None

        if frequencies is not None:
            # Get matrix and vector for impedance fit
            zm_drt, induc_zv, cap_zv, zm_dop, zm_ddt = self._prep_impedance_fit_matrix(frequencies)

            if self.series_neg:
                zm_drt = np.hstack((zm_drt, -zm_drt))
        else:
            # No EIS data
            self.f_fit = []
            zm_drt, induc_zv, cap_zv, zm_dop, zm_ddt = None, None, None, None, None

        # Calculate penalty matrices
        penalty_matrices = self._prep_penalty_matrices(penalty_type, derivative_weights)
        if self.series_neg:
            for key in list(penalty_matrices.keys()):
                if key.find('dop') == -1 and key.find('ddt') == -1:
                    mat = penalty_matrices[key]
                    new_mat = np.kron(np.eye(2), mat)
                    penalty_matrices[key] = new_mat

        # Perform scaling
        i_signal_scaled, v_signal_scaled, z_scaled = self.scale_data(sample_times, sample_i, sample_v, step_times,
                                                                     step_sizes, z, scale_data, rp_scale)
        if self.print_diagnostics:
            print('Finished signal scaling')

        # Estimate chrono baseline after scaling
        if sample_times is not None:
            if self.chrono_mode == 'galv':
                response_baseline = np.median(v_signal_scaled[sample_times < step_times[0]])
            else:
                response_baseline = np.median(i_signal_scaled[sample_times < step_times[0]])
        else:
            response_baseline = None

        # scale chrono response matrix/vectors to input_signal_scale
        if rm_drt is not None:
            rm_drt = rm_drt / self.input_signal_scale
            induc_rv = induc_rv / self.input_signal_scale
            inf_rv = inf_rv / self.input_signal_scale
            cap_rv = cap_rv / self.input_signal_scale
            if rm_dop is not None:
                rm_dop = rm_dop / self.input_signal_scale

        if self.print_diagnostics:
            print('Finished prep_for_fit in {:.2f} s'.format(time.time() - start_time))

        return (sample_times, i_signal_scaled, v_signal_scaled, response_baseline, z_scaled), \
               (rm_drt, induc_rv, inf_rv, cap_rv, rm_dop, rm_ddt,
                zm_drt, induc_zv, cap_zv, zm_dop, zm_ddt, penalty_matrices)

    def _prep_chrono_fit_matrix(self, times, step_times, step_sizes, tau_rise, smooth_inf_response):
        # Recalculate matrices if necessary
        # TODO: incorporate step_times and step_sizes logic
        #  This is a placeholder to avoid issues with corner cases
        #  (e.g. if input_signal remains the same but step_times is manually specified)
        self._recalc_chrono_fit_matrix = True

        if self._recalc_chrono_fit_matrix:
            if self.print_diagnostics:
                print('Calculating chrono response matrix')
            rm, rm_layered = mat1d.construct_response_matrix(self.basis_tau, times, self.step_model, step_times,
                                                             step_sizes, basis_type=self.tau_basis_type,
                                                             epsilon=self.tau_epsilon, tau_rise=tau_rise,
                                                             op_mode=self.chrono_mode,
                                                             integrate_method=self.integrate_method,
                                                             zga_params=self.zga_params,
                                                             interpolate_grids=self.interpolate_lookups['response'])
            if self.print_diagnostics:
                print('Constructed chrono response matrix')
            self.fit_matrices['response'] = rm.copy()
            self.fit_matrices['rm_layered'] = rm_layered.copy()

            induc_rv = mat1d.construct_inductance_response_vector(times, self.step_model, step_times, step_sizes,
                                                                  tau_rise, self.chrono_mode)
            cap_rv = mat1d.construct_capacitance_response_vector(times, self.step_model, step_times, step_sizes,
                                                                 tau_rise, self.chrono_mode)

            self.fit_matrices['inductance_response'] = induc_rv
            self.fit_matrices['capacitance_response'] = cap_rv

            # With all matrices calculated, set recalc flags to False
            self._recalc_chrono_fit_matrix = False
            # self._recalc_chrono_prediction_matrix = False
            # We shouldn't reset _recalc_eis_prediction_matrix here.
            #  Example case: basis_tau changes during fit (triggers recalc, which is then overwritten here),
            #  but f_predict is unchanged
            #  When we next try to call predict_z, recalc status will be False even though basis_tau changed

        # Otherwise, reuse existing matrices as appropriate
        elif self._t_fit_subset_index is not None:
            # times is a subset of self.t_fit. Use sub-matrices of existing A array; do not overwrite
            rm = self.fit_matrices['response'][self._t_fit_subset_index, :].copy()
            induc_rv = self.fit_matrices['inductance_response'][self._t_fit_subset_index].copy()
            cap_rv = self.fit_matrices['capacitance_response'][self._t_fit_subset_index].copy()
        else:
            # All matrix parameters are the same. Use existing matrices
            rm = self.fit_matrices['response'].copy()
            induc_rv = self.fit_matrices['inductance_response'].copy()
            cap_rv = self.fit_matrices['capacitance_response'].copy()

        # Construct R_inf response vector. Fast - always calculate
        inf_rv = mat1d.construct_ohmic_response_vector(times, self.step_model, step_times, step_sizes, tau_rise,
                                                       self.raw_input_signal, smooth_inf_response, self.chrono_mode)
        self.fit_matrices['inf_response'] = inf_rv.copy()

        if self.fit_dop:
            rm_dop, rm_dop_layered = phasance.construct_phasor_v_matrix(times, self.basis_nu,
                                                                        self.nu_basis_type,
                                                                        self.nu_epsilon,
                                                                        self.step_model, step_times,
                                                                        step_sizes, self.chrono_mode)
            self.fit_matrices['rm_dop'] = rm_dop.copy()
            self.fit_matrices['rm_dop_layered'] = rm_dop_layered.copy()
        else:
            rm_dop = None

        return rm, inf_rv, induc_rv, cap_rv, rm_dop

    def _prep_impedance_fit_matrix(self, frequencies):
        self.f_fit = frequencies
        if not self.fit_drt:
            # Pure DOP / DDT mode: empty DRT impedance block
            zm = np.empty((len(frequencies), 0), dtype=complex)
            self.fit_matrices['impedance'] = zm.copy()
            self._recalc_eis_fit_matrix = False
        elif self._recalc_eis_fit_matrix:
            # Matrix calculation is required
            # Real matrix
            zmr = mat1d.construct_impedance_matrix(frequencies, 'real', tau=self.basis_tau,
                                                   basis_type=self.tau_basis_type, epsilon=self.tau_epsilon,
                                                   frequency_precision=self.frequency_precision,
                                                   zga_params=self.zga_params, integrate_method=self.integrate_method,
                                                   interpolate_grids=self.interpolate_lookups['z_real'])
            # Imaginary matrix
            zmi = mat1d.construct_impedance_matrix(frequencies, 'imag', tau=self.basis_tau,
                                                   basis_type=self.tau_basis_type, epsilon=self.tau_epsilon,
                                                   frequency_precision=self.frequency_precision,
                                                   zga_params=self.zga_params, integrate_method=self.integrate_method,
                                                   interpolate_grids=self.interpolate_lookups['z_imag'])
            # Complex matrix
            zm = zmr + 1j * zmi
            self.fit_matrices['impedance'] = zm.copy()

            # With matrices calculated, set recalc flags to False
            self._recalc_eis_fit_matrix = False
            # self._recalc_eis_prediction_matrix = False
            # We shouldn't reset _recalc_eis_prediction_matrix here.
            #  Example case: basis_tau changes during fit (triggers recalc, which is then overwritten here),
            #  but f_predict is unchanged
            #  When we next try to call predict_z, recalc status will be False even though basis_tau changed
        elif self._f_fit_subset_index is not None:
            # frequencies is a subset of self.f_fit. Use sub-matrices of existing matrix; do not overwrite
            zm = self.fit_matrices['impedance'][self._f_fit_subset_index, :].copy()
        else:
            # All matrix parameters are the same. Use existing matrix
            zm = self.fit_matrices['impedance'].copy()

        # Construct inductance impedance vector
        induc_zv = mat1d.construct_inductance_impedance_vector(frequencies)
        cap_zv = mat1d.construct_capacitance_impedance_vector(frequencies)

        # Distribution of phasances
        if self.fit_dop:
            zm_dop = phasance.construct_phasor_z_matrix(frequencies, self.basis_nu,
                                                        self.nu_basis_type, self.nu_epsilon)
        else:
            zm_dop = None
        self.fit_matrices['zm_dop'] = zm_dop

        # Distribution of diffusion times
        if self.fit_ddt:
            zm_ddt = ddt.get_ddt_impedance_matrix(frequencies, self.basis_tau_dd,
                                                  self.ddt_kernel_type, basis_type=self.tau_dd_basis_type,
                                                  ddt_epsilon=self.tau_dd_epsilon, p=self.ddt_p,
                                                  p_grid=self.basis_p, p_epsilon=self.ddt_p_epsilon)
            # Column-wise scaling for numerical conditioning (mirrors DOP scale vector)
            self.ddt_scale_vector = 1 / np.max(np.abs(zm_ddt), axis=0)
        else:
            zm_ddt = None
            self.ddt_scale_vector = None
        self.fit_matrices['zm_ddt'] = zm_ddt

        return zm, induc_zv, cap_zv, zm_dop, zm_ddt

    def _prep_penalty_matrices(self, penalty_type, derivative_weights, truncate=False):
        # Always calculate penalty (derivative) matrices - fast, avoids any gaps in recalc logic
        # if self._recalc_chrono_fit_matrix:
        penalty_matrices = {}
        for k in range(len(derivative_weights)):
            # ---- DRT penalty ----
            if not self.fit_drt:
                # Pure DOP / DDT mode: no DRT penalty block
                penalty_matrices[f'm{k}'] = np.zeros((0, 0))
                if penalty_type == 'discrete':
                    penalty_matrices[f'l{k}'] = np.zeros((0, 0))
            elif penalty_type == 'discrete':
                dk = basis.construct_func_eval_matrix(np.log(self.basis_tau), None, self.tau_basis_type,
                                                      self.tau_epsilon, order=k,
                                                      zga_params=self.zga_params)

                penalty_matrices[f'l{k}'] = dk.copy()
                penalty_matrices[f'm{k}'] = dk.T @ dk
            elif penalty_type == 'integral':
                if truncate:
                    integration_limits = (np.log(self.basis_tau[0]), np.log(self.basis_tau[-1]))
                else:
                    integration_limits = None
                dk = mat1d.construct_integrated_derivative_matrix(np.log(self.basis_tau),
                                                                  basis_type=self.tau_basis_type,
                                                                  order=k, epsilon=self.tau_epsilon,
                                                                  zga_params=self.zga_params,
                                                                  integration_limits=integration_limits)
                penalty_matrices[f'm{k}'] = dk.copy()

            # ---- DOP penalty (independent of fit_drt) ----
            if self.fit_dop:
                if penalty_type == 'discrete':
                    # Use Gaussian derivatives for delta basis
                    dnu = np.median(np.diff(np.sort(self.basis_nu)))
                    dk_dop = basis.construct_func_eval_matrix(self.basis_nu, None,
                                                              basis_type='gaussian',
                                                              order=k, epsilon=1 / dnu)
                    penalty_matrices[f'l{k}_dop'] = dk_dop.copy()
                    penalty_matrices[f'm{k}_dop'] = dk_dop.T @ dk_dop
                elif penalty_type == 'integral':
                    # TODO: truncate dop penalty?
                    if self.nu_basis_type == 'delta':
                        # Use Gaussian derivatives for delta basis
                        dnu = np.median(np.diff(np.sort(self.basis_nu)))
                        dk_dop = mat1d.construct_integrated_derivative_matrix(self.basis_nu,
                                                                              basis_type='gaussian',
                                                                              order=k, epsilon=1 / dnu)
                    else:
                        dk_dop = mat1d.construct_integrated_derivative_matrix(self.basis_nu,
                                                                              basis_type=self.nu_basis_type,
                                                                              order=k, epsilon=self.nu_epsilon)
                    penalty_matrices[f'm{k}_dop'] = dk_dop.copy()

                    # Make a smoothing matrix for the 0th-derivative s vector
                    if k == 0:
                        dnu = np.mean(np.abs(np.diff(self.basis_nu)))
                        gmat = mat1d.construct_integrated_derivative_matrix(self.basis_nu,
                                                                            basis_type='gaussian',
                                                                            order=1, epsilon=1 / dnu)
                        penalty_matrices[f'gmat{k}_dop'] = gmat.copy()

            # ---- DDT penalty (independent of fit_drt) ----
            if self.fit_ddt and penalty_type == 'integral':
                # Integrated derivative penalty on the log-diffusion-time grid.
                # The integrated-derivative matrix is only implemented for the
                # Gaussian basis, which is used regardless of tau_dd_basis_type.
                dln_tau_dd = np.mean(np.diff(np.log(self.basis_tau_dd)))
                if self.tau_dd_epsilon is not None:
                    ddt_eps = self.tau_dd_epsilon
                else:
                    ddt_eps = 1 / dln_tau_dd
                m_tau = mat1d.construct_integrated_derivative_matrix(np.log(self.basis_tau_dd),
                                                                     basis_type='gaussian',
                                                                     order=k, epsilon=ddt_eps)

                if self.basis_p is None:
                    # 1D DDT: penalty only over log(tau_D)
                    dk_ddt = m_tau
                else:
                    # 2D DDT: separable penalty over log(tau_D) and p,
                    # M = M_tau (x) I_p + c_p * I_tau (x) M_p
                    n_tau = len(self.basis_tau_dd)
                    n_p = len(self.basis_p)
                    dp = np.mean(np.diff(self.basis_p))
                    m_p = mat1d.construct_integrated_derivative_matrix(self.basis_p,
                                                                       basis_type='gaussian',
                                                                       order=k, epsilon=1 / dp)
                    c_p = self.ddt_p_penalty_weight
                    dk_ddt = np.kron(m_tau, np.eye(n_p)) + c_p * np.kron(np.eye(n_tau), m_p)
                penalty_matrices[f'm{k}_ddt'] = dk_ddt.copy()

                # Make a smoothing matrix for the 0th-derivative s vector
                if k == 0:
                    m_tau_1 = mat1d.construct_integrated_derivative_matrix(np.log(self.basis_tau_dd),
                                                                            basis_type='gaussian',
                                                                            order=1, epsilon=ddt_eps)
                    if self.basis_p is None:
                        gmat = m_tau_1
                    else:
                        n_tau = len(self.basis_tau_dd)
                        n_p = len(self.basis_p)
                        dp = np.mean(np.diff(self.basis_p))
                        m_p_1 = mat1d.construct_integrated_derivative_matrix(self.basis_p,
                                                                             basis_type='gaussian',
                                                                             order=1, epsilon=1 / dp)
                        c_p = self.ddt_p_penalty_weight
                        gmat = np.kron(m_tau_1, np.eye(n_p)) + c_p * np.kron(np.eye(n_tau), m_p_1)
                    penalty_matrices[f'gmat{k}_ddt'] = gmat.copy()

        self.fit_matrices.update(penalty_matrices)

        if self.print_diagnostics:
            print('Constructed penalty matrices')
        return penalty_matrices

    def _format_qp_matrices(self, rm_drt, inf_rv, induc_rv, cap_rv, rm_dop, rm_ddt,
                            zm_drt, induc_zv, cap_zv, zm_dop, zm_ddt,
                            base_penalty_matrices, v_baseline_penalty, ohmic_penalty,
                            inductance_penalty, capacitance_penalty,
                            vz_offset_scale, background_penalty,
                            inductance_scale, capacitance_scale, penalty_type, derivative_weights):
        """
        Format matrices for quadratic programming solution
        :param capacitance_scale:
        :param capacitance_penalty:
        :param cap_rv:
        :param cap_zv:
        :param background_penalty:
        :param derivative_weights:
        :param rm_drt:
        :param base_penalty_matrices:
        :param inductance_scale:
        :param penalty_type:
        :return:
        """
        # Count number of special params for padding
        num_special = self.get_qp_mat_offset()

        # Extract indices for convenience
        special_indices = {k: self.special_qp_params[k]['index'] for k in self.special_qp_params.keys()}

        # Store inductance scale for reference
        self.inductance_scale = inductance_scale
        self.capacitance_scale = capacitance_scale

        # Get DOP scale vector
        if self.fit_dop:
            if self.normalize_dop and self.basis_tau is not None:
                # If tau_supergrid is set, we should use this for DOP normalization
                # to ensure consistency across spectra (e.g. for DRTMD).
                # Otherwise, we can just use basis_tau
                if self.tau_supergrid is not None:
                    dop_eval_tau = self.tau_supergrid
                else:
                    dop_eval_tau = self.basis_tau
                
                self.dop_scale_vector = phasance.phasor_scale_vector(self.basis_nu, dop_eval_tau)
                
                # Normalize for basis function area
                # print('nu basis area:', basis.get_basis_func_area(self.nu_basis_type, self.nu_epsilon))
                self.dop_scale_vector /= self.nu_basis_area
                
                # self.dop_scale_vector *= 100
                # print(self.dop_scale_vector)
            else:
                self.dop_scale_vector = np.ones(len(self.basis_nu))
        else:
            self.dop_scale_vector = None
        dop_start_index, dop_end_index = self.dop_indices
        ddt_start_index, ddt_end_index = self.ddt_indices

        # Add columns to rm for v_baseline, R_inf, and inductance
        if rm_drt is not None:
            rm = np.empty((rm_drt.shape[0], rm_drt.shape[1] + num_special))

            # Add entries for special parameters
            if 'v_baseline' in special_indices.keys():
                vb_start, vb_end = self.get_special_indices("v_baseline")
                rm[:, vb_start:vb_end], vbase_scale = background.get_baseline_matrix(self.get_fit_times(), self.v_baseline_deg, normalize=True, sqrt=self.v_baseline_sqrt)
                self.v_baseline_scale = vbase_scale
            if 'inductance' in special_indices.keys():
                rm[:, special_indices['inductance']] = induc_rv * inductance_scale  # inductance response
            if 'R_inf' in special_indices.keys():
                rm[:, special_indices['R_inf']] = inf_rv  # R_inf response. only works for galvanostatic mode
            if 'C_inv' in special_indices.keys():
                rm[:, special_indices['C_inv']] = cap_rv * capacitance_scale

            # Add entry for vz_offset if applicable
            if 'vz_offset' in special_indices.keys():
                rm[:, special_indices['vz_offset']] = 0

            if 'background_scale' in special_indices.keys():
                rm[:, special_indices['background_scale']] = 0

            # Add DOP
            if self.fit_dop:
                rm[:, dop_start_index:dop_end_index] = rm_dop * self.dop_scale_vector

            # Add DDT (EIS only - rm_ddt is always None, so no chrono contribution)
            if self.fit_ddt:
                if rm_ddt is not None:
                    rm[:, ddt_start_index:ddt_end_index] = rm_ddt * self.ddt_scale_vector
                else:
                    rm[:, ddt_start_index:ddt_end_index] = 0

            # Insert main DRT matrix
            rm[:, num_special:] = rm_drt
        else:
            # No chrono matrix
            rm = None

        # Add columns to zm (impedance matrix) for v_baseline, R_inf, and inductance
        if zm_drt is not None:
            zm = np.zeros((zm_drt.shape[0], zm_drt.shape[1] + num_special), dtype=complex)

            # Add entries for special parameters
            # Unnecessary with initialization as np.zeros
            # if 'v_baseline' in special_indices.keys():
            #     zm[:, special_indices['v_baseline']] = 0  # v_baseline has no effect on impedance

            if 'inductance' in special_indices.keys():
                zm[:, special_indices['inductance']] = induc_zv * inductance_scale  # inductance contribution

            if 'R_inf' in special_indices.keys():
                zm[:, special_indices['R_inf']] = 1  # R_inf contribution to impedance

            if 'C_inv' in special_indices.keys():
                zm[:, special_indices['C_inv']] = cap_zv * capacitance_scale

            # Add entry for vz_offset if applicable
            if 'vz_offset' in special_indices.keys():
                zm[:, special_indices['vz_offset']] = 0

            if 'background_scale' in special_indices.keys():
                zm[:, special_indices['background_scale']] = 0

            # Add DOP
            if self.fit_dop:
                zm[:, dop_start_index:dop_end_index] = zm_dop * self.dop_scale_vector

            # Add DDT
            if self.fit_ddt:
                zm[:, ddt_start_index:ddt_end_index] = zm_ddt * self.ddt_scale_vector

            # Insert main DRT matrix
            zm[:, num_special:] = zm_drt  # DRT response

            # Stack real and imag matrices
            zm = np.vstack([zm.real, zm.imag])
        else:
            # No EIS matrix
            zm = None

        # Construct L2 penalty matrices
        penalty_matrices = {}
        if penalty_type == 'integral':
            for k in range(len(derivative_weights)):
                # Get penalty matrix for DRT coefficients
                m_drt = base_penalty_matrices[f'm{k}']

                # Add rows/columns for v_baseline, inductance, and R_inf
                m_k = np.zeros((m_drt.shape[0] + num_special, m_drt.shape[1] + num_special))

                # Insert penalties for special parameters
                if 'v_baseline' in special_indices.keys():
                    vb_start, vb_end = self.get_special_indices("v_baseline")
                    if not np.isscalar(v_baseline_penalty):
                        if len(v_baseline_penalty) != vb_end - vb_start:
                            raise ValueError(
                                "If v_baseline_penalty is iterable, it must match the number of "
                                f"v_baseline parameters. Number of v_baseline parameters is {vb_end - vb_start}"
                            )
                        for i, vbi in enumerate(range(vb_start, vb_end)):
                            m_k[vbi, vbi] = v_baseline_penalty[i]
                    else:
                        for vbi in range(vb_start, vb_end):
                            m_k[vbi, vbi] = v_baseline_penalty
                if 'inductance' in special_indices.keys():
                    m_k[special_indices['inductance'], special_indices['inductance']] = inductance_penalty
                if 'R_inf' in special_indices.keys():
                    m_k[special_indices['R_inf'], special_indices['R_inf']] = ohmic_penalty
                if 'C_inv' in special_indices.keys():
                    m_k[special_indices['C_inv'], special_indices['C_inv']] = capacitance_penalty

                # Add entry for vz_offset if applicable
                if 'vz_offset' in special_indices.keys():
                    m_k[special_indices['vz_offset'], special_indices['vz_offset']] = 1 / vz_offset_scale

                # Add entry for background_scale if applicable
                if 'background_scale' in special_indices.keys():
                    m_k[special_indices['background_scale'], special_indices['background_scale']] = background_penalty

                # Add DOP matrix
                if self.fit_dop:
                    m_dop = base_penalty_matrices.get(f'm{k}_dop', 0)
                    m_k[dop_start_index:dop_end_index, dop_start_index:dop_end_index] = m_dop
                    # penalty_matrices[f'm{k}_dop'] = m_dop.copy()

                # Add DDT matrix
                if self.fit_ddt:
                    m_ddt = base_penalty_matrices.get(f'm{k}_ddt', 0)
                    m_k[ddt_start_index:ddt_end_index, ddt_start_index:ddt_end_index] = m_ddt

                # Insert main DRT matrix
                m_k[num_special:, num_special:] = m_drt

                penalty_matrices[f'm{k}'] = m_k.copy()
                # penalty_matrices[f'm{k}_drt'] = m_drt.copy()
        elif penalty_type == 'discrete':
            warnings.warn("penalty_type 'discrete' is deprecated and not supported; errors may occur."
                          " Use penalty_type 'integral' instead")
            if self.fit_dop:
                raise ValueError('DOP fit not implemented with discrete penalty')

            for k in range(len(derivative_weights)):
                # Get penalty matrix for DRT coefficients
                l_drt = base_penalty_matrices[f'l{k}']

                # Add rows/columns for v_baseline, inductance, and R_inf
                l_k = np.zeros((l_drt.shape[0] + num_special, l_drt.shape[1] + num_special))

                # Insert penalties for special parameters
                if 'v_baseline' in special_indices.keys():
                    l_k[special_indices['v_baseline'], special_indices['v_baseline']] = v_baseline_penalty ** 0.5
                if 'inductance' in special_indices.keys():
                    l_k[special_indices['inductance'], special_indices['inductance']] = inductance_penalty ** 0.5
                if 'R_inf' in special_indices.keys():
                    l_k[special_indices['R_inf'], special_indices['R_inf']] = ohmic_penalty ** 0.5
                if 'C_inv' in special_indices.keys():
                    l_k[special_indices['C_inv'], special_indices['C_inv']] = capacitance_penalty ** 0.5

                # Add entry for vz_offset if applicable
                if 'vz_offset' in special_indices.keys():
                    l_k[special_indices['vz_offset'], special_indices['vz_offset']] = 1 / vz_offset_scale ** 0.5

                # Add entry for background_scale if applicable
                if 'background_scale' in special_indices.keys():
                    l_k[special_indices['background_scale'], special_indices[
                        'background_scale']] = background_penalty ** 0.5

                # Add DOP matrix
                if self.fit_dop:
                    l_dop = base_penalty_matrices.get(f'l{k}_dop', 0)
                    l_k[dop_start_index:dop_end_index] = l_dop

                # Insert main DRT matrix
                l_k[num_special:, num_special:] = l_drt

                penalty_matrices[f'l{k}'] = l_k.copy()

                # Calculate norm matrix
                penalty_matrices[f'm{k}'] = l_k.T @ l_k

        self._qp_matrices = {
            'rm': rm,
            'zm': zm,
            'penalty_matrices': penalty_matrices
        }

        return rm, zm, penalty_matrices

    # def _prep_for_hybrid_fit(self, times, i_signal, v_signal, frequencies, z, step_times, downsample, downsample_kw,
    #                          scale_signal, offset_steps, penalty_type, smooth_inf_response):
    #
    #     # Perform preprocessing and matrix preparation for regular fit
    #     sample_data, matrices = self._prep_for_fit(times, i_signal, v_signal, None, None, step_times, downsample,
    #                                                downsample_kw, offset_steps, smooth_inf_response, scale_signal, 7,
    #                                                penalty_type, derivative_weights)
    #
    #     # Extract time response data from sample_data
    #     sample_times, i_signal_scaled, v_signal_scaled, response_baseline = sample_data
    #
    #     # Extract time response matrices and penalty matrics
    #     rm, induc_rv, inf_rv, penalty_matrices = matrices
    #
    #     # Scale measured impedance
    #     z_scaled = self.scale_impedance(z)
    #
    #     self.z_fit = z.copy()
    #     self.z_fit_scaled = z_scaled.copy()
    #
    #     # Construct impedance matrix
    #     zm = self._prep_impedance_fit_matrix(frequencies)
    #
    #     # Construct inductance impedance vector
    #     induc_zv = mat1d.construct_inductance_z_vector(frequencies)
    #
    #     return (sample_times, i_signal_scaled, v_signal_scaled, response_baseline, z_scaled), \
    #            (rm, zm, induc_rv, inf_rv, induc_zv, penalty_matrices)

    def _prep_chrono_prediction_matrix(self, times, input_signal, step_times, step_sizes,
                                       op_mode, offset_steps, step_offset_size, smooth_inf_response):
        # TODO: incorporate step_times and step_sizes logic
        #  This is a placeholder to avoid issues with corner cases
        #  (e.g. if input_signal remains the same but step_times is manually specified)
        self._recalc_chrono_prediction_matrix = True

        # Validate input signal
        if input_signal is not None and step_times is not None:
            raise ValueError('Either input_signal OR (step_times and step_sizes) should be provided; '
                             'received input_signal and step_times')
        elif step_times is not None and step_sizes is None:
            raise ValueError('If input signal steps are provided, both step_times and step_sizes must be provided; '
                             'received step_times only')

        # If input signal is not provided, use self.raw_input_signal
        if input_signal is None and step_times is None:
            if self._t_fit_subset_index is not None:
                input_signal = self.raw_input_signal[self._t_fit_subset_index]
            else:
                input_signal = self.raw_input_signal
            use_fit_signal = True
        else:
            use_fit_signal = False

        # If signal steps are provided instead of input_signal, create dummy signal for validation
        if step_times is not None:
            input_signal = pp.generate_model_signal(times, step_times, step_sizes, None, 'ideal')

        self.t_predict = times
        self.raw_prediction_input_signal = input_signal
        self.chrono_mode_predict = op_mode
        # Final update of t_predict in case recalc status changed
        self.t_predict = times

        if self.print_diagnostics:
            print('recalc_response_prediction_matrix:', self._recalc_chrono_prediction_matrix)

        # Process signal to identify steps
        if use_fit_signal:
            # Allow times to have a different length than input_signal if using fitted signal (input_signal = None)
            # This will break construct_inf_response_vector if smooth_inf_response==False
            step_times = self.step_times
            step_sizes = self.step_sizes
            tau_rise = self.tau_rise
        else:
            if step_times is None:
                # Identify steps in applied signal
                step_times, step_sizes, tau_rise = pp.process_input_signal(times, input_signal, self.step_model,
                                                                           offset_steps, step_offset_size)
            else:
                tau_rise = None

        if self._recalc_chrono_prediction_matrix:
            # Matrix recalculation is required
            rm, rm_layered = mat1d.construct_response_matrix(self.basis_tau, times, self.step_model, step_times,
                                                             step_sizes, basis_type=self.tau_basis_type,
                                                             epsilon=self.tau_epsilon, tau_rise=tau_rise,
                                                             op_mode=op_mode,
                                                             integrate_method=self.integrate_method,
                                                             zga_params=self.zga_params,
                                                             interpolate_grids=self.interpolate_lookups['response'])

            induc_rv = mat1d.construct_inductance_response_vector(times, self.step_model, step_times, step_sizes,
                                                                  tau_rise, op_mode)
            cap_rv = mat1d.construct_capacitance_response_vector(times, self.step_model, step_times, step_sizes,
                                                                 tau_rise, op_mode)

            self.prediction_matrices.update({
                'response': rm.copy(),
                'inductance_response': induc_rv.copy(),
                'capacitance_response': cap_rv.copy(),
            })

            # With prediction matrices calculated, set recalc flag to False
            self._recalc_chrono_prediction_matrix = False
            if self.print_diagnostics:
                print('Calculated response prediction matrices')
        elif self._t_predict_eq_t_fit:
            print('eq')
            # times is the same as self.t_fit. Do not overwrite
            rm = self.fit_matrices['response'].copy()
            induc_rv = self.fit_matrices['inductance_response'].copy()
            cap_rv = self.fit_matrices['capacitance_response'].copy()
        elif self._t_predict_subset_index[0] == 'predict':
            # times is a subset of self.t_predict. Use sub-matrices of existing matrix; do not overwrite
            rm = self.prediction_matrices['response'][self._t_predict_subset_index[1], :].copy()
            induc_rv = self.prediction_matrices['inductance_response'][self._t_predict_subset_index[1]].copy()
            cap_rv = self.prediction_matrices['capacitance_response'][self._t_predict_subset_index[1]].copy()
        elif self._t_predict_subset_index[0] == 'fit':
            print('fit')
            # times is a subset of self.t_fit. Use sub-matrices of existing matrix; do not overwrite
            rm = self.fit_matrices['response'][self._t_predict_subset_index[1], :].copy()
            induc_rv = self.fit_matrices['inductance_response'][self._t_predict_subset_index[1]].copy()
            cap_rv = self.fit_matrices['capacitance_response'][self._t_predict_subset_index[1]].copy()
        else:
            # All matrix parameters are the same. Use existing matrix
            print('predict')
            rm = self.prediction_matrices['response'].copy()
            induc_rv = self.prediction_matrices['inductance_response'].copy()
            cap_rv = self.prediction_matrices['capacitance_response'].copy()

        # Construct R_inf response vector
        inf_rv = mat1d.construct_ohmic_response_vector(times, self.step_model, step_times, step_sizes,
                                                       tau_rise, input_signal, smooth_inf_response,
                                                       op_mode)

        self.prediction_matrices['inf_response'] = inf_rv.copy()

        # Construct DOP matrix
        if self.fit_dop:
            rm_dop, _ = phasance.construct_phasor_v_matrix(times, self.basis_nu, self.nu_basis_type,
                                                           self.nu_epsilon, self.step_model, step_times,
                                                           step_sizes, op_mode)
            self.prediction_matrices['rm_dop'] = rm_dop.copy()
        else:
            rm_dop = None

        if self.series_neg:
            rm = np.hstack((rm, -rm))

        return rm, induc_rv, inf_rv, cap_rv, rm_dop

    def _prep_impedance_prediction_matrix(self, frequencies):
        self.f_predict = frequencies
        if not self.fit_drt:
            zm = np.empty((len(frequencies), 0), dtype=complex)
            self.prediction_matrices['impedance'] = zm.copy()
            self._recalc_eis_prediction_matrix = False
        elif self._recalc_eis_prediction_matrix:
            # Matrix calculation is required
            # Real matrix
            zmr = mat1d.construct_impedance_matrix(frequencies, 'real', tau=self.basis_tau,
                                                   basis_type=self.tau_basis_type, epsilon=self.tau_epsilon,
                                                   frequency_precision=self.frequency_precision,
                                                   zga_params=self.zga_params, integrate_method=self.integrate_method,
                                                   interpolate_grids=self.interpolate_lookups['z_real'])
            # Imaginary matrix
            zmi = mat1d.construct_impedance_matrix(frequencies, 'imag', tau=self.basis_tau,
                                                   basis_type=self.tau_basis_type, epsilon=self.tau_epsilon,
                                                   frequency_precision=self.frequency_precision,
                                                   zga_params=self.zga_params, integrate_method=self.integrate_method,
                                                   interpolate_grids=self.interpolate_lookups['z_imag'])
            # Complex matrix
            zm = zmr + 1j * zmi
            self.prediction_matrices['impedance'] = zm.copy()

            # With prediction matrix calculated, set recalc flag to False
            self._recalc_eis_prediction_matrix = False
        else:
            try:
                if self._f_predict_eq_f_fit:
                    # frequencies is same as self.f_fit. Use existing fit matrix; do not overwrite
                    zm = self.fit_matrices['impedance'].copy()
                elif self._f_predict_subset_index[0] == 'fit':
                    # frequencies is a subset of self.f_fit. Use sub-matrices of existing matrix; do not overwrite
                    zm = self.fit_matrices['impedance'][self._f_predict_subset_index[1], :].copy()
                elif self._f_predict_subset_index[0] == 'predict':
                    # frequencies is a subset of self.f_predict. Use sub-matrices of existing matrix; do not overwrite
                    zm = self.prediction_matrices['impedance'][self._f_predict_subset_index[1], :].copy()
                else:
                    # All matrix parameters are the same. Use existing matrix
                    zm = self.prediction_matrices['impedance'].copy()
            except KeyError:
                # The matrix we're trying to reuse doesn't exist. Recalculate explicitly
                # This can happen if the matrix was deleted or if the DRT instance was loaded from file
                self._recalc_eis_prediction_matrix = True
                zm, _ = self._prep_impedance_prediction_matrix(frequencies)
                self._f_predict_eq_f_fit = False
                self._f_predict_subset_index = ('', [])

        if self.fit_dop:
            zm_dop = phasance.construct_phasor_z_matrix(frequencies, self.basis_nu,
                                                        self.nu_basis_type, self.nu_epsilon)
            self.prediction_matrices['zm_dop'] = zm_dop.copy()
        else:
            zm_dop = None

        if self.fit_ddt:
            zm_ddt = ddt.get_ddt_impedance_matrix(frequencies, self.basis_tau_dd,
                                                  self.ddt_kernel_type, basis_type=self.tau_dd_basis_type,
                                                  ddt_epsilon=self.tau_dd_epsilon, p=self.ddt_p,
                                                  p_grid=self.basis_p, p_epsilon=self.ddt_p_epsilon)
            self.prediction_matrices['zm_ddt'] = zm_ddt.copy()
        else:
            zm_ddt = None

        if self.series_neg:
            zm = np.hstack((zm, -zm))

        return zm, zm_dop, zm_ddt

    def _get_vz_strength_vec(self, times=None, frequencies=None, fit_times=None, step_times=None,
                             fit_frequencies=None, vz_offset_eps=1):
        if fit_times is None:
            fit_times = self.get_fit_times(True)

        if step_times is None:
            step_times = self.nonconsec_step_times

        if fit_frequencies is None:
            fit_frequencies = self.get_fit_frequencies(True)

        if fit_times is None or fit_frequencies is None or vz_offset_eps is None:
            if times is not None:
                chrono_vz_strength = np.ones(len(times))
            else:
                chrono_vz_strength = None

            if frequencies is not None:
                eis_vz_strength = np.ones(len(frequencies))
            else:
                eis_vz_strength = None

            return chrono_vz_strength, eis_vz_strength
        else:
            # VZ offset decays as we move away from overlap
            rbf = basis.get_basis_func('gaussian')

            # Get effective timescale of each data point to determine data limits
            fit_time_deltas = pp.get_time_since_step(fit_times, step_times, prestep_value=-1)
            chrono_tau_min = np.min(fit_time_deltas[fit_time_deltas > 0])
            eis_tau_max = np.max(1 / (2 * np.pi * fit_frequencies))

            # Make vz_offset strength vectors
            # Strength is 1 in overlapped region and then decays in log space
            if times is not None:
                time_deltas = pp.get_time_since_step(times, step_times, prestep_value=-1)
                chrono_vz_strength = np.ones(len(time_deltas))
                chrono_vz_strength[(time_deltas >= eis_tau_max)] = rbf(
                    np.log(time_deltas[time_deltas >= eis_tau_max] / eis_tau_max), vz_offset_eps
                )
                chrono_vz_strength[time_deltas == -1] = 0
            else:
                chrono_vz_strength = None

            if frequencies is not None:
                f_inv = 1 / (2 * np.pi * frequencies)
                eis_vz_strength = np.ones(len(frequencies))
                eis_vz_strength[f_inv <= chrono_tau_min] = rbf(
                    np.log(f_inv[f_inv <= chrono_tau_min] / chrono_tau_min), vz_offset_eps
                )
            else:
                eis_vz_strength = None

            return chrono_vz_strength, eis_vz_strength

    def extract_qphb_parameters(self, x, rescale=True):
        def scale_value(value, scale):
            if rescale:
                return value * scale
            else:
                return scale
            
        special_indices = {k: self.special_qp_params[k]['index'] for k in self.special_qp_params.keys()}
        fit_parameters = {'x': scale_value(x[self.get_qp_mat_offset():], self.coefficient_scale)}

        if 'R_inf' in special_indices.keys():
            fit_parameters['R_inf'] = scale_value(x[special_indices['R_inf']], self.coefficient_scale)
        else:
            fit_parameters['R_inf'] = 0

        if 'v_baseline' in special_indices.keys():
            vb_start, vb_end = self.get_special_indices("v_baseline")
            
            vbx_scaled = x[vb_start:vb_end].copy()
            
            # Remove the columnwise normalization
            vbx_scaled = scale_value(vbx_scaled, 1. / self.v_baseline_scale)
            
            # Subtract the scaled pre-offset from the 0-degree coef 
            vbx_scaled[0] -= self.scaled_response_offset
            
            # Remove the data scaling
            fit_parameters['v_baseline'] = scale_value(
                vbx_scaled,
                self.response_signal_scale
            )

        if 'vz_offset' in special_indices.keys():
            fit_parameters['vz_offset'] = x[special_indices['vz_offset']]

        if 'inductance' in special_indices.keys():
            fit_parameters['inductance'] = scale_value(
                x[special_indices['inductance']],
                self.coefficient_scale * self.inductance_scale
            )
        else:
            fit_parameters['inductance'] = 0

        if 'C_inv' in special_indices.keys():
            fit_parameters['C_inv'] = scale_value(
                x[special_indices['C_inv']], 
                self.coefficient_scale * self.capacitance_scale
            )
        else:
            fit_parameters['C_inv'] = 0

        if 'background_scale' in special_indices.keys():
            fit_parameters['background_scale'] = x[special_indices['background_scale']]

        if self.fit_dop:
            dop_start, dop_end = self.dop_indices
            fit_parameters['x_dop'] = scale_value(
                x[dop_start:dop_end],
                self.dop_scale_vector * self.coefficient_scale
            )

        if self.fit_ddt:
            ddt_start, ddt_end = self.ddt_indices
            fit_parameters['x_ddt'] = scale_value(
                x[ddt_start:ddt_end],
                self.ddt_scale_vector * self.coefficient_scale
            )
            fit_parameters['ddt_p'] = self.ddt_p

        return fit_parameters

    def estimate_chrono_background(self, times, i_signal, v_signal, bkg_iter=1,
                                   gp=None, kernel_type='gaussian', length_scale_bounds=(0.01, 10),
                                   periodicity_bounds=(1e-3, 1e3), noise_level_bounds=(0.1, 10),
                                   kernel_size=1, n_restarts=1, kernel_scale_factor=1, y_err_thresh=1e-3,
                                   linear_downsample=True, linear_sample_interval=None, copy_self=False,
                                   **fit_kw):
        fit_defaults = {'max_iter': 10, 'error_structure': None}
        fit_kw = dict(fit_defaults, **fit_kw)

        # Make a copy to avoid overwriting any attributes during fit
        if copy_self:
            drt_bkg = deepcopy(self)
        else:
            drt_bkg = self
        gps, y_bkg = background.estimate_chrono_background(drt_bkg, times, i_signal, v_signal, max_iter=bkg_iter,
                                                           gp=gp, kernel_type=kernel_type,
                                                           length_scale_bounds=length_scale_bounds,
                                                           periodicity_bounds=periodicity_bounds,
                                                           noise_level_bounds=noise_level_bounds,
                                                           kernel_size=kernel_size, n_restarts=n_restarts,
                                                           kernel_scale_factor=kernel_scale_factor,
                                                           y_err_thresh=y_err_thresh,
                                                           linear_downsample=linear_downsample,
                                                           linear_sample_interval=linear_sample_interval, fit_kw=fit_kw)

        if copy_self:
            return drt_bkg, gps, y_bkg
        else:
            return gps, y_bkg

    # -------------------------------------------------------
    # Attribute management for storing/loading fit parameters
    # -------------------------------------------------------
    @property
    def attribute_categories(self):
        att = {
            'config': [
                'fixed_basis_tau', 'basis_tau', 'tau_basis_type', 'tau_epsilon', 'tau_supergrid',
                'fixed_basis_nu', 'basis_nu', 'nu_basis_type', 'nu_epsilon',
                'fixed_basis_tau_dd', 'basis_tau_dd', 'tau_dd_basis_type', 'tau_dd_epsilon',
                'fit_ddt', 'normalize_ddt', 'ddt_kernel_type',
                'ddt_p', 'fit_ddt_p', 'ddt_p_bounds',
                'ddt_p_grid', 'fit_ddt_2d', 'ddt_p_penalty_weight', 'basis_p', 'ddt_p_epsilon',
                'series_neg', 'fit_dop', 'normalize_dop', 'fit_inductance',
                'step_model', 'chrono_mode', "v_baseline_deg", "v_baseline_sqrt",
                'frequency_precision', 'time_precision', 'input_signal_precision',
                'integrate_method'
            ],
            'fit_core': [
                'fit_parameters', 'fit_type', 'fit_kwargs',
                'special_qp_params',
                # These are necessary for extract_qphb_parameters
                'coefficient_scale', 'inductance_scale', 'capacitance_scale', 'input_signal_scale', 'response_signal_scale',
                'scaled_response_offset', 'impedance_scale', 'dop_scale_vector', "v_baseline_scale",
            ],
            'fit_detail': [
                'qphb_params', 'cvx_result', 'qphb_history', 'interpolate_lookups', 'fit_matrices'
            ],
            'data': [
                't_fit', 'raw_input_signal', 'raw_response_signal', 'scaled_input_signal', 'scaled_response_signal',
                'raw_response_background', 'step_times', 'nonconsec_step_times', 'step_sizes', 'tau_rise',
                'f_fit', 'z_fit', 'z_fit_scaled',
                'chrono_outlier_index', 'chrono_outliers', 'eis_outlier_index', 'eis_outliers',
                '_t_fit_subset_index', '_f_fit_subset_index'
            ],
        }

        return att

    def get_attributes(self, which):
        att_names = None
        if type(which) == str:
            if which == 'all':
                att_names = sum(list(self.attribute_categories.values()), [])
            else:
                which = [which]

        if att_names is None:
            try:
                att_names = sum([self.attribute_categories[c] for c in which], [])
            except KeyError:
                raise ValueError('which argument contains an invalid attribute category. '
                                 'Valid categories: {}'.format(['all'] + list(self.attribute_categories.keys())))

        return {k: deepcopy(getattr(self, k)) for k in att_names}

    def set_attributes(self, att_dict):
        for k, v in att_dict.items():
            setattr(self, k, deepcopy(v))

    def save_attributes(self, which, dest):
        att_dict = self.get_attributes(which)
        with open(dest, 'wb') as f:
            pickle.dump(att_dict, f, pickle.DEFAULT_PROTOCOL)

    def load_attributes(self, source):
        with open(source, 'rb') as f:
            att_dict = pickle.load(f)
        self.set_attributes(att_dict)

    def copy(self):
        return deepcopy(self)
