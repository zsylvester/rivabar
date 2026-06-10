"""
Migration prediction using the Howard and Knutson (1984) meander migration model.

Implements the convolution-based approach described in Sylvester et al. (2019,
Geology), where the predicted migration rate is computed from curvature using
an exponential downstream weighting function.

Key equations
-------------
Nominal migration rate (Eq. 2 in the supplementary material):
    R0(s) = kl * W * C(s)

Adjusted (predicted) migration rate (Eq. 3):
    R1(s) = Omega * R0(s) + Gamma * (integral of R0(s-xi)*G(xi) dxi)
                                   / (integral of G(xi) dxi)

where G(xi) = exp(-2 * Cf * xi / D) is the exponential weighting function,
D is river depth (estimated from width), and Cf is a friction factor.

Depth from width (Eq. 5, from Konsoer et al., 2013):
    W = 18.8 * D^1.41
    => D = (W / 18.8)^(1/1.41)

References
----------
Howard, A.D. and Knutson, T.R., 1984. Sufficient conditions for river
    meandering: A simulation approach. Water Resources Research, 20,
    p.1659-1667.
Sylvester, Z., Durkin, P. and Covault, J.A., 2019. High curvatures drive
    river meandering. Geology, 47(3), p.263-266.
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy import stats
from tqdm import tqdm


def depth_from_width(W):
    """
    Estimate river depth from channel width using the Konsoer et al. (2013)
    regression.

    W = 18.8 * D^1.41  =>  D = (W / 18.8)^(1/1.41)

    Parameters
    ----------
    W : float or array_like
        Channel width in metres.

    Returns
    -------
    D : float or ndarray
        Estimated depth in metres.
    """
    return (np.asarray(W) / 18.8) ** (1.0 / 1.41)


def nominal_migration_rate(curvature, W, kl):
    """
    Compute the nominal migration rate R0 = kl * W * C.

    Parameters
    ----------
    curvature : array_like
        Curvature along the channel (1/m).
    W : float
        Mean channel width in metres.
    kl : float
        Migration rate constant (erodibility), in m/yr.

    Returns
    -------
    R0 : ndarray
        Nominal migration rate (m/yr).
    """
    return -kl * W * np.asarray(curvature)


def predicted_migration_rate(R0, s, D, Cf, Omega=-1.0, Gamma=2.5):
    """
    Compute the predicted migration rate using the Howard and Knutson (1984)
    convolution model.

    R1(s) = Omega * R0(s) + Gamma * conv(R0, G) / integral(G)

    where G(xi) = exp(-2 * Cf * xi / D) and the convolution integrates
    upstream influence (causal, one-sided).

    Parameters
    ----------
    R0 : array_like
        Nominal migration rate along the channel.
    s : array_like
        Along-channel distance (m), must be uniformly spaced.
    D : float
        River depth in metres.
    Cf : float
        Friction factor (dimensionless); typical range 0.003-0.009.
    Omega : float, optional
        Weight for the local contribution (default -1.0).
    Gamma : float, optional
        Weight for the upstream-integrated contribution (default 2.5).
        Note: Omega + Gamma = 1.5 for the standard HK model, so that
        R1 = 1.5 * R0 for a bend with constant curvature.

    Returns
    -------
    R1 : ndarray
        Predicted migration rate (m/yr), same length as R0.
    """
    R0 = np.asarray(R0, dtype=float)
    s = np.asarray(s, dtype=float)
    ds = np.mean(np.diff(s))
    n = len(R0)

    # Build exponential kernel G(xi) = exp(-2*Cf*xi/D)
    # Truncate at 10 * D/(2*Cf) where the kernel is negligible
    if Cf > 0 and D > 0:
        decay_length = D / (2.0 * Cf)
        max_xi = min(10.0 * decay_length, (n - 1) * ds)
        n_kernel = max(int(np.ceil(max_xi / ds)), 1)
        xi = np.arange(n_kernel) * ds
        G = np.exp(-2.0 * Cf * xi / D)
    else:
        # No downstream influence: R1 = (Omega + Gamma) * R0
        return (Omega + Gamma) * R0

    # Normalized convolution: sum(R0 * G) / sum(G), where the denominator at
    # each point is the kernel mass actually available upstream of that point
    # (a true weighted average; near the upstream boundary fewer points exist,
    # so normalizing by the full kernel integral would attenuate the term and
    # flip the sign of R1 there)
    conv_full = np.convolve(R0, G, mode='full')[:n] * ds
    norm = np.convolve(np.ones(n), G, mode='full')[:n] * ds
    upstream_term = conv_full / norm

    R1 = Omega * R0 + Gamma * upstream_term
    return R1


def calibrate_Cf(R0, observed_mr, s, D, Cf_range=(0.001, 0.02),
                 Omega=-1.0, Gamma=2.5, mask=None):
    """
    Optimize the friction factor Cf to minimize the phase shift between
    predicted and observed migration rates.

    Parameters
    ----------
    R0 : array_like
        Nominal migration rate (from kl * W * C).
    observed_mr : array_like
        Observed migration rate (m/yr).
    s : array_like
        Along-channel distance (m).
    D : float
        Estimated river depth (m).
    Cf_range : tuple, optional
        (min, max) bounds for Cf search (default (0.001, 0.02)).
    Omega : float, optional
        Local weight (default -1.0).
    Gamma : float, optional
        Upstream weight (default 2.5).
    mask : array_like of bool, optional
        Points to use when scoring the correlation. The prediction is always
        computed on the full contiguous arrays (the convolution requires
        uniform spacing); the mask only restricts the statistics.

    Returns
    -------
    Cf_opt : float
        Optimal friction factor.
    r_opt : float
        Pearson correlation at optimal Cf.
    """
    R0 = np.asarray(R0, dtype=float)
    observed_mr = np.asarray(observed_mr, dtype=float)
    if mask is None:
        mask = np.ones(len(R0), dtype=bool)

    def neg_correlation(Cf):
        R1 = predicted_migration_rate(R0, s, D, Cf, Omega, Gamma)
        r = np.corrcoef(R1[mask], observed_mr[mask])[0, 1]
        return -r

    result = minimize_scalar(neg_correlation, bounds=Cf_range, method='bounded')
    Cf_opt = result.x
    r_opt = -result.fun
    return Cf_opt, r_opt


def calibrate_kl(predicted, observed, percentile=75):
    """
    Estimate the migration rate constant kl by minimizing the amplitude
    mismatch between predicted and observed migration rates.

    Uses the ratio of the specified percentile of absolute values, following
    the approach in Sylvester et al. (2019).

    Parameters
    ----------
    predicted : array_like
        Predicted migration rate (computed with kl=1).
    observed : array_like
        Observed migration rate (m/yr).
    percentile : float, optional
        Percentile of absolute values to match (default 75).

    Returns
    -------
    kl_scale : float
        Scale factor to apply to kl. If the initial kl was kl_init,
        the refined kl is kl_init * kl_scale.
    """
    predicted = np.asarray(predicted, dtype=float)
    observed = np.asarray(observed, dtype=float)
    p_pred = np.percentile(np.abs(predicted), percentile)
    p_obs = np.percentile(np.abs(observed), percentile)
    if p_pred > 0:
        return p_obs / p_pred
    return 1.0


def calibrate_from_curvature(curvature, observed_mr, s, W,
                             Cf_range=(0.001, 0.02),
                             Omega=-1.0, Gamma=2.5,
                             kl_percentile=75, mask=None):
    """
    Full calibration of D, Cf, and kl from curvature and observed migration
    rate for a single pair.

    Procedure (following Sylvester et al., 2019):
    1. Estimate D from W using the Konsoer et al. (2013) regression.
    2. Compute initial kl from R1 = 1.5 * R0 simplification.
    3. Optimize Cf to minimize the phase shift.
    4. Refine kl to match the amplitude.

    Parameters
    ----------
    curvature : array_like
        Curvature along the channel (1/m).
    observed_mr : array_like
        Observed migration rate (m/yr).
    s : array_like
        Along-channel distance (m).
    W : float
        Mean channel width (m).
    Cf_range : tuple, optional
        Bounds for Cf optimization (default (0.001, 0.02)).
    Omega : float, optional
        Local weight (default -1.0).
    Gamma : float, optional
        Upstream weight (default 2.5).
    kl_percentile : float, optional
        Percentile for amplitude matching (default 75).
    mask : array_like of bool, optional
        Points to use for the calibration statistics (correlations and
        percentiles). The prediction itself is always computed on the full
        contiguous arrays, since the convolution assumes uniform spacing.

    Returns
    -------
    result : dict
        Calibration results with keys:
        - 'D': estimated depth (m)
        - 'Cf': optimized friction factor
        - 'kl': calibrated migration rate constant (m/yr)
        - 'W': mean channel width (m)
        - 'Omega': local weight used
        - 'Gamma': upstream weight used
        - 'r_nominal': correlation between R0 and observed
        - 'r_predicted': correlation between R1 and observed
        - 'R0': nominal migration rate array
        - 'R1': predicted migration rate array
        - 'observed_mr': observed migration rate array
        - 's': along-channel distance array
    """
    curvature = np.asarray(curvature, dtype=float)
    observed_mr = np.asarray(observed_mr, dtype=float)
    s = np.asarray(s, dtype=float)
    if mask is None:
        mask = np.ones(len(curvature), dtype=bool)

    # Step 1: estimate depth
    D = depth_from_width(W)

    # Step 2: initial kl from R1 = 1.5 * R0
    # For constant curvature: R1 = (Omega + Gamma) * R0 = 1.5 * kl * W * C
    # Match the 75th percentile of |observed_mr| to 1.5 * kl * W * |C|_p75
    factor = Omega + Gamma  # should be 1.5
    p_obs = np.percentile(np.abs(observed_mr[mask]), kl_percentile)
    p_curv = np.percentile(np.abs(curvature[mask]), kl_percentile)
    if p_curv > 0 and W > 0 and abs(factor) > 0:
        kl_init = p_obs / (abs(factor) * W * p_curv)
    else:
        kl_init = 1.0

    # Compute nominal migration rate
    R0 = nominal_migration_rate(curvature, W, kl_init)
    r_nominal = np.corrcoef(R0[mask], observed_mr[mask])[0, 1]

    # Step 3: optimize Cf
    Cf_opt, r_predicted = calibrate_Cf(R0, observed_mr, s, D, Cf_range,
                                       Omega, Gamma, mask=mask)

    # Step 4: refine kl
    R1_unscaled = predicted_migration_rate(R0, s, D, Cf_opt, Omega, Gamma)
    kl_scale = calibrate_kl(R1_unscaled[mask], observed_mr[mask], kl_percentile)
    kl_final = kl_init * kl_scale

    # Recompute with final kl
    R0_final = nominal_migration_rate(curvature, W, kl_final)
    R1_final = predicted_migration_rate(R0_final, s, D, Cf_opt, Omega, Gamma)
    r_predicted_final = np.corrcoef(R1_final[mask], observed_mr[mask])[0, 1]

    return {
        'D': D,
        'Cf': Cf_opt,
        'kl': kl_final,
        'W': W,
        'Omega': Omega,
        'Gamma': Gamma,
        'r_nominal': r_nominal,
        'r_predicted': r_predicted_final,
        'R0': R0_final,
        'R1': R1_final,
        'observed_mr': observed_mr,
        's': s,
    }


def calibrate_pair(results, pair_idx, Cf_range=(0.001, 0.02),
                   Omega=-1.0, Gamma=2.5, kl_percentile=75,
                   use_stable_segments=True):
    """
    Calibrate the HK model for a single pair from analysis results.

    Parameters
    ----------
    results : dict
        Results dictionary from ``analyze_river_pairs_filtered`` or
        ``analyze_segment_group``.
    pair_idx : int
        Index of the pair to calibrate.
    Cf_range : tuple, optional
        Bounds for Cf optimization (default (0.001, 0.02)).
    Omega : float, optional
        Local weight (default -1.0).
    Gamma : float, optional
        Upstream weight (default 2.5).
    kl_percentile : float, optional
        Percentile for amplitude matching (default 75).
    use_stable_segments : bool, optional
        If True, only use stable segments (exclude high-variance regions)
        for calibration. The prediction is still computed everywhere.
        Default True.

    Returns
    -------
    result : dict
        Calibration results (see ``calibrate_from_curvature``), plus:
        - 'date1', 'date2': scene dates
        - 'time_gap_years': time gap between scenes
        - 'pair_idx': index in the results dict
    """
    pair_info = results['pair_info'][pair_idx]
    curvature = results['curvatures'][pair_idx]
    distances = results['migration_distances'][pair_idx]
    s = results['along_channel_distances'][pair_idx]
    time_gap_years = pair_info['time_gap_years']
    W = np.mean(pair_info['width1'])

    # Convert distances to migration rate
    observed_mr = distances / time_gap_years

    # Determine which points to use for the calibration statistics. The
    # prediction is always computed on the full contiguous arrays: slicing
    # out high-variance regions before the convolution would concatenate
    # spatially disjoint points and distort the upstream-influence kernel.
    mask = None
    if use_stable_segments:
        segment_results = pair_info.get('segment_results', [])
        if segment_results:
            # Build mask of stable segment points
            mask = np.zeros(len(curvature), dtype=bool)
            for seg in segment_results:
                mask[seg['start_idx']:seg['end_idx'] + 1] = True

    # Calibrate; statistics restricted to stable segments via the mask
    cal = calibrate_from_curvature(curvature, observed_mr, s, W,
                                   Cf_range, Omega, Gamma, kl_percentile,
                                   mask=mask)

    cal['date1'] = pair_info['date1']
    cal['date2'] = pair_info['date2']
    cal['time_gap_years'] = time_gap_years
    cal['pair_idx'] = pair_idx

    # Residual standard deviation (on stable segments only)
    residuals = observed_mr - cal['R1']
    if mask is not None:
        cal['residual_std'] = np.std(residuals[mask])
    else:
        cal['residual_std'] = np.std(residuals)

    return cal


def calibrate_segment(results, df, pair_class_filter='good',
                      Cf_range=(0.001, 0.02), Omega=-1.0, Gamma=2.5,
                      kl_percentile=75, use_stable_segments=True,
                      min_time_gap_years=0):
    """
    Calibrate the HK model for all qualifying pairs in a segment.

    Parameters
    ----------
    results : dict
        Results dictionary from ``analyze_river_pairs_filtered`` or
        ``analyze_segment_group``.
    df : pandas.DataFrame
        DataFrame from ``create_dataframe_from_results``, used to filter
        pairs by classification.
    pair_class_filter : str or list of str, optional
        Only calibrate pairs with these classifications (default 'good').
    Cf_range : tuple, optional
        Bounds for Cf optimization.
    Omega : float, optional
        Local weight (default -1.0).
    Gamma : float, optional
        Upstream weight (default 2.5).
    kl_percentile : float, optional
        Percentile for amplitude matching (default 75).
    use_stable_segments : bool, optional
        Whether to exclude high-variance segments during calibration
        (default True).
    min_time_gap_years : float, optional
        Minimum time gap (in years) for a pair to be included in
        calibration (default 0). Short-gap pairs tend to have noisier
        migration estimates, which can inflate kl.

    Returns
    -------
    result : dict
        Aggregated calibration results:
        - 'pair_calibrations': list of per-pair calibration dicts
        - 'D': estimated depth (same for all pairs, depends only on W)
        - 'Cf_median': median Cf across pairs
        - 'Cf_values': array of all Cf values
        - 'kl_median': median kl
        - 'kl_25': 25th percentile of kl
        - 'kl_75': 75th percentile of kl
        - 'kl_values': array of all kl values
        - 'r_predicted_median': median prediction correlation
        - 'r_predicted_values': array of all prediction correlations
        - 'n_pairs_calibrated': number of pairs used
        - 'W_mean': mean channel width across pairs
    """
    if isinstance(pair_class_filter, str):
        pair_class_filter = [pair_class_filter]

    # Find qualifying pairs
    qualifying = df.index[df['pair_class'].isin(pair_class_filter)].tolist()

    # Filter by minimum time gap
    if min_time_gap_years > 0:
        qualifying = [idx for idx in qualifying
                      if results['pair_info'][idx]['time_gap_years']
                      >= min_time_gap_years]

    pair_calibrations = []
    for idx in qualifying:
        try:
            cal = calibrate_pair(results, idx, Cf_range, Omega, Gamma,
                                 kl_percentile, use_stable_segments)
            pair_calibrations.append(cal)
        except Exception as e:
            print(f"  Pair {idx}: calibration failed - {e}")

    if not pair_calibrations:
        return {
            'pair_calibrations': [],
            'n_pairs_calibrated': 0,
        }

    Cf_values = np.array([c['Cf'] for c in pair_calibrations])
    kl_values = np.array([c['kl'] for c in pair_calibrations])
    r_values = np.array([c['r_predicted'] for c in pair_calibrations])
    W_values = np.array([c['W'] for c in pair_calibrations])
    residual_stds = np.array([c['residual_std'] for c in pair_calibrations])

    return {
        'pair_calibrations': pair_calibrations,
        'D': pair_calibrations[0]['D'],  # same for all (depends on mean W)
        'Cf_median': np.median(Cf_values),
        'Cf_values': Cf_values,
        'kl_median': np.median(kl_values),
        'kl_25': np.percentile(kl_values, 25),
        'kl_75': np.percentile(kl_values, 75),
        'kl_values': kl_values,
        'r_predicted_median': np.median(r_values),
        'r_predicted_values': r_values,
        'n_pairs_calibrated': len(pair_calibrations),
        'W_mean': np.mean(W_values),
        'Omega': Omega,
        'Gamma': Gamma,
        'residual_std': np.median(residual_stds),
    }


def _migrate_forward(x, y, W, D, Cf, kl, Omega, Gamma,
                     total_years, delta_s, cfl_factor=0.3):
    """
    Forward-integrate channel migration using the HK model over multiple
    small time steps (meanderpy-style).

    At each step: compute curvature from current geometry, predict migration
    rate, displace points along the local normal, then resample to maintain
    uniform spacing.

    Parameters
    ----------
    x, y : ndarray
        Initial centerline coordinates.
    W : float
        Mean channel width (m).
    D : float
        Estimated depth (m).
    Cf : float
        Friction factor.
    kl : float
        Migration rate constant (m/yr).
    Omega, Gamma : float
        HK model weights.
    total_years : float
        Total prediction time horizon (years).
    delta_s : float
        Target node spacing (m) for resampling.
    cfl_factor : float, optional
        Maximum displacement per step as a fraction of delta_s
        (default 0.3).

    Returns
    -------
    x, y : ndarray
        Predicted centerline coordinates after total_years.
    """
    from .utils import resample_and_smooth, compute_s_distance
    from scipy.signal import savgol_filter

    # Determine time step from CFL condition:
    # max(|MR|) * dt <= cfl_factor * delta_s
    # Initial estimate of max MR to set dt
    s = compute_s_distance(x, y)
    curv = _curvature_from_xy(x, y)
    R0 = nominal_migration_rate(curv, W, kl)
    R1 = predicted_migration_rate(R0, s, D, Cf, Omega, Gamma)
    max_mr = np.max(np.abs(R1))
    if max_mr > 0:
        dt = cfl_factor * delta_s / max_mr
    else:
        return x, y

    n_steps = max(int(np.ceil(total_years / dt)), 1)
    dt = total_years / n_steps

    for _ in range(n_steps):
        # Compute curvature from current geometry
        s = compute_s_distance(x, y)
        curv = _curvature_from_xy(x, y)

        # Predict migration rate
        R0 = nominal_migration_rate(curv, W, kl)
        R1 = predicted_migration_rate(R0, s, D, Cf, Omega, Gamma)

        # Compute tangent vectors normalized by arc length
        dx_ds = np.gradient(x) / np.gradient(s)
        dy_ds = np.gradient(y) / np.gradient(s)

        # Displace along normal
        x = x - R1 * dy_ds * dt
        y = y + R1 * dx_ds * dt

        # Resample to maintain uniform spacing
        x, y = resample_and_smooth(x, y, delta_s,
                                   smoothing_factor=0,
                                   compute_curvature=False)

    return x, y


def _migrate_forward_local(x, y, W, D, Cf, kl_s, s_kl, Omega, Gamma,
                           total_years, delta_s, cfl_factor=0.3):
    """
    Forward-integrate channel migration with spatially-varying kl.

    Same as ``_migrate_forward`` but accepts a kl(s) profile instead of
    a scalar.  At each time step the kl profile is re-interpolated onto
    the current (evolving) along-channel coordinate.

    Parameters
    ----------
    x, y : ndarray
        Initial centerline coordinates.
    W : float
        Mean channel width (m).
    D : float
        Estimated depth (m).
    Cf : float
        Friction factor.
    kl_s : ndarray
        Spatially-varying kl values.
    s_kl : ndarray
        Along-channel distances corresponding to *kl_s* (on the
        initial geometry).  Used as the reference for interpolation;
        assumed to be tied to along-channel position (not geographic
        coordinates).
    Omega, Gamma : float
        HK model weights.
    total_years : float
        Total prediction time horizon (years).
    delta_s : float
        Target node spacing (m) for resampling.
    cfl_factor : float, optional
        Maximum displacement per step as a fraction of delta_s
        (default 0.3).

    Returns
    -------
    x, y : ndarray
        Predicted centerline coordinates after total_years.
    """
    from .utils import resample_and_smooth, compute_s_distance

    kl_s = np.asarray(kl_s, dtype=float)
    s_kl = np.asarray(s_kl, dtype=float)

    # Normalise s_kl to [0, 1] for position-based interpolation
    s_kl_norm = (s_kl - s_kl[0]) / (s_kl[-1] - s_kl[0])

    # Initial estimate of max MR for CFL
    s = compute_s_distance(x, y)
    s_norm = (s - s[0]) / (s[-1] - s[0]) if s[-1] > s[0] else np.zeros_like(s)
    kl_interp = np.interp(s_norm, s_kl_norm, kl_s)
    curv = _curvature_from_xy(x, y)
    R0 = nominal_migration_rate(curv, W, kl_interp)
    R1 = predicted_migration_rate(R0, s, D, Cf, Omega, Gamma)
    max_mr = np.max(np.abs(R1))
    if max_mr > 0:
        dt = cfl_factor * delta_s / max_mr
    else:
        return x, y

    n_steps = max(int(np.ceil(total_years / dt)), 1)
    dt = total_years / n_steps

    for _ in range(n_steps):
        s = compute_s_distance(x, y)
        s_norm = ((s - s[0]) / (s[-1] - s[0])
                  if s[-1] > s[0] else np.zeros_like(s))
        kl_interp = np.interp(s_norm, s_kl_norm, kl_s)

        curv = _curvature_from_xy(x, y)
        R0 = nominal_migration_rate(curv, W, kl_interp)
        R1 = predicted_migration_rate(R0, s, D, Cf, Omega, Gamma)

        dx_ds = np.gradient(x) / np.gradient(s)
        dy_ds = np.gradient(y) / np.gradient(s)

        x = x - R1 * dy_ds * dt
        y = y + R1 * dx_ds * dt

        x, y = resample_and_smooth(x, y, delta_s,
                                   smoothing_factor=0,
                                   compute_curvature=False)

    return x, y


def predict_forward_local(river, local_calibration, segment_calibration=None,
                          path=None, delta_s=50, smoothing_factor=1e6,
                          pixel_size=None, prediction_years=None):
    """
    Predict migration using spatially-varying kl from local calibration.

    Parameters
    ----------
    river : River
        Most recent River object.
    local_calibration : dict
        Output of ``calibrate_pair_local`` or
        ``calibrate_segment_local``.
    segment_calibration : dict or None
        If provided (output of ``calibrate_segment``), uses its
        ``'residual_std'`` for the uncertainty envelope.  Otherwise
        no envelope is computed.
    path : list of tuples, optional
        Sub-path to use.
    delta_s : float, optional
        Resampling interval (default 50 m).
    smoothing_factor : float, optional
        Spline smoothing factor (default 1e6).
    pixel_size : float, optional
        Pixel size for width conversion.
    prediction_years : float, optional
        Time horizon in years for forward prediction.

    Returns
    -------
    prediction : dict
        - 'x', 'y': smoothed centerline coordinates
        - 's': along-channel distance
        - 'curvature': curvature values
        - 'width': channel width
        - 'kl_local': interpolated kl profile on the centerline
        - 'predicted_mr': predicted migration rate (local kl)
        - 'predicted_mr_global': predicted MR with global kl
        If *prediction_years* is given:
        - 'predicted_x', 'predicted_y': predicted channel position
        If *segment_calibration* is given and *prediction_years* set:
        - 'predicted_x_low/high', 'predicted_y_low/high': envelope
    """
    from .utils import get_width_and_curvature, compute_s_distance

    x, y, s, width, curvature, _ = get_width_and_curvature(
        river, delta_s=delta_s, smoothing_factor=smoothing_factor,
        path=path, pixel_size=pixel_size)

    W = np.mean(width)

    # Extract parameters from local calibration
    # Works with both calibrate_pair_local and calibrate_segment_local output
    if 's_reference' in local_calibration:
        # calibrate_segment_local output
        if 'kl_local_median' in local_calibration:
            kl_s = local_calibration['kl_local_median']
        else:
            # temporally-windowed calibration: use the most recent window
            kl_s = local_calibration['kl_local_series'][-1]['kl_median']
        s_kl = local_calibration['s_reference']
        Cf = (local_calibration['Cf_global'] if 'Cf_global' in local_calibration
              else local_calibration['Cf'])
        kl_global = np.nanmedian(kl_s)
    else:
        # calibrate_pair_local output
        kl_s = local_calibration.get('kl_local_full',
                                     local_calibration['kl_local'])
        s_kl = local_calibration.get('s_full', local_calibration['s'])
        Cf = local_calibration['Cf']
        kl_global = local_calibration['kl_global']

    D = depth_from_width(W)
    Omega = local_calibration.get('Omega', -1.0)
    Gamma = local_calibration.get('Gamma', 2.5)

    # Interpolate kl onto current centerline using DTW alignment
    if 'x_reference' in local_calibration:
        kl_local = _map_to_reference_s(
            local_calibration['x_reference'],
            local_calibration['y_reference'],
            s_kl, kl_s, x, y, s)
    else:
        # calibrate_pair_local doesn't have reference coords — use
        # the pair's own centerline coords if available, else normalised
        kl_local = np.interp(
            (s - s[0]) / (s[-1] - s[0]) if s[-1] > s[0]
            else np.zeros_like(s),
            (s_kl - s_kl[0]) / (s_kl[-1] - s_kl[0]),
            kl_s)
    # Fill NaN gaps with global kl
    nan_mask = np.isnan(kl_local)
    if np.any(nan_mask):
        kl_local[nan_mask] = kl_global

    # Predicted MR with local kl
    R0_local = nominal_migration_rate(curvature, W, kl_local)
    R1_local = predicted_migration_rate(R0_local, s, D, Cf, Omega, Gamma)

    # Global kl for comparison
    R0_global = nominal_migration_rate(curvature, W, kl_global)
    R1_global = predicted_migration_rate(R0_global, s, D, Cf, Omega, Gamma)

    result = {
        'x': x, 'y': y, 's': s,
        'curvature': curvature,
        'width': width,
        'kl_local': kl_local,
        'kl_global': kl_global,
        'predicted_mr': R1_local,
        'predicted_mr_global': R1_global,
        'D': D, 'Cf': Cf,
    }

    if prediction_years is not None:
        px, py = _migrate_forward_local(
            x.copy(), y.copy(), W, D, Cf, kl_local, s,
            Omega, Gamma, prediction_years, delta_s)
        result['predicted_x'] = px
        result['predicted_y'] = py

        # Uncertainty envelope from segment calibration if available
        if segment_calibration is not None:
            residual_std = segment_calibration.get('residual_std', 0.0)
            residual_offset = residual_std * prediction_years
            s_pred = compute_s_distance(px, py)
            dx_ds = np.gradient(px) / np.gradient(s_pred)
            dy_ds = np.gradient(py) / np.gradient(s_pred)
            result['predicted_x_low'] = px + residual_offset * dy_ds
            result['predicted_y_low'] = py - residual_offset * dx_ds
            result['predicted_x_high'] = px - residual_offset * dy_ds
            result['predicted_y_high'] = py + residual_offset * dx_ds

    return result


def _curvature_from_xy(x, y):
    """Compute curvature from x, y coordinates using central differences."""
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    return curvature


def predict_forward(river, calibration, path=None, delta_s=50,
                    smoothing_factor=1e6, pixel_size=None,
                    prediction_years=None):
    """
    Predict migration rate from the most recent centerline using calibrated
    HK model parameters.

    Parameters
    ----------
    river : River
        Most recent River object.
    calibration : dict
        Output of ``calibrate_segment``.  Must have at least
        ``'Cf_median'``, ``'kl_median'``, ``'kl_25'``, ``'kl_75'``,
        ``'D'``, ``'Omega'``, ``'Gamma'``.
    path : list of tuples, optional
        Sub-path to use.  If *None*, uses ``river.main_path``.
    delta_s : float, optional
        Resampling interval (default 50 m).
    smoothing_factor : float, optional
        Spline smoothing factor (default 1e6).
    pixel_size : float, optional
        Pixel size for width conversion.
    prediction_years : float, optional
        Time horizon in years for computing predicted channel position.
        If *None*, only migration rate is returned (no position prediction).

    Returns
    -------
    prediction : dict
        - 'x', 'y': smoothed centerline coordinates
        - 's': along-channel distance
        - 'curvature': curvature values
        - 'width': channel width
        - 'predicted_mr': predicted migration rate (median kl)
        - 'predicted_mr_low': predicted MR minus residual std
        - 'predicted_mr_high': predicted MR plus residual std
        - 'residual_std': median residual std from calibration pairs
        - 'D', 'Cf', 'kl_median', 'kl_25', 'kl_75': parameters used
        If *prediction_years* is given, also includes:
        - 'predicted_x', 'predicted_y': predicted channel position
        - 'predicted_x_low', 'predicted_y_low': lower bound position
        - 'predicted_x_high', 'predicted_y_high': upper bound position
    """
    from .utils import get_width_and_curvature

    x, y, s, width, curvature, _ = get_width_and_curvature(
        river, delta_s=delta_s, smoothing_factor=smoothing_factor,
        path=path, pixel_size=pixel_size)

    W = np.mean(width)
    D = calibration['D']
    Cf = calibration['Cf_median']
    Omega = calibration.get('Omega', -1.0)
    Gamma = calibration.get('Gamma', 2.5)

    def _predict_with_kl(kl):
        R0 = nominal_migration_rate(curvature, W, kl)
        return predicted_migration_rate(R0, s, D, Cf, Omega, Gamma)

    mr_median = _predict_with_kl(calibration['kl_median'])

    # Uncertainty envelope based on residual std from calibration pairs
    residual_std = calibration.get('residual_std', 0.0)
    mr_low = mr_median - residual_std
    mr_high = mr_median + residual_std

    result = {
        'x': x,
        'y': y,
        's': s,
        'curvature': curvature,
        'width': width,
        'predicted_mr': mr_median,
        'predicted_mr_low': mr_low,
        'predicted_mr_high': mr_high,
        'residual_std': residual_std,
        'D': D,
        'Cf': Cf,
        'kl_median': calibration['kl_median'],
        'kl_25': calibration['kl_25'],
        'kl_75': calibration['kl_75'],
    }

    if prediction_years is not None:
        from .utils import compute_s_distance

        # Run forward model with median parameters
        px, py = _migrate_forward(
            x.copy(), y.copy(), W, D, Cf, calibration['kl_median'],
            Omega, Gamma, prediction_years, delta_s)
        result['predicted_x'] = px
        result['predicted_y'] = py

        # Build envelope by offsetting the predicted centerline by
        # +/- residual_std * prediction_years along the local normal
        residual_offset = residual_std * prediction_years
        s_pred = compute_s_distance(px, py)
        dx_ds = np.gradient(px) / np.gradient(s_pred)
        dy_ds = np.gradient(py) / np.gradient(s_pred)

        result['predicted_x_low'] = px + residual_offset * dy_ds
        result['predicted_y_low'] = py - residual_offset * dx_ds
        result['predicted_x_high'] = px - residual_offset * dy_ds
        result['predicted_y_high'] = py + residual_offset * dx_ds

    return result


def temporal_cross_validate(results, df, pair_class_filter='good',
                            n_holdout=1, holdout_date=None,
                            Cf_range=(0.001, 0.02), Omega=-1.0, Gamma=2.5,
                            kl_percentile=75, use_stable_segments=True,
                            use_forward_model=False, delta_s=50,
                            min_time_gap_years=0,
                            max_time_gap_years=None,
                            min_prediction_years=0,
                            use_local_kl=False, window_length=None):
    """
    Calibrate on older pairs and evaluate on held-out recent pairs.

    Test pairs are those where date1 is the last pre-cutoff scene and date2
    is post-cutoff, mimicking the forward-prediction scenario: the model
    sees only the training-era curvature and must predict migration into
    unseen time.

    Parameters
    ----------
    results : dict
        Results dictionary from ``analyze_river_pairs_filtered`` or
        ``analyze_segment_group``.
    df : pandas.DataFrame
        DataFrame from ``create_dataframe_from_results``, used to filter
        pairs by classification.
    pair_class_filter : str or list of str, optional
        Only use pairs with these classifications (default 'good').
    n_holdout : int, optional
        Number of the most recent *scenes* to hold out (default 1).
        Ignored if *holdout_date* is given.
    holdout_date : datetime-like, optional
        Explicit cutoff date. Pairs with both dates before this are
        training; pairs with date1 before and date2 on or after are test.
        If *None*, the cutoff is inferred from *n_holdout*.
    Cf_range : tuple, optional
        Bounds for Cf optimization (default (0.001, 0.02)).
    Omega : float, optional
        Local weight (default -1.0).
    Gamma : float, optional
        Upstream weight (default 2.5).
    kl_percentile : float, optional
        Percentile for amplitude matching (default 75).
    use_stable_segments : bool, optional
        Whether to exclude high-variance segments during calibration
        (default True).
    use_forward_model : bool, optional
        If True, test evaluation uses the iterative forward model
        (``_migrate_forward``) to predict the time-2 centerline from
        time-1 geometry, then compares against the observed time-2
        centerline. This accounts for curvature evolution during
        migration. Requires ``results['centerline_coords']``.
        If False (default), uses the static single-step HK prediction.
    delta_s : float, optional
        Node spacing (m) for the forward model resampling (default 50).
        Only used when *use_forward_model* is True.
    min_time_gap_years : float, optional
        Minimum time gap (in years) for a pair to be included in the
        training set (default 0). Short-gap pairs tend to have noisier
        migration estimates, which can inflate kl. This filter only
        applies to training pairs; test pairs are always evaluated
        regardless of their time gap.
    max_time_gap_years : float or None, optional
        Maximum time gap (in years) for a pair to be included in the
        training set (default *None*, no upper limit). Long-gap pairs
        may have unreliable migration estimates due to cutoffs and
        cumulative geometry changes. Only applies to training pairs;
        test pairs are always evaluated to allow assessing the
        predictability horizon.
    min_prediction_years : float, optional
        Minimum time (in years) between the cutoff date and a test
        pair's date2 (default 0).  Excludes test pairs whose date2 is
        too close to the cutoff, where minimal channel change makes
        prediction trivially easy.
    use_local_kl : bool, optional
        If True, calibrate spatially-varying kl on training pairs and
        use the aggregated kl(s) profile for test evaluation (both
        static and forward model).  Default False.
    window_length : float or None, optional
        Spatial window for local kl estimation (metres).  Auto-estimated
        if *None*.  Only used when *use_local_kl* is True.

    Returns
    -------
    result : dict
        - 'calibration': training-set calibration dict (same format as
          ``calibrate_segment`` output)
        - 'test_pairs': list of dicts, one per test pair, each with:
            - 'pair_idx': index in results
            - 'date1', 'date2': scene dates
            - 'time_gap_years': time between scenes
            - 'r_predicted': Pearson r between HK-predicted and observed MR
            - 'rmse': root mean squared error (m/yr)
            - 'predicted_mr': predicted migration rate array
            - 'observed_mr': observed migration rate array
            - 's': along-channel distance array
            If *use_forward_model* is True, also includes:
            - 'predicted_x', 'predicted_y': forward-predicted centerline
            - 'observed_x2', 'observed_y2': observed time-2 centerline
            - 'position_error_mean': mean Euclidean distance between
              predicted and observed time-2 centerlines (m)
            - 'position_error_median': median distance (m)
            - 'r_forward': correlation between forward-model migration
              distances and observed migration distances
        - 'cutoff_date': the date used to split train/test
        - 'n_train': number of training pairs
        - 'n_test': number of test pairs
        - 'r_test_median': median correlation across test pairs
        - 'rmse_test_median': median RMSE across test pairs
        - 'train_pair_indices': list of training pair indices
        - 'test_pair_indices': list of test pair indices
        - 'use_forward_model': whether the forward model was used
        If *use_local_kl* is True, also includes:
        - 'local_calibration': dict with 'kl_median', 'kl_25', 'kl_75',
          's_reference' arrays from the training-set local calibration
    """
    import pandas as pd

    if isinstance(pair_class_filter, str):
        pair_class_filter = [pair_class_filter]

    # Find qualifying pairs
    qualifying = df.index[df['pair_class'].isin(pair_class_filter)].tolist()
    if len(qualifying) < 2:
        raise ValueError(
            f"Need at least 2 qualifying pairs for cross-validation, "
            f"got {len(qualifying)}")

    # Collect all unique scene dates from qualifying pairs
    all_dates = set()
    for idx in qualifying:
        pi = results['pair_info'][idx]
        all_dates.add(pi['date1'])
        all_dates.add(pi['date2'])
    sorted_dates = sorted(all_dates)

    # Determine cutoff date
    if holdout_date is not None:
        cutoff = holdout_date
    else:
        if n_holdout >= len(sorted_dates):
            raise ValueError(
                f"n_holdout={n_holdout} but only {len(sorted_dates)} "
                f"unique scene dates")
        cutoff = sorted_dates[-n_holdout]

    # Split into training and test pairs
    train_indices = []
    test_indices = []
    for idx in qualifying:
        pi = results['pair_info'][idx]
        if pi['date1'] < cutoff and pi['date2'] < cutoff:
            train_indices.append(idx)
        elif pi['date1'] < cutoff and pi['date2'] >= cutoff:
            test_indices.append(idx)
        # Pairs where both dates are post-cutoff are excluded (no
        # training-era curvature available)

    if len(train_indices) == 0:
        raise ValueError("No training pairs before cutoff date "
                         f"{cutoff}. Try an earlier cutoff or fewer "
                         "holdout scenes.")
    if len(test_indices) == 0:
        raise ValueError("No test pairs crossing the cutoff date "
                         f"{cutoff}. Need pairs where date1 < cutoff "
                         "and date2 >= cutoff.")

    if min_prediction_years > 0:
        import pandas as pd
        cutoff_ts = pd.Timestamp(cutoff)
        min_pred_delta = pd.Timedelta(days=min_prediction_years * 365.25)
        test_indices = [
            idx for idx in test_indices
            if pd.Timestamp(results['pair_info'][idx]['date2'])
            >= cutoff_ts + min_pred_delta]
        if len(test_indices) == 0:
            raise ValueError(
                f"No test pairs with date2 >= {min_prediction_years} "
                f"years after cutoff {cutoff}.")

    # --- Training: calibrate on older pairs ---
    # Filter training pairs by minimum time gap
    if min_time_gap_years > 0:
        train_indices = [idx for idx in train_indices
                         if results['pair_info'][idx]['time_gap_years']
                         >= min_time_gap_years]
        if len(train_indices) == 0:
            raise ValueError(
                f"No training pairs with time gap >= "
                f"{min_time_gap_years} years. Try a smaller threshold.")

    if max_time_gap_years is not None:
        train_indices = [idx for idx in train_indices
                         if results['pair_info'][idx]['time_gap_years']
                         <= max_time_gap_years]
        if len(train_indices) == 0:
            raise ValueError(
                f"No training pairs with time gap <= "
                f"{max_time_gap_years} years.")

    pair_calibrations = []
    for idx in tqdm(train_indices, desc='Calibrating training pairs'):
        try:
            cal = calibrate_pair(results, idx, Cf_range, Omega, Gamma,
                                 kl_percentile, use_stable_segments)
            pair_calibrations.append(cal)
        except Exception as e:
            print(f"  Training pair {idx}: calibration failed - {e}")

    if not pair_calibrations:
        raise ValueError("All training pair calibrations failed")

    Cf_values = np.array([c['Cf'] for c in pair_calibrations])
    kl_values = np.array([c['kl'] for c in pair_calibrations])
    r_values = np.array([c['r_predicted'] for c in pair_calibrations])
    W_values = np.array([c['W'] for c in pair_calibrations])
    residual_stds = np.array([c['residual_std'] for c in pair_calibrations])

    calibration = {
        'pair_calibrations': pair_calibrations,
        'D': pair_calibrations[0]['D'],
        'Cf_median': np.median(Cf_values),
        'Cf_values': Cf_values,
        'kl_median': np.median(kl_values),
        'kl_25': np.percentile(kl_values, 25),
        'kl_75': np.percentile(kl_values, 75),
        'kl_values': kl_values,
        'r_predicted_median': np.median(r_values),
        'r_predicted_values': r_values,
        'n_pairs_calibrated': len(pair_calibrations),
        'W_mean': np.mean(W_values),
        'Omega': Omega,
        'Gamma': Gamma,
        'residual_std': np.median(residual_stds),
    }

    # --- Local kl training (optional) ---
    local_cal_data = None
    if use_local_kl:
        # Calibrate local kl on each training pair, then aggregate
        local_profiles = []
        local_s_grids = []
        local_coords = []
        for idx in tqdm(train_indices, desc='Calibrating local kl'):
            try:
                lcal = calibrate_pair_local(
                    results, idx,
                    use_stable_segments=use_stable_segments,
                    Cf=calibration['Cf_median'],
                    Omega=Omega, Gamma=Gamma,
                    kl_percentile=kl_percentile,
                    window_length=window_length)
                kl_prof = lcal.get('kl_local_full', lcal['kl_local'])
                s_prof = lcal.get('s_full', lcal['s'])
                coords = results['centerline_coords'][idx]
                local_profiles.append(kl_prof)
                local_s_grids.append(s_prof)
                local_coords.append(coords)
                if window_length is None:
                    window_length = lcal['window_length']
            except Exception as e:
                print(f"  Training pair {idx}: local kl failed - {e}")

        if local_profiles:
            # Use the first training pair's centerline as reference
            x_ref = local_coords[0]['x1']
            y_ref = local_coords[0]['y1']
            from .utils import compute_s_distance
            s_ref = compute_s_distance(x_ref, y_ref)

            remapped = []
            for kl_prof, s_prof, coords in zip(
                    local_profiles, local_s_grids, local_coords):
                kl_on_ref = _map_to_reference_s(
                    coords['x1'], coords['y1'], s_prof, kl_prof,
                    x_ref, y_ref, s_ref)
                if not np.all(np.isnan(kl_on_ref)):
                    remapped.append(kl_on_ref)

            if remapped:
                stack = np.array(remapped)
                local_cal_data = {
                    'kl_median': np.nanmedian(stack, axis=0),
                    'kl_25': np.nanpercentile(stack, 25, axis=0),
                    'kl_75': np.nanpercentile(stack, 75, axis=0),
                    's_reference': s_ref,
                    'x_reference': x_ref,
                    'y_reference': y_ref,
                    'Cf': calibration['Cf_median'],
                    'D': calibration['D'],
                    'window_length': window_length,
                    'n_profiles': len(remapped),
                }

    # --- Test: evaluate on held-out pairs ---
    Cf_trained = calibration['Cf_median']
    kl_trained = calibration['kl_median']
    D_trained = calibration['D']

    if use_forward_model:
        from .utils import correlate_curves, compute_migration_distances

    test_results = []
    for idx in tqdm(test_indices, desc='Evaluating test pairs'):
        pi = results['pair_info'][idx]
        curvature = results['curvatures'][idx]
        distances = results['migration_distances'][idx]
        s = results['along_channel_distances'][idx]
        time_gap_years = pi['time_gap_years']
        W = np.mean(pi['width1'])

        observed_mr = distances / time_gap_years

        # --- Determine kl for this test pair ---
        if use_local_kl and local_cal_data is not None:
            # Map the aggregated kl(s) profile onto this test pair's
            # centerline using DTW geometric alignment
            coords = results['centerline_coords'][idx]
            kl_for_test = _map_to_reference_s(
                local_cal_data['x_reference'],
                local_cal_data['y_reference'],
                local_cal_data['s_reference'],
                local_cal_data['kl_median'],
                coords['x1'], coords['y1'], s)
            # Fill any NaN gaps (e.g. at endpoints) with global kl
            nan_mask = np.isnan(kl_for_test)
            if np.any(nan_mask):
                kl_for_test[nan_mask] = kl_trained
        else:
            kl_for_test = kl_trained

        # --- Static prediction (always computed) ---
        R0 = nominal_migration_rate(curvature, W, kl_for_test)
        R1 = predicted_migration_rate(R0, s, D_trained, Cf_trained,
                                      Omega, Gamma)

        # Evaluate on stable segments only if available
        segment_results = pi.get('segment_results', [])
        if use_stable_segments and segment_results:
            mask = np.zeros(len(curvature), dtype=bool)
            for seg in segment_results:
                mask[seg['start_idx']:seg['end_idx'] + 1] = True
            R1_eval = R1[mask]
            obs_eval = observed_mr[mask]
        else:
            R1_eval = R1
            obs_eval = observed_mr

        if len(R1_eval) > 2:
            r = np.corrcoef(R1_eval, obs_eval)[0, 1]
            rmse = np.sqrt(np.mean((R1_eval - obs_eval) ** 2))
        else:
            r = np.nan
            rmse = np.nan

        test_entry = {
            'pair_idx': idx,
            'date1': pi['date1'],
            'date2': pi['date2'],
            'time_gap_years': time_gap_years,
            'r_predicted': r,
            'rmse': rmse,
            'predicted_mr': R1,
            'observed_mr': observed_mr,
            's': s,
        }

        # --- Iterative forward model ---
        if use_forward_model:
            try:
                coords = results['centerline_coords'][idx]
                x1, y1 = coords['x1'].copy(), coords['y1'].copy()
                x2_obs, y2_obs = coords['x2'].copy(), coords['y2'].copy()

                # Forward-integrate from time-1 geometry
                if use_local_kl and local_cal_data is not None:
                    from .utils import compute_s_distance as _cs
                    s1_fwd = _cs(x1, y1)
                    kl_fwd = _map_to_reference_s(
                        local_cal_data['x_reference'],
                        local_cal_data['y_reference'],
                        local_cal_data['s_reference'],
                        local_cal_data['kl_median'],
                        x1, y1, s1_fwd)
                    nan_mask = np.isnan(kl_fwd)
                    if np.any(nan_mask):
                        kl_fwd[nan_mask] = kl_trained
                    px, py = _migrate_forward_local(
                        x1.copy(), y1.copy(), W, D_trained, Cf_trained,
                        kl_fwd, s1_fwd, Omega, Gamma,
                        time_gap_years, delta_s)
                else:
                    px, py = _migrate_forward(
                        x1.copy(), y1.copy(), W, D_trained, Cf_trained,
                        kl_trained, Omega, Gamma, time_gap_years, delta_s)

                # DTW-align predicted time-2 against observed time-2
                p_fwd, q_fwd, cost_fwd = correlate_curves(
                    px, x2_obs, py, y2_obs)
                del cost_fwd

                # Position error: Euclidean distance between DTW-matched
                # points on predicted and observed time-2 centerlines
                pos_errors = np.sqrt(
                    (px[p_fwd] - x2_obs[q_fwd]) ** 2 +
                    (py[p_fwd] - y2_obs[q_fwd]) ** 2)

                # DTW cost: time-1 vs forward-predicted time-2
                # Compare against the observed DTW cost to gauge
                # whether the predicted shape is as realistic as the
                # observed one
                p_t1_pred, q_t1_pred, dtw_cost_matrix = \
                    correlate_curves(x1, px, y1, py)
                dtw_cost_predicted = float(dtw_cost_matrix[-1, -1])
                del dtw_cost_matrix
                dtw_cost_observed = pi.get('dtw_cost', np.nan)

                # Migration distances: forward-predicted vs observed,
                # both measured from time-1 centerline
                fwd_distances, fwd_valid = compute_migration_distances(
                    x1, y1, px, py, p_t1_pred, q_t1_pred)

                # Correlation of forward-model migration vs observed
                # migration (both from time-1), restricted to stable
                # segments when requested
                both_valid = fwd_valid & ~np.isnan(distances)
                if use_stable_segments and segment_results:
                    stable_mask = np.zeros(len(curvature), dtype=bool)
                    for seg in segment_results:
                        stable_mask[seg['start_idx']:seg['end_idx'] + 1] = True
                    both_valid = both_valid & stable_mask
                if np.sum(both_valid) > 2:
                    fwd_mr = fwd_distances[both_valid] / time_gap_years
                    obs_mr_valid = observed_mr[both_valid]
                    r_forward = np.corrcoef(fwd_mr, obs_mr_valid)[0, 1]
                    rmse_forward = np.sqrt(
                        np.mean((fwd_mr - obs_mr_valid) ** 2))
                else:
                    r_forward = np.nan
                    rmse_forward = np.nan

                test_entry.update({
                    'predicted_x': px,
                    'predicted_y': py,
                    'observed_x2': x2_obs,
                    'observed_y2': y2_obs,
                    'position_error_mean': np.mean(pos_errors),
                    'position_error_median': np.median(pos_errors),
                    'r_forward': r_forward,
                    'rmse_forward': rmse_forward,
                    'observed_distances': distances,
                    'forward_distances': fwd_distances,
                    'dtw_cost_predicted': dtw_cost_predicted,
                    'dtw_cost_observed': dtw_cost_observed,
                })
            except Exception as e:
                print(f"  Test pair {idx}: forward model failed - {e}")
                test_entry.update({
                    'r_forward': np.nan,
                    'rmse_forward': np.nan,
                    'position_error_mean': np.nan,
                    'position_error_median': np.nan,
                })

        test_results.append(test_entry)

    r_test = [t['r_predicted'] for t in test_results
              if not np.isnan(t['r_predicted'])]
    rmse_test = [t['rmse'] for t in test_results
                 if not np.isnan(t['rmse'])]

    output = {
        'calibration': calibration,
        'test_pairs': test_results,
        'cutoff_date': cutoff,
        'n_train': len(train_indices),
        'n_test': len(test_indices),
        'r_test_median': np.median(r_test) if r_test else np.nan,
        'rmse_test_median': np.median(rmse_test) if rmse_test else np.nan,
        'train_pair_indices': train_indices,
        'test_pair_indices': test_indices,
        'use_forward_model': use_forward_model,
    }

    if use_forward_model:
        r_fwd = [t['r_forward'] for t in test_results
                 if not np.isnan(t.get('r_forward', np.nan))]
        rmse_fwd = [t['rmse_forward'] for t in test_results
                    if not np.isnan(t.get('rmse_forward', np.nan))]
        pos_err = [t['position_error_mean'] for t in test_results
                   if not np.isnan(t.get('position_error_mean', np.nan))]
        output['r_forward_median'] = np.median(r_fwd) if r_fwd else np.nan
        output['rmse_forward_median'] = (
            np.median(rmse_fwd) if rmse_fwd else np.nan)
        output['position_error_mean_median'] = (
            np.median(pos_err) if pos_err else np.nan)

    if use_local_kl and local_cal_data is not None:
        output['local_calibration'] = local_cal_data

    return output


def detect_cutoff_risk(river, path=None, neck_width_threshold=2.0,
                       delta_s=50, smoothing_factor=1e6, pixel_size=None,
                       search_distance_wavelengths=0.5):
    """
    Identify bends approaching meander cutoff by measuring neck width.

    For each bend apex (curvature extremum), measures the minimum distance
    from the apex to other parts of the centerline beyond the adjacent
    inflection points. When this distance drops below
    ``neck_width_threshold * channel_width``, the bend is flagged.

    Parameters
    ----------
    river : River
        River object.
    path : list of tuples, optional
        Sub-path to use.
    neck_width_threshold : float, optional
        Cutoff warning when neck_width / channel_width < this value
        (default 2.0).
    delta_s : float, optional
        Resampling interval (default 50 m).
    smoothing_factor : float, optional
        Spline smoothing factor (default 1e6).
    pixel_size : float, optional
        Pixel size for width conversion.
    search_distance_wavelengths : float, optional
        Minimum along-channel distance (in wavelengths) to skip when
        searching for the nearest point on the centerline (default 0.5).

    Returns
    -------
    risks : list of dict
        One entry per bend, each with:
        - 'bend_index': index of the curvature extremum
        - 'apex_x', 'apex_y': coordinates of the bend apex
        - 'along_channel_distance': s-distance of the apex
        - 'neck_width_m': minimum distance to another part of the channel
        - 'channel_width_m': local channel width
        - 'neck_ratio': neck_width / channel_width
        - 'risk_level': 'high', 'moderate', or 'low'
    """
    from .utils import get_width_and_curvature
    from scipy.spatial import cKDTree

    x, y, s, width, curvature, _ = get_width_and_curvature(
        river, delta_s=delta_s, smoothing_factor=smoothing_factor,
        path=path, pixel_size=pixel_size)

    n = len(x)
    coords = np.column_stack([x, y])
    W_mean = np.mean(width)

    # Estimate wavelength as 2 * pi * W_mean * some factor
    # Use a simpler approach: skip distance = search_distance_wavelengths
    # times mean meander wavelength. Approximate wavelength ~ 12 * W_mean.
    skip_distance = search_distance_wavelengths * 12.0 * W_mean

    # Find curvature extrema (bend apexes)
    from scipy.signal import argrelextrema
    maxima = argrelextrema(np.abs(curvature), np.greater, order=5)[0]

    # Build k-d tree for fast nearest-neighbor queries
    tree = cKDTree(coords)

    risks = []
    for apex_idx in maxima:
        apex_s = s[apex_idx]
        local_width = width[apex_idx]

        # Find points far enough away along the channel
        s_diff = np.abs(s - apex_s)
        far_mask = s_diff > skip_distance
        far_indices = np.where(far_mask)[0]

        if len(far_indices) == 0:
            continue

        # Find minimum distance from apex to far-away points
        dists = np.sqrt((x[far_indices] - x[apex_idx])**2 +
                        (y[far_indices] - y[apex_idx])**2)
        neck_width = np.min(dists)
        neck_ratio = neck_width / local_width if local_width > 0 else np.inf

        if neck_ratio > 5.0:
            risk_level = 'low'
        elif neck_ratio > neck_width_threshold:
            risk_level = 'moderate'
        else:
            risk_level = 'high'

        risks.append({
            'bend_index': int(apex_idx),
            'apex_x': x[apex_idx],
            'apex_y': y[apex_idx],
            'along_channel_distance': apex_s,
            'neck_width_m': neck_width,
            'channel_width_m': local_width,
            'neck_ratio': neck_ratio,
            'risk_level': risk_level,
        })

    return risks


def track_parameter_stability(pair_calibrations):
    """
    Analyze how calibrated parameters vary across time intervals.

    Parameters
    ----------
    pair_calibrations : list of dict
        Per-pair calibration results from ``calibrate_segment``.

    Returns
    -------
    stability : dict
        - 'temporal_df': DataFrame with mid_date, D, Cf, kl, r_predicted,
          time_gap_years
        - 'kl_trend': dict with 'rho' (Spearman), 'p_value'
        - 'Cf_trend': dict with 'rho', 'p_value'
    """
    import pandas as pd

    records = []
    for cal in pair_calibrations:
        mid_date = cal['date1'] + (cal['date2'] - cal['date1']) / 2
        records.append({
            'mid_date': mid_date,
            'date1': cal['date1'],
            'date2': cal['date2'],
            'D': cal['D'],
            'Cf': cal['Cf'],
            'kl': cal['kl'],
            'r_predicted': cal['r_predicted'],
            'time_gap_years': cal['time_gap_years'],
        })

    temporal_df = pd.DataFrame(records).sort_values('mid_date')

    result = {'temporal_df': temporal_df}

    if len(temporal_df) >= 3:
        mid_ordinal = temporal_df['mid_date'].apply(
            lambda d: d.toordinal() if hasattr(d, 'toordinal') else d)

        for param in ['kl', 'Cf']:
            rho, p = stats.spearmanr(mid_ordinal, temporal_df[param])
            result[f'{param}_trend'] = {'rho': rho, 'p_value': p}
    else:
        for param in ['kl', 'Cf']:
            result[f'{param}_trend'] = {'rho': np.nan, 'p_value': np.nan}

    return result


def _estimate_window_length(curvature, s):
    """Estimate a spatial window length of ~2 meander wavelengths from
    curvature zero-crossings.

    Parameters
    ----------
    curvature : array_like
        Curvature along the channel.
    s : array_like
        Along-channel distance (m).

    Returns
    -------
    window_length : float
        Estimated window length in metres.
    """
    from .analysis import find_zero_crossings
    loc_zero, _ = find_zero_crossings(curvature)
    if len(loc_zero) < 3:
        # Fallback: 20% of total length
        return 0.2 * (s[-1] - s[0])
    half_wavelengths = np.diff(s[loc_zero])
    # 2 full wavelengths = 4 half-wavelengths
    return 4.0 * np.median(half_wavelengths)


def calibrate_local_kl(curvature, observed_mr, s, W,
                       Cf=None, Cf_range=(0.001, 0.02),
                       Omega=-1.0, Gamma=2.5,
                       kl_percentile=75,
                       window_length=None,
                       min_window_points=20,
                       mask=None):
    """
    Estimate spatially-varying kl along the channel using a moving window.

    Uses a global Cf (optimised once over the full river) and slides a
    spatial window to estimate local kl by amplitude-matching the HK
    predicted migration rate to the observed migration rate.

    Parameters
    ----------
    curvature : array_like
        Curvature along the channel (1/m).
    observed_mr : array_like
        Observed migration rate (m/yr).
    s : array_like
        Along-channel distance (m).
    W : float
        Mean channel width (m).
    Cf : float or None
        Friction factor.  If *None*, optimised globally via
        ``calibrate_Cf``.
    Cf_range : tuple, optional
        Bounds for Cf optimisation (default (0.001, 0.02)).
    Omega : float, optional
        Local weight (default -1.0).
    Gamma : float, optional
        Upstream weight (default 2.5).
    kl_percentile : float, optional
        Percentile for amplitude matching (default 75).
    window_length : float or None
        Spatial window length in metres.  If *None*, auto-estimated from
        curvature zero-crossings (~2 meander wavelengths).
    min_window_points : int, optional
        Minimum number of points in a window for a valid kl estimate
        (default 20).
    mask : array_like of bool, optional
        Points to use for the calibration statistics (Cf correlation and
        window percentiles), e.g. a stable-segment mask. The prediction is
        always computed on the full contiguous arrays, since the convolution
        assumes uniform spacing.

    Returns
    -------
    result : dict
        - 'kl_local': array of kl values (same length as curvature)
        - 'kl_window_centers': along-channel positions of window centres
        - 'kl_window_values': kl at each window centre (before interp)
        - 'Cf': global Cf used
        - 'D': depth estimate
        - 'W': mean channel width
        - 'R1_local': predicted migration rate using kl_local
        - 'R1_global': predicted migration rate using global kl
        - 'kl_global': global kl for comparison
        - 'window_length': window length used (metres)
        - 'observed_mr': observed migration rate (pass-through)
        - 's': along-channel distance (pass-through)
    """
    curvature = np.asarray(curvature, dtype=float)
    observed_mr = np.asarray(observed_mr, dtype=float)
    s = np.asarray(s, dtype=float)
    if mask is None:
        mask = np.ones(len(curvature), dtype=bool)

    # --- Step 1: depth ---
    D = depth_from_width(W)

    # --- Step 2: global Cf ---
    # Use kl=1 for initial R0 so that kl cancels out of the Cf optimisation
    R0_unit = nominal_migration_rate(curvature, W, 1.0)
    if Cf is None:
        Cf, _ = calibrate_Cf(R0_unit, observed_mr, s, D, Cf_range,
                             Omega, Gamma, mask=mask)

    # --- Step 3: global kl for comparison ---
    R1_unit = predicted_migration_rate(R0_unit, s, D, Cf, Omega, Gamma)
    p_obs = np.percentile(np.abs(observed_mr[mask]), kl_percentile)
    p_r1 = np.percentile(np.abs(R1_unit[mask]), kl_percentile)
    kl_global = p_obs / p_r1 if p_r1 > 0 else 1.0

    R0_global = nominal_migration_rate(curvature, W, kl_global)
    R1_global = predicted_migration_rate(R0_global, s, D, Cf, Omega, Gamma)

    # --- Step 4: auto-estimate window length ---
    if window_length is None:
        window_length = _estimate_window_length(curvature, s)

    # --- Step 5: sliding window local kl ---
    half_win = window_length / 2.0
    step = window_length / 2.0  # 50% overlap
    s_min, s_max = s[0], s[-1]

    window_centers = []
    kl_values = []
    sc = s_min + half_win
    while sc <= s_max - half_win:
        in_window = (s >= sc - half_win) & (s <= sc + half_win) & mask
        n_pts = np.sum(in_window)
        if n_pts >= min_window_points:
            p_obs_local = np.percentile(np.abs(observed_mr[in_window]),
                                        kl_percentile)
            p_r1_local = np.percentile(np.abs(R1_unit[in_window]),
                                       kl_percentile)
            kl_local = p_obs_local / p_r1_local if p_r1_local > 0 else kl_global
            window_centers.append(sc)
            kl_values.append(kl_local)
        sc += step

    window_centers = np.array(window_centers)
    kl_values = np.array(kl_values)

    # --- Step 6: interpolate to every point ---
    if len(window_centers) >= 2:
        kl_local = np.interp(s, window_centers, kl_values)
    elif len(window_centers) == 1:
        kl_local = np.full_like(s, kl_values[0])
    else:
        # Not enough data for windowing — fall back to global kl
        kl_local = np.full_like(s, kl_global)

    # --- Step 7: compute R1 with local kl ---
    # R1_unit was computed with kl=1, so R1_local = kl_local * R1_unit
    R1_local = kl_local * R1_unit

    return {
        'kl_local': kl_local,
        'kl_window_centers': window_centers,
        'kl_window_values': kl_values,
        'Cf': Cf,
        'D': D,
        'W': W,
        'R1_local': R1_local,
        'R1_global': R1_global,
        'kl_global': kl_global,
        'window_length': window_length,
        'observed_mr': observed_mr,
        's': s,
    }


def calibrate_pair_local(results, pair_idx, use_stable_segments=True,
                         **kwargs):
    """
    Calibrate local kl for a single pair from analysis results.

    Thin wrapper around ``calibrate_local_kl`` that extracts arrays from
    the ``results`` dictionary (analogous to ``calibrate_pair``).

    Parameters
    ----------
    results : dict
        Results from ``analyze_river_pairs_filtered`` or
        ``analyze_segment_group``.
    pair_idx : int
        Index of the pair.
    use_stable_segments : bool, optional
        If True, mask out high-variance regions before calibration
        (default True).
    **kwargs
        Forwarded to ``calibrate_local_kl``.

    Returns
    -------
    result : dict
        Output of ``calibrate_local_kl``, plus ``'date1'``, ``'date2'``,
        ``'time_gap_years'``, ``'pair_idx'``.
    """
    pair_info = results['pair_info'][pair_idx]
    curvature = results['curvatures'][pair_idx]
    distances = results['migration_distances'][pair_idx]
    s = results['along_channel_distances'][pair_idx]
    time_gap_years = pair_info['time_gap_years']
    W = np.mean(pair_info['width1'])
    observed_mr = distances / time_gap_years

    # Restrict calibration statistics to stable segments if requested.
    # The arrays passed to calibrate_local_kl are always the full contiguous
    # ones: slicing out high-variance regions before the convolution would
    # concatenate spatially disjoint points and distort the kernel.
    mask = None
    if use_stable_segments:
        segment_results = pair_info.get('segment_results', [])
        if segment_results:
            mask = np.zeros(len(curvature), dtype=bool)
            for seg in segment_results:
                mask[seg['start_idx']:seg['end_idx'] + 1] = True

    cal = calibrate_local_kl(curvature, observed_mr, s, W,
                             mask=mask, **kwargs)

    # Everything is already on the full extent; keep the *_full keys as
    # aliases for backward compatibility with existing consumers
    cal['kl_local_full'] = cal['kl_local']
    cal['R1_local_full'] = cal['R1_local']
    cal['R1_global_full'] = cal['R1_global']
    cal['observed_mr_full'] = observed_mr
    cal['s_full'] = s

    cal['date1'] = pair_info['date1']
    cal['date2'] = pair_info['date2']
    cal['time_gap_years'] = time_gap_years
    cal['pair_idx'] = pair_idx
    return cal


def _map_to_reference_s(x_pair, y_pair, s_pair, values,
                        x_ref, y_ref, s_ref, max_dtw_points=1000):
    """
    Remap values from one centerline's s-coordinate onto a reference
    centerline's s-coordinate using DTW alignment.

    To limit memory usage, both centerlines are downsampled to at most
    *max_dtw_points* before running DTW.  The resulting correspondence
    is then used to interpolate values onto the full-resolution
    ``s_ref`` grid.

    Parameters
    ----------
    x_pair, y_pair : array_like
        Coordinates of the source centerline.
    s_pair : array_like
        Along-channel distance on the source centerline.
    values : array_like
        Values to remap (same length as s_pair).
    x_ref, y_ref : array_like
        Coordinates of the reference centerline.
    s_ref : array_like
        Along-channel distance on the reference centerline.
    max_dtw_points : int, optional
        Maximum number of points per centerline for the DTW computation
        (default 1000).  Reduces memory from O(N^2) to O(max^2).

    Returns
    -------
    values_on_ref : ndarray
        Values interpolated onto ``s_ref``.
    """
    from .utils import correlate_curves

    x_pair = np.asarray(x_pair, dtype=float)
    y_pair = np.asarray(y_pair, dtype=float)
    x_ref = np.asarray(x_ref, dtype=float)
    y_ref = np.asarray(y_ref, dtype=float)
    s_pair = np.asarray(s_pair, dtype=float)
    s_ref = np.asarray(s_ref, dtype=float)
    values = np.asarray(values, dtype=float)

    n_pair = len(x_pair)
    n_ref = len(x_ref)

    # Downsample if needed
    if n_pair > max_dtw_points:
        step_p = n_pair / max_dtw_points
        idx_p = np.round(np.arange(0, n_pair, step_p)).astype(int)
        idx_p = idx_p[idx_p < n_pair]
    else:
        idx_p = np.arange(n_pair)

    if n_ref > max_dtw_points:
        step_r = n_ref / max_dtw_points
        idx_r = np.round(np.arange(0, n_ref, step_r)).astype(int)
        idx_r = idx_r[idx_r < n_ref]
    else:
        idx_r = np.arange(n_ref)

    p, q, _ = correlate_curves(
        x_pair[idx_p], x_ref[idx_r],
        y_pair[idx_p], y_ref[idx_r])
    import gc
    gc.collect()

    # Map DTW indices back to original arrays
    p_orig = idx_p[p]
    q_orig = idx_r[q]

    # Build a mapping: for each reference point q_orig[i], the matched
    # source value is values[p_orig[i]].  Average duplicates.
    from collections import defaultdict
    ref_to_vals = defaultdict(list)
    for pi, qi in zip(p_orig, q_orig):
        if pi < len(values):
            ref_to_vals[qi].append(values[pi])

    mapped_s = []
    mapped_v = []
    for qi in sorted(ref_to_vals.keys()):
        if qi < len(s_ref):
            mapped_s.append(s_ref[qi])
            mapped_v.append(np.mean(ref_to_vals[qi]))

    if len(mapped_s) < 2:
        return np.full_like(s_ref, np.nan)

    return np.interp(s_ref, mapped_s, mapped_v)


def calibrate_segment_local(results, df, pair_class_filter='good',
                            window_length=None,
                            time_window_years=5.0,
                            time_step_years=None,
                            use_stable_segments=True,
                            min_time_gap_years=0,
                            reference_river=None,
                            delta_s=100,
                            **kwargs):
    """
    Aggregate spatially-varying kl profiles across pairs with optional
    temporal windowing.

    For each qualifying pair, ``calibrate_pair_local`` produces a kl(s)
    profile.  These profiles are DTW-aligned to a reference centerline
    and aggregated (median, IQR).  With temporal windowing, a sliding
    time window selects subsets of pairs, yielding kl(s, t).

    Parameters
    ----------
    results : dict
        Results from ``analyze_river_pairs_filtered``.
    df : DataFrame
        Analysis DataFrame with ``pair_class`` column.
    pair_class_filter : str or list of str
        Pair class(es) to include (default ``'good'``).
    window_length : float or None
        Spatial window in metres (auto-estimated if *None*).
    time_window_years : float or None
        Width of the temporal window in years.  If *None*, all pairs are
        used (no temporal windowing).
    time_step_years : float or None
        Step between temporal window centres (default: half of
        *time_window_years*).
    use_stable_segments : bool, optional
        Exclude high-variance regions (default True).
    min_time_gap_years : float, optional
        Minimum pair time gap (default 0).
    reference_river : River or None
        River object to use as the spatial reference for the common
        s-grid.  If *None*, uses the last river in
        ``results['filtered_rivers']``.
    delta_s : float, optional
        Resampling interval for the reference s-grid (default 100 m).
    **kwargs
        Forwarded to ``calibrate_local_kl``.

    Returns
    -------
    result : dict
        Always contains:
        - 's_reference': common along-channel distance grid
        - 'x_reference', 'y_reference': reference centerline coords
        - 'Cf_global': global Cf used
        - 'window_length': spatial window used

        Without temporal window (``time_window_years=None``):
        - 'kl_local_median': median kl(s) profile
        - 'kl_local_25', 'kl_local_75': IQR bounds
        - 'n_pairs': number of pairs used
        - 'pair_indices': list of pair indices used

        With temporal window:
        - 'time_centers': list of datetime window centres
        - 'kl_local_series': list of dicts, one per time centre, each
          with 'kl_median', 'kl_25', 'kl_75', 'n_pairs', 'pair_indices'
    """
    import pandas as pd
    from .utils import get_width_and_curvature

    if isinstance(pair_class_filter, str):
        pair_class_filter = [pair_class_filter]

    # --- Identify qualifying pairs ---
    qualifying = df.index[df['pair_class'].isin(pair_class_filter)].tolist()
    if min_time_gap_years > 0:
        qualifying = [idx for idx in qualifying
                      if results['pair_info'][idx]['time_gap_years']
                      >= min_time_gap_years]
    if not qualifying:
        raise ValueError("No qualifying pairs found")

    # --- Build reference centerline ---
    if reference_river is None:
        filtered_rivers = results.get('filtered_rivers', [])
        if not filtered_rivers:
            raise ValueError(
                "results does not contain 'filtered_rivers'. "
                "Pass a River object via the reference_river parameter.")
        reference_river = filtered_rivers[-1]
    ref_river = reference_river

    pixel_size = kwargs.pop('pixel_size', None)
    x_ref, y_ref, s_ref, w_ref, _, _ = get_width_and_curvature(
        ref_river, delta_s=delta_s, pixel_size=pixel_size)

    # --- Calibrate each pair and remap to reference ---
    pair_calibrations = {}
    for idx in qualifying:
        try:
            cal = calibrate_pair_local(results, idx,
                                       use_stable_segments=use_stable_segments,
                                       window_length=window_length,
                                       **kwargs)
            coords = results['centerline_coords'][idx]
            kl_profile = cal.get('kl_local_full', cal['kl_local'])
            s_profile = cal.get('s_full', cal['s'])

            kl_on_ref = _map_to_reference_s(
                coords['x1'], coords['y1'], s_profile, kl_profile,
                x_ref, y_ref, s_ref)

            pair_calibrations[idx] = {
                'kl_on_ref': kl_on_ref,
                'cal': cal,
                'date1': cal['date1'],
                'date2': cal['date2'],
            }
            if window_length is None:
                window_length = cal['window_length']
        except Exception as e:
            print(f"  Pair {idx}: local calibration failed - {e}")

    if not pair_calibrations:
        raise ValueError("All pair calibrations failed")

    first_cal = next(iter(pair_calibrations.values()))['cal']
    Cf_global = first_cal['Cf']

    output = {
        's_reference': s_ref,
        'x_reference': x_ref,
        'y_reference': y_ref,
        'Cf_global': Cf_global,
        'window_length': window_length,
    }

    def _aggregate_pairs(indices):
        """Stack kl profiles for given pair indices and return stats."""
        profiles = []
        for idx in indices:
            kl = pair_calibrations[idx]['kl_on_ref']
            if not np.all(np.isnan(kl)):
                profiles.append(kl)
        if not profiles:
            n = len(s_ref)
            return {
                'kl_median': np.full(n, np.nan),
                'kl_25': np.full(n, np.nan),
                'kl_75': np.full(n, np.nan),
                'n_pairs': 0,
                'pair_indices': [],
            }
        stack = np.array(profiles)
        return {
            'kl_median': np.nanmedian(stack, axis=0),
            'kl_25': np.nanpercentile(stack, 25, axis=0),
            'kl_75': np.nanpercentile(stack, 75, axis=0),
            'n_pairs': len(profiles),
            'pair_indices': list(indices),
        }

    if time_window_years is None:
        # --- No temporal windowing: aggregate all ---
        agg = _aggregate_pairs(list(pair_calibrations.keys()))
        output['kl_local_median'] = agg['kl_median']
        output['kl_local_25'] = agg['kl_25']
        output['kl_local_75'] = agg['kl_75']
        output['n_pairs'] = agg['n_pairs']
        output['pair_indices'] = agg['pair_indices']
    else:
        # --- Temporal windowing ---
        if time_step_years is None:
            time_step_years = time_window_years / 2.0

        pair_midpoints = {}
        for idx, info in pair_calibrations.items():
            d1 = info['date1']
            d2 = info['date2']
            if isinstance(d1, str):
                d1 = pd.Timestamp(d1)
            if isinstance(d2, str):
                d2 = pd.Timestamp(d2)
            pair_midpoints[idx] = d1 + (d2 - d1) / 2

        all_midpoints = sorted(pair_midpoints.values())
        t_min = all_midpoints[0]
        t_max = all_midpoints[-1]

        half_win = pd.Timedelta(days=time_window_years * 365.25 / 2)
        step = pd.Timedelta(days=time_step_years * 365.25)

        time_centers = []
        kl_series = []
        tc = t_min + half_win
        while tc <= t_max - half_win:
            in_window = [idx for idx, mp in pair_midpoints.items()
                         if abs((mp - tc).total_seconds())
                         <= half_win.total_seconds()]
            if in_window:
                agg = _aggregate_pairs(in_window)
                time_centers.append(tc)
                kl_series.append(agg)
            tc += step

        if not time_centers:
            agg = _aggregate_pairs(list(pair_calibrations.keys()))
            time_centers = [t_min + (t_max - t_min) / 2]
            kl_series = [agg]

        output['time_centers'] = time_centers
        output['kl_local_series'] = kl_series

    return output
