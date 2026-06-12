# Migration prediction

`rivabar.prediction` implements the Howard & Knutson (1984) convolution model for meander
migration, following Sylvester et al. (2019, *Geology*): the predicted migration rate is

    R1(s) = Omega * R0(s) + Gamma * conv(R0, G) / integral(G)

where `R0 = kl * W * C` is the nominal (curvature-driven) migration rate and
`G(xi) = exp(-2 * Cf * xi / D)` weights the upstream curvature history. Depth `D` is
estimated from width via the Konsoer et al. (2013) regression.

## Calibration

Starting from a pairwise analysis results dictionary and its classification DataFrame:

```python
import rivabar as rb

calibration = rb.calibrate_segment(results, df, pair_class_filter='good')
print(calibration['Cf_median'], calibration['kl_median'])
```

Each `'good'` pair is calibrated by optimizing `Cf` (the phase of the migration signal)
and amplitude-matching `kl`; the segment-level result aggregates medians and quartiles.
High-variance (cutoff-affected) reaches are excluded from the statistics, while the
convolution itself always runs on the full contiguous centerline.

### Spatially-varying erodibility

`kl` often varies along a reach with bank material and vegetation while `Cf` stays
roughly uniform. `calibrate_local_kl` estimates a kl(s) profile with a sliding window
(auto-sized to ~2 meander wavelengths), and `calibrate_segment_local` aggregates profiles
across pairs by DTW-aligning them to a reference centerline:

```python
local_cal = rb.calibrate_segment_local(results, df, pair_class_filter='good')
```

## Forward prediction and validation

```python
# forward prediction with uncertainty envelope
prediction = rb.predict_forward(rivers[-1], calibration, prediction_years=20)

# spatially-varying version
prediction = rb.predict_forward_local(rivers[-1], local_cal, prediction_years=20)

# hold out the most recent scenes and test the model on them
cv = rb.temporal_cross_validate(results, df, n_holdout=3, use_forward_model=True)

# neck cutoff risk along the most recent centerline
risks = rb.detect_cutoff_risk(rivers[-1], threshold=2.0)
```
