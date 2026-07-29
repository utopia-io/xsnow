"""
Tests for the sk38 and sk38_ccl extensions.

The core test compares the vectorized extension output against an
independent, loop-based scalar implementation of the same physics.
"""
from __future__ import annotations

import functools
import math

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xsnow import xsnowDataset
from xsnow.extensions.sk38 import (
    SKIER_LINE_LOAD,
    alpha_max_skier_stress,
    compute_skier_penetration,
)
from xsnow.extensions import sk38 as sk38_ext  # noqa: F401
from xsnow.extensions import sk38_ccl as sk38_ccl_ext  # noqa: F401

RHO_ICE = 917.0
G = 9.81
CAP = 5.0


# ---------------------- dataset construction --------------------------------
def build_profile_ds(profiles):
    """
    Build an xsnowDataset from a list of per-layer profile dicts.

    Each profile is a dict with 1-D arrays ``height`` [m], ``density``,
    ``shear_strength`` [Pa], ``grain_type``, ``grain_size`` [m], ``rta``.
    Profiles are stacked along ``time``; the layer axis is NaN-padded.
    """
    n_layer = max(len(p['height']) for p in profiles) + 1  # + padding row
    time = pd.date_range('2025-01-01', periods=len(profiles), freq='D')

    def stack(key):
        out = np.full((1, len(profiles), 1, 1, n_layer), np.nan)
        for t, prof in enumerate(profiles):
            out[0, t, 0, 0, :len(prof[key])] = prof[key]
        return out

    dims = ('location', 'time', 'slope', 'realization', 'layer')
    coords = {
        'location': ['SYN1'], 'time': time.values,
        'slope': [0], 'realization': [0], 'layer': np.arange(n_layer),
    }
    data = {
        var: xr.DataArray(stack(var), dims=dims, coords=coords)
        for var in ('height', 'density', 'shear_strength',
                    'grain_type', 'grain_size', 'rta')
    }
    return xsnowDataset(xr.Dataset(data))


def default_profile(n=8):
    """A layered snowpack with a pronounced weak layer at index 2."""
    height = np.cumsum(np.full(n, 0.12))              # 12 cm layers
    density = np.linspace(320.0, 120.0, n)            # denser at the bottom
    shear = np.linspace(4200.0, 900.0, n)             # Pa
    gtype = np.full(n, 330.0)                         # rounded grains
    gsize = np.full(n, 0.0005)                        # 0.5 mm
    rta = np.full(n, 0.9)
    # weak layer: light, coarse, weak faceted layer
    density[2], shear[2], gtype[2], gsize[2] = 140.0, 300.0, 430.0, 0.0015
    return {
        'height': height, 'density': density, 'shear_strength': shear,
        'grain_type': gtype, 'grain_size': gsize, 'rta': rta,
    }


# ---------------------- literal scalar port of the R reference --------------
def reference_sk38_ccl(profile, inclination=38.0, line_load=SKIER_LINE_LOAD,
                       rta_threshold=None, cap=CAP,
                       penetrated_layer='full', return_ccl=False):
    """
    Independent loop-based scalar implementation for a single profile.

    ``penetrated_layer='residual'`` switches the equivalent-slab
    treatment of the skier-penetrated layers (residual thickness, slab
    surface at the penetration front). ``rta_threshold=None`` (default)
    returns the pure index; a float applies the RTA gate on top.
    Not-computable layers (topmost, skier-penetrated) are NaN.
    """
    theta = math.radians(inclination)
    h_vert = np.asarray(profile['height'], dtype=float)
    rho = np.asarray(profile['density'], dtype=float)
    tau = np.asarray(profile['shear_strength'], dtype=float)
    gt = np.asarray(profile['grain_type'], dtype=float)
    gs = np.asarray(profile['grain_size'], dtype=float)
    rta = np.asarray(profile['rta'], dtype=float)
    n = len(h_vert)

    v = np.diff(np.concatenate([[0.0], h_vert]))      # vertical thickness
    H = h_vert * math.cos(theta)                      # slope-normal tops
    d = v * math.cos(theta)                           # slope-normal thickness

    # --- skier penetration (Index_PK) ---
    cum = 0.0
    msum = 0.0
    wsum = 0.0
    for i in range(n - 1, -1, -1):
        w = max(min(v[i], 0.30 - cum), 0.0)
        msum += rho[i] * w
        wsum += w
        cum += v[i]
        if cum >= 0.30:
            break
    pk1 = 34.6 / (msum / wsum)

    pk2 = np.inf
    cum = 0.0
    for i in range(n - 1, -1, -1):
        cum_prev = cum
        cum += v[i]
        if gt[i] == 772 and rho[i] >= 500.0:
            run = 0.0
            for j in range(i, -1, -1):
                if gt[j] == 772 and rho[j] >= 500.0:
                    run += v[j]
                    if run >= 0.03:
                        break
                else:
                    break
            if run >= 0.03:
                pk2 = cum_prev
                break
        if cum >= pk1:
            break
    pk = min(pk1, pk2)

    # --- full-slab quantities ---
    e_lyr = 5.07e9 * (rho / RHO_ICE) ** 5.13
    tau_g = np.full(n, np.nan)
    sig_n = np.full(n, np.nan)
    tau_p = np.full(n, np.nan)
    ccl = np.full(n, np.nan)
    for i in range(n - 1):
        wa = d[i + 1:].sum()
        rho_sl = (rho[i + 1:] * d[i + 1:]).sum() / wa
        e_sl = ((e_lyr[i + 1:] ** (1 / 3) * d[i + 1:]).sum() / wa) ** 3
        d_sl = H[-1] - H[i]
        tau_g[i] = rho_sl * G * d_sl * math.sin(theta)
        sig_n[i] = rho_sl * G * d_sl * math.cos(theta)
        f1 = math.floor(gt[i] / 100.0)
        if f1 == 6:
            add = 0.0
        elif f1 == 1:
            add = sig_n[i] * (8e-5 * tau[i] + 2.2e-5 * sig_n[i] + 0.056)
        else:
            add = sig_n[i] * (8e-5 * tau[i] + 0.224)
        tau_p[i] = tau[i] + add
        h_eq = d_sl * (e_sl / e_lyr[i]) ** (1 / 3)
        f_wl = 4.6e-9 * (rho[i] / RHO_ICE * gs[i] / 0.00125) ** -2
        ccl[i] = (
            math.sqrt(f_wl) * math.sqrt(h_eq * e_lyr[i] / (1 - 0.2 ** 2))
            * (-tau_g[i] + math.sqrt(
                tau_g[i] ** 2 + 2 * sig_n[i] * (tau_p[i] - tau_g[i])))
            / sig_n[i]
        )

    # --- penetrated layers (first penetrated = fp, 0-based) ---
    min_h = H[-1] - pk * math.cos(theta)
    fp = int(np.argmax(H > min_h))     # H is monotonic; pk > 0 => exists

    # equivalent-slab weights and surface for l_sk
    thick_sk = d.copy()
    if penetrated_layer == 'full':
        top_sk = H[fp]
    else:
        thick_sk[fp] = min_h - (H[fp] - d[fp])   # residual below the front
        thick_sk[fp + 1:] = 0.0
        top_sk = min_h

    # --- skier crack length on the truncated slab ---
    alpha_star = alpha_max_skier_stress(theta)
    out = np.full(n, np.nan)
    for i in range(n):
        if i == n - 1:
            continue                                   # top layer: -999
        if tau_p[i] < tau_g[i]:
            out[i] = 0.0                               # spontaneous
            continue
        if i >= fp:
            continue                                   # penetrated: -999
        w = thick_sk[i + 1:fp + 1]
        e_sl = ((e_lyr[i + 1:fp + 1] ** (1 / 3) * w).sum() / w.sum()) ** 3
        h_sk = (top_sk - H[i]) * (e_sl / e_lyr[i]) ** (1 / 3)

        def f(a, h_sk=h_sk, i=i):
            return (2 * line_load * math.cos(a) * math.sin(a) ** 2
                    * math.sin(a + theta) / (math.pi * h_sk)
                    + tau_g[i] - tau_p[i])

        if f(alpha_star) > 0:
            a1 = _bisect(f, 1e-12, alpha_star, rising=True)
            a2 = _bisect(f, alpha_star, math.pi / 2 - 1e-12, rising=False)
            lsk = h_sk * (1 / math.tan(a1) - 1 / math.tan(a2))
            out[i] = min(ccl[i] / lsk, cap)
        else:
            out[i] = cap
    if rta_threshold is not None:
        finite = np.isfinite(out)
        out[finite & (rta < rta_threshold)] = cap
    if return_ccl:
        return out, ccl
    return out


def _bisect(f, lo, hi, rising, iters=80):
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        val = f(mid)
        if (val < 0) == rising:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------- tests -----------------------------------------------
def _force_residual_convention(monkeypatch):
    """Route the index through the 'residual' penetrated-layer branch."""
    monkeypatch.setattr(
        sk38_ccl_ext, '_prepare_skier_mechanics',
        functools.partial(sk38_ext._prepare_skier_mechanics,
                          penetrated_layer='residual'),
    )


class TestAgainstScalarReference:
    @pytest.mark.parametrize('penetrated_layer', ['full', 'residual'])
    def test_matches_reference_port(self, penetrated_layer, monkeypatch):
        profiles = [default_profile(n) for n in (5, 8, 12)]
        # vary the second profile to exercise different regimes
        profiles[1]['shear_strength'] *= 3.0
        profiles[2]['rta'][::2] = 0.5
        ds = build_profile_ds(profiles)

        if penetrated_layer == 'residual':
            _force_residual_convention(monkeypatch)
        result = ds.compute_sk38_ccl()
        got = result['sk38_ccl'].data[0, :, 0, 0, :]

        for t, prof in enumerate(profiles):
            expected = reference_sk38_ccl(
                prof, penetrated_layer=penetrated_layer)
            n = len(prof['height'])
            np.testing.assert_allclose(
                got[t, :n], expected, rtol=1e-9, atol=1e-12,
                err_msg=f'profile {t} mismatch ({penetrated_layer})',
            )
            assert np.all(np.isnan(got[t, n:]))        # padding stays NaN

    def test_penetrated_layer_conventions_differ(self, monkeypatch):
        prof = default_profile(12)
        # weaken so several layers stay below the index cap
        prof['shear_strength'] = prof['shear_strength'] * 0.45
        ds = build_profile_ds([prof])
        full = ds.compute_sk38_ccl()['sk38_ccl'].data[0, 0, 0, 0]
        _force_residual_convention(monkeypatch)
        res = ds.compute_sk38_ccl()['sk38_ccl'].data[0, 0, 0, 0]
        # same computability pattern, different equivalent-slab numbers
        np.testing.assert_array_equal(np.isnan(full), np.isnan(res))
        both = (np.isfinite(full) & (full != CAP) & (full > 0)
                & np.isfinite(res) & (res != CAP) & (res > 0))
        assert both.any()
        assert not np.allclose(full[both], res[both])

    def test_invalid_penetrated_layer_raises(self):
        ds = build_profile_ds([default_profile()])
        with pytest.raises(ValueError, match='penetrated_layer'):
            sk38_ext._prepare_skier_mechanics(
                ds, penetrated_layer='partial')

    def test_rta_gated_index_matches_reference_port(self):
        profiles = [default_profile(8), default_profile(12)]
        profiles[1]['rta'][::2] = 0.5
        ds = build_profile_ds(profiles)
        result = ds.compute_sk38_ccl_rta()
        got = result['sk38_ccl_rta'].data[0, :, 0, 0, :]
        for t, prof in enumerate(profiles):
            expected = reference_sk38_ccl(prof, rta_threshold=0.8)
            n = len(prof['height'])
            np.testing.assert_allclose(
                got[t, :n], expected, rtol=1e-9, atol=1e-12,
                err_msg=f'profile {t} mismatch (gated)',
            )

    def test_gate_matches_pure_plus_filter(self):
        prof = default_profile(12)
        prof['rta'][::3] = 0.4
        ds = build_profile_ds([prof])
        pure = ds.compute_sk38_ccl()['sk38_ccl'].data[0, 0, 0, 0]
        gated = ds.compute_sk38_ccl_rta()['sk38_ccl_rta'].data[0, 0, 0, 0]
        n = len(prof['height'])
        expected = pure.copy()
        mask = np.isfinite(expected[:n]) & (prof['rta'] < 0.8)
        expected[:n][mask] = CAP
        np.testing.assert_array_equal(gated, expected)

    def test_weak_layer_is_critical_layer(self):
        prof = default_profile()
        ds = build_profile_ds([prof])
        got = ds.compute_sk38_ccl()['sk38_ccl'].data[0, 0, 0, 0]
        computable = np.isfinite(got) & (got != CAP)
        assert computable.any()
        assert np.nanargmin(np.where(computable, got, np.nan)) == 2


class TestSentinels:
    def test_rta_gate(self):
        prof = default_profile()
        prof['rta'][:] = 0.5
        ds = build_profile_ds([prof])
        got = ds.compute_sk38_ccl_rta()['sk38_ccl_rta'].data[0, 0, 0, 0]
        finite = np.isfinite(got)
        assert finite.any()
        assert np.all(got[finite] == CAP)

    def test_pure_index_ignores_rta(self):
        prof = default_profile(12)
        # weaken so several layers stay below the index cap
        prof['shear_strength'] = prof['shear_strength'] * 0.45
        prof['rta'][:] = 0.0
        ds = build_profile_ds([prof])
        pure = ds.compute_sk38_ccl()['sk38_ccl'].data[0, 0, 0, 0]
        assert np.any(
            np.isfinite(pure) & (pure != CAP) & (pure != 0.0)
        )
        # the pure index does not need an rta variable at all
        no_rta = xsnowDataset(ds.data.drop_vars('rta'))
        pure2 = no_rta.compute_sk38_ccl()['sk38_ccl'].data[0, 0, 0, 0]
        np.testing.assert_array_equal(pure, pure2)

    def test_spontaneous_failure_is_zero(self):
        prof = default_profile()
        prof['shear_strength'][2] = 1.0     # essentially no strength
        ds = build_profile_ds([prof])
        got = ds.compute_sk38_ccl()['sk38_ccl'].data[0, 0, 0, 0]
        assert got[2] == 0.0

    def test_very_strong_profile_is_stable(self):
        prof = default_profile()
        prof['shear_strength'][:] = 1.0e6
        prof['rta'][:] = 0.9
        ds = build_profile_ds([prof])
        got = ds.compute_sk38_ccl()['sk38_ccl'].data[0, 0, 0, 0]
        finite = np.isfinite(got)
        assert finite.any()
        assert np.all(got[finite] == CAP)

    def test_penetrated_and_top_layers_are_nan(self):
        prof = default_profile()
        ds = build_profile_ds([prof])
        result = ds.compute_sk38_ccl()
        got = result['sk38_ccl'].data[0, 0, 0, 0]
        thick = np.diff(np.concatenate([[0.0], prof['height']]))
        pen = float(compute_skier_penetration(
            prof['density'][None, :], thick[None, :],
            prof['grain_type'][None, :], np)[0])
        theta = math.radians(38.0)
        h_norm = prof['height'] * math.cos(theta)
        penetrated = h_norm > (h_norm[-1] - pen * math.cos(theta))
        assert penetrated[-1]                          # top always penetrated
        assert np.all(np.isnan(got[:len(prof['height'])][penetrated]))


class TestPenetration:
    def test_crust_limits_penetration(self):
        # 35 cm of light snow (rho=80) above a 4 cm dense melt-freeze
        # crust: unimpeded penetration would be 34.6/80 = 0.4325 m, the
        # crust stops the skier at 0.35 m.
        height = np.array([0.10, 0.20, 0.30, 0.34, 0.45, 0.57, 0.69])
        thick = np.diff(np.concatenate([[0.0], height]))
        density = np.array([300.0, 300.0, 300.0, 550.0, 80.0, 80.0, 80.0])
        gtype_soft = np.full(7, 330.0)
        gtype_crust = gtype_soft.copy()
        gtype_crust[3] = 772.0
        depth_to_crust = thick[4:].sum()               # 0.35 m

        pen_soft = compute_skier_penetration(
            density[None, :], thick[None, :], gtype_soft[None, :], np,
        )[0]
        pen_crust = compute_skier_penetration(
            density[None, :], thick[None, :], gtype_crust[None, :], np,
        )[0]
        assert pen_soft == pytest.approx(34.6 / 80.0)
        assert pen_crust == pytest.approx(depth_to_crust)

        # crust rule disabled -> penetration as if soft
        pen_nocrust = compute_skier_penetration(
            density[None, :], thick[None, :], gtype_crust[None, :], np,
            use_crust_limit=False,
        )[0]
        assert pen_nocrust == pytest.approx(pen_soft)

        # a too-thin crust (< 3 cm) does not stop the skier
        thick_thin = thick.copy()
        thick_thin[3] = 0.02
        pen_thin = compute_skier_penetration(
            density[None, :], thick_thin[None, :], gtype_crust[None, :], np,
        )[0]
        assert pen_thin == pytest.approx(34.6 / 80.0)


class TestSk38:
    def test_basic_properties(self):
        prof = default_profile()
        ds = build_profile_ds([prof])
        got = ds.compute_sk38()['sk38'].data[0, 0, 0, 0]
        n = len(prof['height'])
        finite = np.isfinite(got[:n])
        assert finite.any()
        assert np.all(got[:n][finite] > 0)
        assert np.isnan(got[n - 1])                    # top layer
        # the weak layer must be the least stable computable layer
        assert np.nanargmin(got[:n]) == 2

    def test_stronger_layer_scores_higher(self):
        weak = default_profile()
        strong = default_profile()
        strong['shear_strength'] = strong['shear_strength'] * 2.0
        ds = build_profile_ds([weak, strong])
        got = ds.compute_sk38()['sk38'].data[0, :, 0, 0, :]
        assert got[1, 2] > got[0, 2]


class TestProRoundTrip:
    def test_sk38_ccl_roundtrips_through_pro(self, tmp_path):
        import xsnow as xs
        from xsnow.io.base import CODE_TO_NAME_MAP

        assert CODE_TO_NAME_MAP['0608'] == 'sk38_ccl'

        ds = build_profile_ds([default_profile()])
        with_meta = ds.data.assign_coords(
            latitude=('location', [46.9]),
            longitude=('location', [10.1]),
            altitude=('location', [2300.0]),
        )
        res = xsnowDataset(with_meta).compute_sk38_ccl()
        path = tmp_path / 'SYN1.pro'
        res.to_pro(str(path))

        header = [line for line in path.read_text().splitlines()
                  if line.startswith('0608')]
        assert header and 'sk38_ccl' in header[0]

        back = xs.read(str(path))
        assert 'sk38_ccl' in back.data
        a = np.asarray(res['sk38_ccl'].data)[0, :, 0, 0]
        b = np.asarray(back.data['sk38_ccl'].data)[0, :, 0, 0]
        a = a[:, :b.shape[1]]
        finite = np.isfinite(a)
        # compact PRO float formatting keeps >= 4 significant digits
        np.testing.assert_allclose(a[finite], b[finite], rtol=1e-3)


class TestValidation:
    def test_missing_variable_raises(self):
        prof = default_profile()
        ds = build_profile_ds([prof])
        broken = xsnowDataset(ds.data.drop_vars('shear_strength'))
        with pytest.raises(ValueError, match='shear_strength'):
            broken.compute_sk38_ccl()

    def test_missing_rta_raises(self):
        prof = default_profile()
        ds = build_profile_ds([prof])
        no_rta = xsnowDataset(ds.data.drop_vars('rta'))
        with pytest.raises(ValueError, match="'rta' not found"):
            no_rta.compute_sk38_ccl_rta()

    def test_gated_index_replaces_pure_variable(self):
        prof = default_profile()
        ds = build_profile_ds([prof])
        res = ds.compute_sk38_ccl_rta()
        assert 'sk38_ccl_rta' in res.data
        assert 'sk38_ccl' not in res.data


class TestCupyApiSurface:
    def test_computation_stays_within_cupy_api(self, monkeypatch):
        """
        Execute on numpy while restricting the backend to the cupy API
        surface: every ``xp.<name>`` access is validated against the
        installed cupy module (catches e.g. the missing
        ``cupy.errstate``). Runs only where cupy is installed; a real
        GPU is not required.
        """
        cupy = pytest.importorskip('cupy')
        # sub-attribute accesses bypass the proxy; assert the one we use
        assert hasattr(cupy.maximum, 'accumulate')

        class CupySurface:
            def __getattr__(self, name):
                if not hasattr(cupy, name):
                    raise AttributeError(
                        f"cupy has no attribute {name!r}")
                return getattr(np, name)

        surface = CupySurface()
        from xsnow.extensions import critical_crack_length as ccl_ext
        for mod in (sk38_ext, sk38_ccl_ext, ccl_ext):
            monkeypatch.setattr(mod, 'get_backend', lambda ds: surface)

        ds = build_profile_ds([default_profile()])
        assert 'sk38' in ds.compute_sk38().data
        assert 'sk38_ccl_rta' in ds.compute_sk38_ccl_rta().data
        assert 'rc_flat' in ds.compute_critical_crack_length().data
        assert 'rc_slope' in ds.compute_critical_crack_length(
            inclination=38.0).data


class TestCclDispatch:
    def test_slope_ccl_matches_reference_port(self):
        prof = default_profile()
        ds = build_profile_ds([prof])
        via_ccl = ds.compute_critical_crack_length(inclination=38.0)
        got = via_ccl['rc_slope'].data[0, 0, 0, 0]
        _, expected_ccl = reference_sk38_ccl(prof, return_ccl=True)
        n = len(prof['height'])
        np.testing.assert_allclose(
            got[:n], expected_ccl, rtol=1e-9, atol=1e-12, equal_nan=True,
        )
        assert np.all(np.isnan(got[n:]))

    def test_flat_default_unchanged_and_needs_no_grain_type(self):
        ds = build_profile_ds([default_profile()])
        no_gt = xsnowDataset(ds.data.drop_vars('grain_type'))
        flat = no_gt.compute_critical_crack_length()
        assert 'rc_flat' in flat.data
        assert np.isfinite(flat['rc_flat'].data).any()
        with pytest.raises(ValueError, match='grain_type'):
            no_gt.compute_critical_crack_length(inclination=38.0)
