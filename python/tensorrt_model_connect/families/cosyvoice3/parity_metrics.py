# SPDX-License-Identifier: Apache-2.0
"""Family-local parity diagnostics; the caller's numerical gate is unchanged."""

import numpy as np


def compare_outputs(actual, expected, *, atol, rtol):
    """Reject invalid comparisons and explain an elementwise allclose result."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or actual.size == 0:
        raise ValueError("Parity outputs must have identical, nonempty shapes")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Parity outputs must both be finite")
    difference = np.abs(actual - expected)
    tolerance = atol + rtol * np.abs(expected)
    return {
        "passed": bool(np.allclose(actual, expected, atol=atol, rtol=rtol)),
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
        "max_tolerance_ratio": float((difference / tolerance).max()),
        "elements_over_tolerance": int(np.count_nonzero(difference > tolerance)),
        "total_elements": int(difference.size),
        "max_abs_error_by_batch": [float(value.max()) for value in difference],
    }
