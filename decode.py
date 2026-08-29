"""
decode.py — EMG to hand posture, two ways, so they can be compared fairly.

DIRECT       EMG -> all joint angles.        What most pattern-recognition
                                             systems attempt. Every joint is a
                                             separate thing to learn per user.

SYNERGY      EMG -> k coefficients -> pose.  The basis comes from video of
                                             intact hands and is never
                                             relearned. Only the small EMG-to-
                                             coefficient map is per-user.

The whole argument is that the second needs far less data from the person
wearing the device. This file exists to test that rather than assert it.
"""

import numpy as np


def _ridge(X, Y, lam=1e-2):
    """
    Ridge regression with an intercept.

    Regularized rather than plain least squares on purpose: the regime that
    matters is a handful of calibration gestures against several EMG channels,
    where ordinary least squares is unstable or underdetermined. Choosing a
    method that degrades gracefully with tiny n is part of the design, not an
    implementation detail.
    """
    X = np.atleast_2d(X); Y = np.atleast_2d(Y)
    Xa = np.hstack([X, np.ones((X.shape[0], 1))])
    A = Xa.T @ Xa + lam * np.eye(Xa.shape[1])
    A[-1, -1] -= lam                                  # do not penalize intercept
    return np.linalg.solve(A, Xa.T @ Y)


def _apply(W, X):
    X = np.atleast_2d(X)
    return np.hstack([X, np.ones((X.shape[0], 1))]) @ W


class DirectDecoder:
    """EMG straight to joint angles. The baseline."""

    def fit(self, emg, poses, lam=1e-2):
        self.W = _ridge(emg, poses, lam)
        return self

    def predict(self, emg):
        return _apply(self.W, emg)


class SynergyDecoder:
    """
    EMG to synergy coefficients, then the fixed basis expands to joint angles.

    The basis is passed in already fitted. That is the point: it came from
    video of other people's hands and costs the user nothing.
    """

    def __init__(self, basis):
        self.basis = basis

    def fit(self, emg, poses, lam=1e-2):
        self.W = _ridge(emg, self.basis.encode(poses), lam)
        return self

    def predict(self, emg):
        return self.basis.decode(_apply(self.W, emg))


def joint_error(pred, true):
    """Mean absolute joint error in degrees. The unit a clinician thinks in."""
    return float(np.abs(np.atleast_2d(pred) - np.atleast_2d(true)).mean())


def calibration_curve(emg_train, pose_train, emg_test, pose_test,
                      basis, sizes, repeats=12, seed=0):
    """
    Error against number of calibration gestures, for both decoders.

    This is the experiment the project turns on. If the synergy decoder reaches
    usable error with ten gestures while the direct decoder still needs
    hundreds, the argument holds. If it does not, the argument fails and you
    should say so on the slide rather than quietly dropping the comparison.
    """
    rng = np.random.default_rng(seed)
    out = {"n": [], "direct": [], "synergy": [], "direct_sd": [], "synergy_sd": []}

    for n in sizes:
        if n > len(emg_train):
            continue
        d_errs, s_errs = [], []
        for _ in range(repeats):
            idx = rng.choice(len(emg_train), n, replace=False)
            e, p = emg_train[idx], pose_train[idx]
            d_errs.append(joint_error(DirectDecoder().fit(e, p).predict(emg_test), pose_test))
            s_errs.append(joint_error(SynergyDecoder(basis).fit(e, p).predict(emg_test), pose_test))
        out["n"].append(n)
        out["direct"].append(float(np.mean(d_errs)))
        out["synergy"].append(float(np.mean(s_errs)))
        out["direct_sd"].append(float(np.std(d_errs)))
        out["synergy_sd"].append(float(np.std(s_errs)))
    return out


def gestures_to_reach(curve, target_deg, key="synergy"):
    """Fewest calibration gestures that got under a target error. None if never."""
    for n, e in zip(curve["n"], curve[key]):
        if e <= target_deg:
            return n
    return None
