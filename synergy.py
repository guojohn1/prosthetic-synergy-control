"""
synergy.py — the low-dimensional space of human hand posture.

The scientific claim this rests on, from Santello, Flanders and Soechting
(J. Neurosci. 1998, 5 subjects, 57 objects, 15 measured joint angles):

    the first TWO principal components account for >80% of posture variance
    (they report an average of 84%), and three reach about 90%.

Components 3 through 6 carried small but real object-specific information,
which is why they described a two-level scheme: coarse synergies plus fine
corrections. That structure is the whole reason few-channel control of a
many-jointed hand is possible at all.

This module extracts that basis from hand posture data and, critically,
reports whether YOUR data reproduces their number. If it does not, say so.
"""

import numpy as np

SANTELLO_PC1_2 = 0.84      # their reported average for two components
SANTELLO_PC1_3 = 0.90      # approximate, three components


class SynergyBasis:
    """
    PCA over hand postures. Deliberately not a neural network.

    A linear basis is the right tool here for three reasons. It is what the
    original result used, so a comparison is meaningful. It inverts exactly,
    so latent-to-pose is a matrix multiply with no inference cost on an
    embedded target. And it is inspectable, which matters when the output
    drives a physical hand attached to a person.
    """

    def __init__(self, n_components=2):
        self.k = n_components
        self.mean_ = None
        self.components_ = None          # (k, n_joints)
        self.explained_ = None           # full spectrum, all components

    def fit(self, poses):
        """poses: (n_samples, n_joints) array of joint angles."""
        X = np.asarray(poses, dtype=float)
        if X.ndim != 2:
            raise ValueError("poses must be (n_samples, n_joints)")
        if X.shape[0] < X.shape[1]:
            raise ValueError(
                f"need more samples than joints, got {X.shape[0]} samples "
                f"for {X.shape[1]} joints. Record more grasps.")

        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        # SVD rather than an eigendecomposition of the covariance: better
        # conditioned, and it is what every reference implementation uses.
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        var = S ** 2
        self.explained_ = var / var.sum()
        self.components_ = Vt[: self.k]
        return self

    def encode(self, poses):
        """pose -> synergy coefficients. This is what a decoder must output."""
        return (np.atleast_2d(poses) - self.mean_) @ self.components_.T

    def decode(self, coeffs):
        """synergy coefficients -> full joint angles. One matrix multiply."""
        return np.atleast_2d(coeffs) @ self.components_ + self.mean_

    def reconstruction_error(self, poses):
        """Mean absolute joint error after a round trip through k dimensions."""
        X = np.atleast_2d(poses)
        return float(np.abs(self.decode(self.encode(X)) - X).mean())

    def cumulative_variance(self, k=None):
        k = k or self.k
        return float(self.explained_[:k].sum())

    def santello_report(self):
        """
        Compare against the published result. Print this. If your numbers are
        far off, the honest conclusion is that your posture set is unlike
        theirs (too few objects, too little variety), not that they were wrong.
        """
        v2 = float(self.explained_[:2].sum())
        v3 = float(self.explained_[:3].sum())
        lines = [
            "SYNERGY STRUCTURE vs Santello et al. 1998",
            f"  PC1+PC2   yours {v2*100:5.1f}%   published ~{SANTELLO_PC1_2*100:.0f}%",
            f"  PC1+PC2+3 yours {v3*100:5.1f}%   published ~{SANTELLO_PC1_3*100:.0f}%",
        ]
        agree = abs(v2 - SANTELLO_PC1_2) < 0.10
        lines.append("  -> consistent with the published synergy structure" if agree
                     else "  -> NOT consistent. Report this honestly rather than tuning to match.")
        return "\n".join(lines), agree


def dimensionality_needed(poses, tolerance_deg=5.0):
    """
    How many synergies to reconstruct every posture within a tolerance.

    Reported in degrees because that is the unit a prosthetist thinks in, and
    because 'we explain 84% of variance' does not tell anyone whether the
    thumb ends up in the right place.
    """
    for k in range(1, min(poses.shape) + 1):
        if SynergyBasis(k).fit(poses).reconstruction_error(poses) <= tolerance_deg:
            return k
    return poses.shape[1]
