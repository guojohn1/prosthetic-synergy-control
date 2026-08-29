"""
emg.py — a forward model from intended hand posture to recorded EMG.

Why simulate rather than only record: you cannot ask an amputee to produce
labelled data, which is precisely the problem the project exists to address.
A simulator with an explicit, defensible amputation model lets you ASK the
transfer question quantitatively before you ever touch a person.

The anatomy this is built on, because it is what makes the severity axis
mean something rather than being a knob labelled "harder":

  EXTRINSIC hand muscles (flexor digitorum superficialis and profundus,
  extensor digitorum) sit in the FOREARM. They survive a transradial
  amputation, and they drive gross flexion and extension. This is the entire
  reason surface myoelectric control works at all.

  INTRINSIC hand muscles (interossei, lumbricals, thenar group) sit IN THE
  HAND. They are lost with the hand, and they are what produce independent
  finger movement and fine posture.

So amputation does not attenuate the signal uniformly. It removes the fine,
high-synergy-order content and leaves the coarse content. That is the shape
of the real problem, and it is what this model reproduces.
"""

import numpy as np


class EMGSimulator:
    """
    Intended synergy coefficients in, EMG channel envelopes out.

    severity: 0.0 = intact limb, all muscles present
              1.0 = only extrinsic forearm muscles remain
    """

    def __init__(self, n_channels=4, n_synergies=3, severity=0.0,
                 noise=0.05, crosstalk=0.25, seed=0):
        if not 0.0 <= severity <= 1.0:
            raise ValueError("severity must be in [0, 1]")
        self.n_channels = n_channels
        self.n_synergies = n_synergies
        self.severity = severity
        self.noise = noise
        self.rng = np.random.default_rng(seed)

        # Synergy -> muscle, through AGONIST/ANTAGONIST PAIRS.
        #
        # A muscle can only pull. Flexion and extension are produced by
        # different muscles, so a signed synergy coefficient must be split:
        # its positive part drives one muscle group, its negative part drives
        # the opposing group. Both resulting activations are non-negative.
        #
        # This is not a modelling nicety. Collapsing a signed coefficient onto
        # a single muscle and then rectifying (as real EMG envelopes are)
        # destroys the sign, and with it most of the recoverable information.
        # The physiology and the signal processing agree here.
        n_pairs = 4
        self.n_muscles = 2 * n_pairs               # agonist + antagonist each
        self.n_extrinsic = 4                       # first 4 survive amputation
        M = np.abs(self.rng.normal(0, 1, (n_pairs, n_synergies))) + 0.2
        for s_i in range(n_synergies):
            intrinsic_bias = s_i / max(n_synergies - 1, 1)
            M[:2, s_i] *= (1.0 - 0.7 * intrinsic_bias)     # extrinsic pairs
            M[2:, s_i] *= (0.3 + 0.7 * intrinsic_bias)     # intrinsic pairs
        self.synergy_to_pair = M

        # Muscle -> electrode. Surface electrodes are not selective: each one
        # picks up several nearby muscles. That is volume conduction, and it
        # is why raw channel count overstates how much independent
        # information you actually have.
        A = np.abs(self.rng.normal(0, 1, (n_channels, self.n_muscles)))
        A = A + crosstalk * A.mean()
        self.muscle_to_electrode = A / A.sum(axis=1, keepdims=True)

        self.shift = np.zeros((n_channels, self.n_muscles))

    def apply_electrode_shift(self, magnitude=0.3):
        """
        Perturb the muscle-to-electrode map.

        This is the single most cited reason deployed decoders degrade: the
        socket moves on the limb, the electrodes move relative to the muscles,
        and a decoder calibrated yesterday is wrong today. Being able to
        simulate it is how you test whether a low-dimensional decoder is more
        robust than a high-dimensional one, which is a real claim to make.
        """
        p = self.rng.normal(0, magnitude, self.muscle_to_electrode.shape)
        A = np.abs(self.muscle_to_electrode + p * self.muscle_to_electrode.mean())
        self.muscle_to_electrode = A / A.sum(axis=1, keepdims=True)

    def _surviving(self):
        """Per-muscle gain after amputation, over all agonist/antagonist slots."""
        g = np.ones(self.n_muscles)
        g[self.n_extrinsic:] *= (1.0 - self.severity)      # intrinsics fade out
        # Residual-limb atrophy and scarring degrade what remains, mildly.
        g[: self.n_extrinsic] *= (1.0 - 0.25 * self.severity)
        return g

    def record(self, coeffs):
        """
        coeffs: (n_samples, n_synergies) intended synergy activations.
        returns: (n_samples, n_channels) EMG envelopes, non-negative.
        """
        C = np.atleast_2d(np.asarray(coeffs, dtype=float))
        if C.shape[1] != self.n_synergies:
            raise ValueError(f"expected {self.n_synergies} synergies, got {C.shape[1]}")

        # Split each signed synergy into agonist (positive part) and
        # antagonist (negative part) drive. Both are non-negative, as muscle
        # activation must be.
        pair_drive = C @ self.synergy_to_pair.T        # (n, n_pairs), signed
        muscle = np.concatenate([np.maximum(pair_drive, 0),
                                 np.maximum(-pair_drive, 0)], axis=1)
        muscle = muscle * self._surviving()

        e = muscle @ self.muscle_to_electrode.T
        # Multiplicative plus additive noise: EMG amplitude noise scales with
        # activation, and there is a baseline floor from electronics and motion.
        e = e * (1 + self.rng.normal(0, self.noise, e.shape))
        e = e + np.abs(self.rng.normal(0, self.noise * max(e.std(), 1e-9), e.shape))
        return np.maximum(e, 0.0)

    def information_retained(self, n_probe=400):
        """
        How much of the intended synergy signal actually survives to the
        electrodes at this severity. Measured, not assumed.

        Reported as the mean canonical correlation between intended synergy
        coefficients and recorded channels. It is the honest answer to
        "does this still work on an amputee", within the model's assumptions.
        """
        # Probe with realistically scaled coefficients. Zero-mean unit-variance
        # probes are not what a hand produces and give a misleading answer.
        C = self.rng.normal(0, 1, (n_probe, self.n_synergies)) * \
            np.linspace(3.0, 1.0, self.n_synergies)
        E = self.record(C)
        Cc = C - C.mean(0)
        Ec = E - E.mean(0)
        # canonical correlation via whitened cross-covariance
        def whiten(X):
            U, S, _ = np.linalg.svd(X, full_matrices=False)
            return U[:, S > 1e-9]
        Qc, Qe = whiten(Cc), whiten(Ec)
        s = np.linalg.svd(Qc.T @ Qe, compute_uv=False)
        return float(np.clip(s, 0, 1).mean())
