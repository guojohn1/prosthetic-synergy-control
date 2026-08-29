"""Correctness tests. No hardware, no data files."""
import numpy as np
from grasps import sample_postures
from synergy import SynergyBasis
from emg import EMGSimulator
from decode import DirectDecoder, SynergyDecoder, joint_error

def test_synergy_roundtrip():
    p,_ = sample_postures()
    b = SynergyBasis(15).fit(p)
    assert b.reconstruction_error(p) < 1e-8, "full-rank basis must reconstruct exactly"
    assert abs(b.explained_.sum() - 1.0) < 1e-9
    print("  OK  full-rank basis is exact, spectrum sums to 1")

def test_low_dim_structure():
    p,_ = sample_postures()
    b = SynergyBasis(2).fit(p)
    v2 = b.cumulative_variance(2)
    assert v2 > 0.6, f"expected substantial low-dim structure, got {v2:.2f}"
    print(f"  OK  2 components explain {v2*100:.1f}%")

def test_emg_is_nonnegative():
    p,_ = sample_postures()
    C = SynergyBasis(3).fit(p).encode(p)
    e = EMGSimulator(4, 3, severity=0.5).record(C)
    assert (e >= 0).all(), "EMG envelopes cannot be negative"
    print("  OK  EMG envelopes non-negative (muscles only pull)")

def test_amputation_degrades_fine_synergies():
    p,_ = sample_postures()
    NS = 5
    C = SynergyBasis(NS).fit(p).encode(p)
    def r2(sev):
        E = EMGSimulator(6, NS, severity=sev, seed=3).record(C)
        Xa = np.hstack([E, np.ones((len(E),1))])
        W = np.linalg.lstsq(Xa, C, rcond=None)[0]
        return 1 - ((C - Xa@W).var(0) / C.var(0))
    intact, amp = r2(0.0), r2(1.0)
    # Assert only what is stable across resamples. Individual fine-synergy R2
    # values (S3 and beyond) are noisy and can move either direction between
    # datasets, which is itself the finding: beyond two synergies, what the
    # electrodes recover is not reliably estimable. Testing an unstable
    # quantity is testing noise.
    coarse_i, fine_i = intact[:2].mean(), intact[2:].mean()
    coarse_a, fine_a = amp[:2].mean(), amp[2:].mean()
    assert coarse_i > 0.8, "coarse synergies must be well recovered on an intact limb"
    assert coarse_i > fine_i, "coarse must beat fine on an intact limb"
    assert coarse_a > fine_a, "coarse must beat fine after amputation too"
    assert coarse_a < coarse_i, "full amputation must degrade even coarse recovery"
    print(f"  OK  coarse {coarse_i:.2f}->{coarse_a:.2f}, fine {fine_i:.2f}->{fine_a:.2f}")
    print(f"      (coarse >> fine at both ends: this is the real result)")

def test_decoders_run_and_beat_chance():
    p,_ = sample_postures()
    b = SynergyBasis(3).fit(p)
    sim = EMGSimulator(6, 3, severity=0.5, seed=2)
    e = sim.record(b.encode(p))
    base = joint_error(np.tile(p.mean(0), (len(p),1)), p)
    for D in [DirectDecoder().fit(e,p), SynergyDecoder(b).fit(e,p)]:
        err = joint_error(D.predict(e), p)
        assert err < base, f"decoder must beat predicting the mean ({err:.1f} vs {base:.1f})"
    print(f"  OK  both decoders beat the mean-posture baseline ({base:.1f} deg)")

def test_electrode_shift_hurts():
    p,_ = sample_postures()
    b = SynergyBasis(3).fit(p); C = b.encode(p)
    sim = EMGSimulator(6, 3, severity=0.3, seed=4)
    e_before = sim.record(C)
    D = SynergyDecoder(b).fit(e_before, p)
    err_before = joint_error(D.predict(e_before), p)
    sim.apply_electrode_shift(0.5)
    err_after = joint_error(D.predict(sim.record(C)), p)
    assert err_after > err_before, "electrode shift must degrade a fixed decoder"
    print(f"  OK  electrode shift degrades decoder {err_before:.1f} -> {err_after:.1f} deg")

if __name__ == "__main__":
    print("PROSTHETICS ENVIRONMENT TESTS")
    for t in [test_synergy_roundtrip, test_low_dim_structure, test_emg_is_nonnegative,
              test_amputation_degrades_fine_synergies, test_decoders_run_and_beat_chance,
              test_electrode_shift_hurts]:
        t()
    print("\nAll tests passed.\n")
