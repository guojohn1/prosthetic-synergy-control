"""
experiments.py — the three results this environment exists to produce.

Run: python experiments.py
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from grasps import sample_postures, train_test_split
from synergy import SynergyBasis, dimensionality_needed
from emg import EMGSimulator
from decode import calibration_curve, gestures_to_reach

def r2_per_synergy(E, C):
    Xa = np.hstack([E, np.ones((len(E),1))])
    W = np.linalg.lstsq(Xa, C, rcond=None)[0]
    return 1 - ((C - Xa@W).var(0) / C.var(0))

def main():
    poses,_ = sample_postures(n_per_grasp=14)
    print(f"postures {poses.shape[0]}, joints {poses.shape[1]}\n")

    # 1 --------------------------------------------------------------
    b2 = SynergyBasis(2).fit(poses)
    rep, agree = b2.santello_report()
    print(rep)
    print(f"  2-synergy reconstruction error: {b2.reconstruction_error(poses):.2f} deg")
    print(f"  synergies for <5 deg:           {dimensionality_needed(poses, 5.0)}\n")

    # 2 --------------------------------------------------------------
    NS = 5
    full = SynergyBasis(NS).fit(poses); C = full.encode(poses)
    print("PER-SYNERGY RECOVERY FROM 6 EMG CHANNELS")
    print(f"  {'severity':>9}" + "".join(f"{'S'+str(i+1):>7}" for i in range(NS)))
    rows = []
    for sev in [0.0, 0.25, 0.5, 0.75, 1.0]:
        r = r2_per_synergy(EMGSimulator(6, NS, severity=sev, seed=3).record(C), C)
        rows.append(r); print(f"  {sev:9.2f}" + "".join(f"{v:7.2f}" for v in r))
    print("  -> coarse synergies survive amputation, fine ones do not\n")

    # 3 --------------------------------------------------------------
    tr, te = train_test_split(poses)
    sim = EMGSimulator(6, NS, severity=0.7, noise=0.10, seed=11)
    etr, ete = sim.record(full.encode(tr)), sim.record(full.encode(te))
    basis = SynergyBasis(3).fit(poses)
    cur = calibration_curve(etr, tr, ete, te, basis,
                            sizes=[4,6,10,15,25,40,70,110], repeats=25)
    print("CALIBRATION BURDEN (severity 0.7, 6 ch, 10% noise, k=3)")
    print(f"  {'gestures':>9}{'direct':>12}{'synergy':>11}{'advantage':>11}")
    for i,n in enumerate(cur["n"]):
        d,s = cur['direct'][i], cur['synergy'][i]
        print(f"  {n:9d}{d:9.2f}deg{s:8.2f}deg{(d-s)/d*100:10.0f}%")

    # figures
    fig, ax = plt.subplots(1, 3, figsize=(15, 4), dpi=150)
    ax[0].bar(range(1,7), b2.explained_[:6]*100, color="#2b6cb0")
    ax[0].axhline(0, color="k", lw=.6)
    ax[0].set_title(f"Synergy spectrum\nPC1+2 = {b2.cumulative_variance(2)*100:.1f}%"
                    f" (Santello ~84%)", fontsize=10)
    ax[0].set_xlabel("principal component"); ax[0].set_ylabel("% variance")

    R = np.array(rows)
    for i in range(NS):
        ax[1].plot([0,.25,.5,.75,1.0], R[:,i], marker="o", ms=4,
                   label=f"synergy {i+1}")
    ax[1].set_xlabel("amputation severity"); ax[1].set_ylabel("R² recovered")
    ax[1].set_title("Amputation removes FINE synergies,\nnot coarse ones", fontsize=10)
    ax[1].legend(fontsize=7, frameon=False); ax[1].set_ylim(-0.05, 1.05)

    ax[2].plot(cur["n"], cur["direct"], marker="o", ms=4, label="direct (15 DOF)", color="#c53030")
    ax[2].plot(cur["n"], cur["synergy"], marker="s", ms=4, label="synergy (3 DOF)", color="#17a673")
    ax[2].set_xscale("log"); ax[2].set_xlabel("calibration gestures")
    ax[2].set_ylabel("joint error (deg)")
    ax[2].set_title("Synergy decoder helps in the\nlow-data regime only", fontsize=10)
    ax[2].legend(fontsize=8, frameon=False)
    for a in ax: a.spines[["top","right"]].set_visible(False)
    fig.tight_layout(); fig.savefig("results.png", bbox_inches="tight")
    print("\nwrote results.png")

if __name__ == "__main__":
    main()
