# Synergy-based prosthetic control: a training environment

Runs today, no hardware, no downloads, no network.

```
pip install numpy scipy matplotlib mujoco
python test_env.py      # correctness, ~5 seconds
python experiments.py   # the three results + results.png
```

## The thesis

A cutting-edge prosthetic hand has ~20 DOF. A user has ~4 noisy EMG channels.
Everyone treats that as an impossible gap.

Santello, Flanders and Soechting (J. Neurosci. 1998, 5 subjects, 57 objects,
15 joint angles) found **two principal components explain about 84% of hand
posture variance.** So the hand is not functionally 20-dimensional. Four
channels is roughly the right number, *if you map to the right space.*

The basis comes from watching intact hands. It is never relearned per user.

## What the experiments actually found

**1. The low-dimensional structure reproduces.** Our postures give 86.1% on two
components against Santello's ~84%. Nothing in `grasps.py` imposes a synergy
basis: each grasp is written joint-by-joint from its functional description,
with independent per-joint noise added. The structure emerges from the task.

**2. Amputation is selective, not uniform.** This is the real result.

```
 severity     S1     S2     S3     S4     S5
     0.00   0.98   0.95   0.57   0.17   0.40
     1.00   0.78   0.70   0.29   0.12   0.26
```

Coarse synergies survive. Fine ones collapse. That matches the anatomy
(extrinsic forearm muscles survive a transradial amputation, intrinsic hand
muscles do not) and it explains the clinical reality: gross grasp works on a
myoelectric hand today, dexterity does not.

**3. The calibration-burden claim is WEAKER than advertised.** Honest result:

```
 gestures      direct    synergy  advantage
        6    42.20deg   36.78deg        13%
       15    20.29deg   18.96deg         7%
       40    14.72deg   14.50deg         1%
      110    13.48deg   13.47deg         0%
```

The synergy decoder wins by 7 to 13% in the low-data regime and the advantage
vanishes with more data. Real, but not transformative. **Report this. A team
that tests its own headline claim and reports it coming in smaller is more
credible than one that only shows what worked.**

## What the experiment taught us

The right target is not calibration time. It is that **amputation removes
exactly the fine synergies the residual limb can no longer encode**, so a
video-learned prior should be aimed at *reconstructing* those, not at saving
setup minutes. That reframing came from the data, not from the pitch.

## Files

| File | Purpose |
|---|---|
| `synergy.py` | PCA basis, Santello comparison |
| `grasps.py` | Posture data. **Stand-in. Replace with your own recordings.** |
| `emg.py` | Forward model, agonist/antagonist pairs, amputation severity |
| `decode.py` | Direct vs synergy decoders, calibration curves |
| `hand.py` | MuJoCo hand driven by two numbers |
| `experiments.py` | The three results |
| `test_env.py` | Correctness tests |

## Bugs worth knowing about, already fixed

**MuJoCo XML defaults to DEGREES.** `range="-0.2 1.7"` meant 1.7 degrees of
travel. The hand silently refused to close while every other number looked
fine. `<compiler angle="radian"/>` is load-bearing.

**A muscle can only pull.** Mapping a signed synergy coefficient onto one
muscle and then rectifying (as EMG envelopes are) destroys the sign. Splitting
into agonist/antagonist pairs took recovery R² from 0.14 to 0.77. The
physiology and the signal processing agree.

## Getting real data (the honest plan)

You do not need web video and you do not need EEG.

- **Postures:** webcam + MediaPipe on your own hands. Santello used 57 objects
  and 5 subjects. Do 40 objects and 3 teammates in one afternoon. That is a
  genuine replication and it beats a downloaded dataset because it is yours.
- **EMG:** MyoWare 2.0, $42.95, plus $9.95 electrodes. Plugs into the ESP32.
  Or use the simulator, which is why it exists.
- **EEG:** skip entirely. Best published non-invasive work gets 60% on three
  fingers. It does not have the bandwidth for coordinated control.

## What this does not show

Simulated EMG is not real EMG. An intact forearm is not a residual limb.
Nothing here is a clinical claim. It is a control architecture with a
measurement environment attached, which is what lets you find out whether the
architecture is any good before you build hardware around it.
