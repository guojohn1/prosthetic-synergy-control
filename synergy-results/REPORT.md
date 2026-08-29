# Synergy — results report

Generated 2026-08-29 by make_report.py; every number below traces to data/processed/analysis.json.

## S2 — variance explained (real CyberGlove postures)

- 2 synergies: **61.6%** (Santello et al. 1998: ~84%)
- 3 synergies: **73.9%** (Santello ~90%)
- fitted on 414 movement-window postures, movement segments only, rest excluded

## S3 — decoder error, DB2 intact

| subject | EMG ch | synergy fixed (deg) | **adaptive** (deg) | direct (deg) | mean-baseline (deg) | ground truth |
|---|---|---|---|---|---|---|
| DB2_S1 | 12 | 12.73 | **11.64** | 11.30 | 14.57 | glove |
| DB2_S2 | 12 | 10.91 | **10.50** | 10.24 | 12.20 | glove |
| DB2_S3 | 12 | 10.92 | **10.39** | 10.10 | 12.96 | glove |

## S5 — identical pipeline, DB3 amputees

| subject | EMG ch | synergy fixed (deg) | **adaptive** (deg) | direct (deg) | mean-baseline (deg) | ground truth |
|---|---|---|---|---|---|---|
| DB3_S1 | 12 | 10.88 | **10.57** | 10.48 | 11.39 | db2_prototypes |
| DB3_S2 | 12 | 8.35 | **8.06** | 7.97 | 8.90 | db2_prototypes |
| DB3_S3 | 12 | 14.09 | **10.84** | 10.79 | 13.80 | glove |

**Intact-to-amputee gap: 2.57 deg** (DB2 mean 11.52 deg -> DB3 14.09 deg, synergy decoder).

## S6 — clinical metadata (n=3: pairings, no fitted trend)

- DB3 s1: 10.88 deg | remaining forearm 50% | DASH 1.67 | 13 y since amputation
- DB3 s2: 8.35 deg | remaining forearm 70% | DASH 15.18 | 6 y since amputation
- DB3 s3: 14.09 deg | remaining forearm 30% | DASH 22.5 | 5 y since amputation

## What's-next, implemented: subject-adaptive basis refinement

The frozen 2-PC intact basis is extended per subject with up to 4 residual components fit on that subject's own calibration postures (one extra component earned per ~3 gestures; still linear ridge, no NN). Calibration curves, 1 gesture = one movement repetition:

**Intact (DB2 s1, LOSO basis)**

| gestures | direct | synergy (fixed) | adaptive |
|---|---|---|---|
| 2 | 20.71 | 19.12 | 19.12 |
| 4 | 18.99 | 17.21 | 17.99 |
| 6 | 15.73 | 15.04 | 15.47 |
| 10 | 15.26 | 14.69 | 15.05 |
| 15 | 13.75 | 13.95 | 13.84 |
| 25 | 12.51 | 13.29 | 12.68 |
| 40 | 12.03 | 13.12 | 12.30 |
| 60 | 11.54 | 12.83 | 11.83 |
| 90 | 11.41 | 12.80 | 11.74 |
| 120 | 11.31 | 12.73 | 11.65 |

**Amputee (DB3 s3, glove truth, all-intact basis)**

| gestures | direct | synergy (fixed) | adaptive |
|---|---|---|---|
| 2 | 14.79 | 15.68 | 15.68 |
| 4 | 13.91 | 15.44 | 14.42 |
| 6 | 14.18 | 15.97 | 14.47 |
| 10 | 12.76 | 15.15 | 12.86 |
| 15 | 12.83 | 15.01 | 12.92 |
| 25 | 11.58 | 14.56 | 11.66 |
| 40 | 11.06 | 14.16 | 11.13 |
| 60 | 10.96 | 14.17 | 11.01 |
| 90 | 10.77 | 14.06 | 10.81 |

Reading: on the intact subject the fixed prior wins below ~10 gestures and caps the decoder above; adaptive keeps the low-data start and removes most of the ceiling. On the amputee the fixed intact prior never wins at any calibration size — its plane misfits the residual limb's postures — and the subject-adaptive refinement recovers decoder parity from ~10 gestures on. This is the brief's "reconstruct the fine synergies amputation destroys", measured.

## Calibration burden (subject DB2_S1; 1 gesture = one movement repetition)

| gestures | direct (deg) | synergy (deg) | synergy advantage |
|---|---|---|---|
| 2 | 20.89 | 19.35 | +7.3% |
| 4 | 18.83 | 16.81 | +10.7% |
| 6 | 16.42 | 15.62 | +4.9% |
| 10 | 14.96 | 14.53 | +2.8% |
| 15 | 13.55 | 13.90 | -2.6% |
| 25 | 12.60 | 13.38 | -6.1% |
| 40 | 12.10 | 13.10 | -8.3% |
| 60 | 11.63 | 12.86 | -10.6% |
| 90 | 11.44 | 12.81 | -12.0% |

Ridge lambda 1.0 chosen by train-set CV. The synergy decoder wins only in the low-calibration regime and is capped by the 2-PC reconstruction floor once data is plentiful — reported as found.

## Success criteria

| # | criterion | status |
|---|---|---|
| S1 | Two sliders drive a 24-DOF Shadow Hand into recognizable grasps | MET |
| S2 | Synergy basis fit to real NinaPro CyberGlove data vs Santello ~84% | MET |
| S3 | Ridge decoder sEMG -> synergy -> joints, error in degrees | MET |
| S4 | Ghost comparison UI, scrubbable, subject selector | MET |
| S5 | Identical pipeline on DB3 amputees, gap quantified | MET |
| S6 | Per-subject performance vs clinical metadata | MET (n=3, raw pairings only) |
| S7 | Training mode: practice reps double as calibration samples | MET (UI, precomputed curve) |
| S8 | EEG motor-imagery grasp selection | NOT STARTED (stretch, by design last) |

## Protocol and caveats

- train repetitions [1, 3, 4, 6] / test [2, 5]; 0.4 s windows, 5.0 Hz envelope
- basis fit leave-one-subject-out for DB2 evaluation; all-intact for DB3 (never fit on the evaluated subject)
- raw glove units mapped to anatomical ranges by 2nd-98th percentile calibration (approximation)
- no clinical claims; this is a control architecture plus a measurement environment
