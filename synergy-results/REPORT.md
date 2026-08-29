# Synergy — results report

Generated 2026-08-29 by make_report.py; every number below traces to data/processed/analysis.json.

## S2 — variance explained (real CyberGlove postures)

- 2 synergies: **61.6%** (Santello et al. 1998: ~84%)
- 3 synergies: **73.9%** (Santello ~90%)
- fitted on 414 movement-window postures, movement segments only, rest excluded

## S3 — decoder error, DB2 intact

| subject | EMG ch | synergy err (deg) | direct err (deg) | mean-baseline (deg) | ground truth |
|---|---|---|---|---|---|
| DB2_S1_E2_A1 | 12 | 12.46 | 11.13 | 14.13 | glove |
| DB2_S2_E2_A1 | 12 | 11.08 | 10.40 | 12.31 | glove |
| DB2_S3_E2_A1 | 12 | 11.94 | 11.11 | 14.04 | glove |

## S5 — identical pipeline, DB3 amputees

| subject | EMG ch | synergy err (deg) | direct err (deg) | mean-baseline (deg) | ground truth |
|---|---|---|---|---|---|
| DB3_S1_E2_A1 | 12 | 8.19 | 7.83 | 8.65 | db2_prototypes |
| DB3_S2_E2_A1 | 12 | 8.20 | 7.90 | 8.41 | db2_prototypes |
| DB3_S3_E2_A1 | 12 | 14.04 | 10.79 | 13.80 | glove |

**Intact-to-amputee gap: 2.21 deg** (DB2 mean 11.83 deg -> DB3 mean 14.04 deg, synergy decoder).

## S6 — clinical metadata (n=3: pairings, no fitted trend)

- DB3 s1: 8.19 deg | remaining forearm 50% | DASH 1.67 | 13 y since amputation
- DB3 s2: 8.20 deg | remaining forearm 70% | DASH 15.18 | 6 y since amputation
- DB3 s3: 14.04 deg | remaining forearm 30% | DASH 22.5 | 5 y since amputation

## Calibration burden (subject DB2_S1_E2_A1; 1 gesture = one movement repetition)

| gestures | direct (deg) | synergy (deg) | synergy advantage |
|---|---|---|---|
| 2 | 27.28 | 22.40 | +17.9% |
| 4 | 21.16 | 19.17 | +9.4% |
| 6 | 15.89 | 15.27 | +3.9% |
| 10 | 14.68 | 14.60 | +0.6% |
| 15 | 13.61 | 13.86 | -1.8% |
| 25 | 12.37 | 13.19 | -6.6% |
| 40 | 11.74 | 12.81 | -9.2% |
| 60 | 11.29 | 12.53 | -11.0% |
| 90 | 11.14 | 12.47 | -11.9% |

Ridge lambda 0.01 chosen by train-set CV. The synergy decoder wins only in the low-calibration regime and is capped by the 2-PC reconstruction floor once data is plentiful — reported as found.

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
