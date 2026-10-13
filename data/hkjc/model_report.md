# Model report (2026-10-13)

## History

- Meetings: 712 (2018-09-02 to 2026-10-07), races: 6,779, runner rows: 82,434.
- Five scheduled meetings returned no runners and were saved as `.none`
  (abandoned meetings): 2018-09-16, 2019-09-18, 2019-11-13, 2022-11-02, 2023-10-08.
- No download errors.

## Held-out pseudo-R^2 (1,060 test races; train 3,368, blend 1,355)

| model       | pseudo-R^2 | vs public |
|-------------|-----------:|----------:|
| public      | 0.1883     |           |
| fundamental | 0.1168     | -0.0715   |
| benter      | 0.1887     | +0.0004   |
| anchored    | 0.1897     | +0.0014   |

The blend gives the fundamental model a slightly *negative* weight
(alpha -0.167, beta 1.208), so the "benter" gain mostly comes from sharpening
the public odds rather than from the fundamental factors.

## Fit output (a), verbatim

```
races: train 3368  blend 1355  test 1060

Fundamental conditional logit (coef per 1 sd of the factor):
  factor               coef    s.e.       z
  speed_rating       +0.236   0.044    +5.4
  first_start        +0.009   0.027    +0.3
  weight_disadv      -0.090   0.088    -1.0
  rating_rel         +0.019   0.089    +0.2
  log_days           -0.040   0.037    -1.1
  layoff             +0.013   0.032    +0.4
  quick_backup       +0.010   0.025    +0.4
  jockey_logit       +0.318   0.020   +16.2
  trainer_logit      +0.086   0.022    +4.0
  jt_interaction     +0.036   0.023    +1.6
  last_pos           -0.223   0.031    -7.2
  avg_pos3           -0.066   0.051    -1.3
  log_starts         -0.094   0.027    -3.4
  win_rate           +0.116   0.021    +5.4
  last_log_odds      -0.189   0.041    -4.7
  avg_log_odds3      -0.168   0.042    -4.0
  dist_change        -0.015   0.024    -0.6
  abs_dist_change    -0.060   0.024    -2.5
  avg_lbw3           +0.016   0.050    +0.3
  draw_frac          -0.110   0.028    -3.9
  draw_hv            -0.252   0.037    -6.8
  draw_sprint        -0.000   0.038    -0.0
  class_drop         +0.017   0.022    +0.8
  body_wt_change     -0.034   0.020    -1.7

Combined model: alpha (fundamental) -0.167   beta (public) 1.208
Held-out pseudo-R^2:  public 0.1883   fundamental 0.1168   benter 0.1887   anchored 0.1897
benter {'races': {'train': 3368, 'blend': 1355, 'test': 1060}, 'r2_public': np.float64(0.18830699093265868), 'r2_fundamental': np.float64(0.11677368122100595), 'r2_benter': np.float64(0.18866483548664403), 'r2_anchored': np.float64(0.18967219820456793), 'r2_combined': np.float64(0.18866483548664403)}
races: train 3368  blend 1355  test 1060

Fundamental conditional logit (coef per 1 sd of the factor):
  factor               coef    s.e.       z
  speed_rating       +0.236   0.044    +5.4
  first_start        +0.009   0.027    +0.3
  weight_disadv      -0.090   0.088    -1.0
  rating_rel         +0.019   0.089    +0.2
  log_days           -0.040   0.037    -1.1
  layoff             +0.013   0.032    +0.4
  quick_backup       +0.010   0.025    +0.4
  jockey_logit       +0.318   0.020   +16.2
  trainer_logit      +0.086   0.022    +4.0
  jt_interaction     +0.036   0.023    +1.6
  last_pos           -0.223   0.031    -7.2
  avg_pos3           -0.066   0.051    -1.3
  log_starts         -0.094   0.027    -3.4
  win_rate           +0.116   0.021    +5.4
  last_log_odds      -0.189   0.041    -4.7
  avg_log_odds3      -0.168   0.042    -4.0
  dist_change        -0.015   0.024    -0.6
  abs_dist_change    -0.060   0.024    -2.5
  avg_lbw3           +0.016   0.050    +0.3
  draw_frac          -0.110   0.028    -3.9
  draw_hv            -0.252   0.037    -6.8
  draw_sprint        -0.000   0.038    -0.0
  class_drop         +0.017   0.022    +0.8
  body_wt_change     -0.034   0.020    -1.7

Combined model: alpha (fundamental) -0.167   beta (public) 1.208
Held-out pseudo-R^2:  public 0.1883   fundamental 0.1168   benter 0.1887   anchored 0.1897
anchored {'races': {'train': 3368, 'blend': 1355, 'test': 1060}, 'r2_public': np.float64(0.18830699093265868), 'r2_fundamental': np.float64(0.11677368122100595), 'r2_benter': np.float64(0.18866483548664403), 'r2_anchored': np.float64(0.18967219820456793), 'r2_combined': np.float64(0.18967219820456793)}
```

## Replay: `python -m benter.tips --replay 40 --combine benter`, verbatim

```
2026-03-29: 0 bets, staked 0, returned 0   (model 0.196 vs odds 0.195)
2026-04-01: 0 bets, staked 0, returned 0   (model 0.200 vs odds 0.199)
2026-04-06: 0 bets, staked 0, returned 0   (model 0.201 vs odds 0.199)
2026-04-08: 0 bets, staked 0, returned 0   (model 0.199 vs odds 0.197)
2026-04-12: 0 bets, staked 0, returned 0   (model 0.200 vs odds 0.198)
2026-04-15: 0 bets, staked 0, returned 0   (model 0.199 vs odds 0.198)
2026-04-19: 0 bets, staked 0, returned 0   (model 0.197 vs odds 0.196)
2026-04-22: 0 bets, staked 0, returned 0   (model 0.195 vs odds 0.194)
2026-04-26: 0 bets, staked 0, returned 0   (model 0.194 vs odds 0.193)
2026-04-29: 0 bets, staked 0, returned 0   (model 0.196 vs odds 0.195)
2026-05-03: 0 bets, staked 0, returned 0   (model 0.196 vs odds 0.195)
2026-05-06: 0 bets, staked 0, returned 0   (model 0.199 vs odds 0.197)
2026-05-09: 0 bets, staked 0, returned 0   (model 0.200 vs odds 0.198)
2026-05-13: 0 bets, staked 0, returned 0   (model 0.193 vs odds 0.192)
2026-05-17: 0 bets, staked 0, returned 0   (model 0.193 vs odds 0.192)
2026-05-20: 0 bets, staked 0, returned 0   (model 0.194 vs odds 0.193)
2026-05-24: 0 bets, staked 0, returned 0   (model 0.194 vs odds 0.193)
2026-05-27: 0 bets, staked 0, returned 0   (model 0.196 vs odds 0.194)
2026-05-31: 0 bets, staked 0, returned 0   (model 0.196 vs odds 0.195)
2026-06-03: 0 bets, staked 0, returned 0   (model 0.195 vs odds 0.194)
2026-06-07: 0 bets, staked 0, returned 0   (model 0.193 vs odds 0.192)
2026-06-10: 0 bets, staked 0, returned 0   (model 0.191 vs odds 0.190)
2026-06-13: 0 bets, staked 0, returned 0   (model 0.192 vs odds 0.191)
2026-06-21: 0 bets, staked 0, returned 0   (model 0.191 vs odds 0.191)
2026-06-24: 0 bets, staked 0, returned 0   (model 0.193 vs odds 0.192)
2026-06-27: 0 bets, staked 0, returned 0   (model 0.191 vs odds 0.191)
2026-07-01: 0 bets, staked 0, returned 0   (model 0.193 vs odds 0.192)
2026-07-04: 0 bets, staked 0, returned 0   (model 0.192 vs odds 0.191)
2026-07-08: 0 bets, staked 0, returned 0   (model 0.192 vs odds 0.192)
2026-07-12: 0 bets, staked 0, returned 0   (model 0.192 vs odds 0.191)
2026-07-15: 0 bets, staked 0, returned 0   (model 0.192 vs odds 0.191)
2026-09-06: 0 bets, staked 0, returned 0   (model 0.190 vs odds 0.190)
2026-09-09: 0 bets, staked 0, returned 0   (model 0.189 vs odds 0.189)
2026-09-13: 0 bets, staked 0, returned 0   (model 0.190 vs odds 0.190)
2026-09-16: 0 bets, staked 0, returned 0   (model 0.190 vs odds 0.189)
2026-09-23: 0 bets, staked 0, returned 0   (model 0.192 vs odds 0.191)
2026-09-27: 0 bets, staked 0, returned 0   (model 0.192 vs odds 0.191)
2026-10-01: 0 bets, staked 0, returned 0   (model 0.190 vs odds 0.190)
2026-10-04: 0 bets, staked 0, returned 0   (model 0.190 vs odds 0.190)
2026-10-07: 0 bets, staked 0, returned 0   (model 0.188 vs odds 0.188)


TOTAL over 40 meetings: staked HKD 0, returned HKD 0, profit HKD +0
```

## Replay: `python -m benter.tips --replay 40 --combine anchored`, verbatim

```
2026-03-29: 0 bets, staked 0, returned 0   (model 0.197 vs odds 0.195)
2026-04-01: 0 bets, staked 0, returned 0   (model 0.201 vs odds 0.199)
2026-04-06: 0 bets, staked 0, returned 0   (model 0.201 vs odds 0.199)
2026-04-08: 0 bets, staked 0, returned 0   (model 0.199 vs odds 0.197)
2026-04-12: 0 bets, staked 0, returned 0   (model 0.200 vs odds 0.198)
2026-04-15: 0 bets, staked 0, returned 0   (model 0.200 vs odds 0.198)
2026-04-19: 0 bets, staked 0, returned 0   (model 0.198 vs odds 0.196)
2026-04-22: 0 bets, staked 0, returned 0   (model 0.196 vs odds 0.194)
2026-04-26: 1 bets, staked 10, returned 0   (model 0.195 vs odds 0.193)
2026-04-29: 0 bets, staked 0, returned 0   (model 0.196 vs odds 0.195)
2026-05-03: 0 bets, staked 0, returned 0   (model 0.196 vs odds 0.195)
2026-05-06: 0 bets, staked 0, returned 0   (model 0.199 vs odds 0.197)
2026-05-09: 0 bets, staked 0, returned 0   (model 0.199 vs odds 0.198)
2026-05-13: 0 bets, staked 0, returned 0   (model 0.193 vs odds 0.192)
2026-05-17: 0 bets, staked 0, returned 0   (model 0.193 vs odds 0.192)
2026-05-20: 0 bets, staked 0, returned 0   (model 0.194 vs odds 0.193)
2026-05-24: 0 bets, staked 0, returned 0   (model 0.195 vs odds 0.193)
2026-05-27: 0 bets, staked 0, returned 0   (model 0.196 vs odds 0.194)
2026-05-31: 0 bets, staked 0, returned 0   (model 0.197 vs odds 0.195)
2026-06-03: 0 bets, staked 0, returned 0   (model 0.196 vs odds 0.194)
2026-06-07: 0 bets, staked 0, returned 0   (model 0.194 vs odds 0.192)
2026-06-10: 0 bets, staked 0, returned 0   (model 0.192 vs odds 0.190)
2026-06-13: 0 bets, staked 0, returned 0   (model 0.193 vs odds 0.191)
2026-06-21: 0 bets, staked 0, returned 0   (model 0.193 vs odds 0.191)
2026-06-24: 0 bets, staked 0, returned 0   (model 0.194 vs odds 0.192)
2026-06-27: 0 bets, staked 0, returned 0   (model 0.192 vs odds 0.191)
2026-07-01: 0 bets, staked 0, returned 0   (model 0.194 vs odds 0.192)
2026-07-04: 0 bets, staked 0, returned 0   (model 0.193 vs odds 0.191)
2026-07-08: 0 bets, staked 0, returned 0   (model 0.193 vs odds 0.192)
2026-07-12: 0 bets, staked 0, returned 0   (model 0.193 vs odds 0.191)
2026-07-15: 0 bets, staked 0, returned 0   (model 0.193 vs odds 0.191)
2026-09-06: 0 bets, staked 0, returned 0   (model 0.191 vs odds 0.190)
2026-09-09: 0 bets, staked 0, returned 0   (model 0.190 vs odds 0.189)
2026-09-13: 0 bets, staked 0, returned 0   (model 0.191 vs odds 0.190)
2026-09-16: 0 bets, staked 0, returned 0   (model 0.191 vs odds 0.189)
2026-09-23: 0 bets, staked 0, returned 0   (model 0.192 vs odds 0.191)
2026-09-27: 0 bets, staked 0, returned 0   (model 0.192 vs odds 0.191)
2026-10-01: 0 bets, staked 0, returned 0   (model 0.191 vs odds 0.190)
2026-10-04: 0 bets, staked 0, returned 0   (model 0.191 vs odds 0.190)
2026-10-07: 0 bets, staked 0, returned 0   (model 0.189 vs odds 0.188)
  2026-04-26-ST-R07  WIN No. 1.0 odds   3.0  HKD 10 -> 0

TOTAL over 40 meetings: staked HKD 10, returned HKD 0, profit HKD -10
```
