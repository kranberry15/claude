# HKJC Overlay Tracker

A Benter-style handicapping model for Hong Kong Jockey Club racing, plus a race-card
tracker page that flags runners the tote is underpricing ("overlays") and logs them so
you can see whether the edge holds up.

Method follows Bill Benter, *Computer Based Horse Race Handicapping and Wagering
Systems: A Report* (1994):

1. **Fundamental model**: a multinomial (conditional) logit over the runners in each race,
   fit on form, draw, jockey/trainer and market-memory features.
2. **Combined model**: blend the fundamental probability with the public's implied
   probability, `c ∝ exp(α·ln f + β·ln π)`, fit on races the fundamental model never saw.
3. **Bet** when `model chance × odds − 1` clears a threshold, sized by fractional Kelly.

## Data

Benter never published his data. The largest free HKJC history with odds is the Kaggle
dataset [gdaley/hkracing](https://www.kaggle.com/datasets/gdaley/hkracing): 6,349 Sha Tin and
Happy Valley races from June 1997 to August 2005, with final win odds. `scripts/fetch_data.sh`
downloads it into `data/` (Kaggle CLI if present, otherwise a public GitHub mirror). The CSVs
are git-ignored.

Splits (by date, no shuffling):

| Period | Use |
|---|---|
| 1997-06 → 1998-08 | warm-up: history only, so features aren't all "debut" |
| 1998-09 → 2002-08 | fit fundamental model (3,016 races) |
| 2002-09 → 2003-08 | fit α, β blend (769 races) |
| 2003-09 → 2005-08 | held-out test (1,543 races) |

## Results on the held-out seasons

| Pseudo-R² (Benter's measure) | |
|---|---|
| Public odds | 0.1959 |
| Fundamental model | 0.1176 |
| Benter two-stage blend (α 0.036, β 1.028) | 0.1962 |
| Market-anchored blend | 0.1961 |

The public odds in this era are much sharper than in Benter's 1980s data. The classic
two-stage blend puts almost all its weight on the public and never finds a positive edge
after the ~17.5% take.

The **market-anchored** variant fits the fundamentals jointly with `ln(public p)`, so each
coefficient measures what the public *misprices*. Its weight on the public is 1.155 (> 1):
favourites win more often than their odds say (the favourite–longshot bias). It flags a thin
tail of overlays:

| Min edge | Max odds | Bets | ROI | 90% bootstrap CI |
|---|---|---|---|---|
| every runner | 99 | 19,383 | −27.1% | −31% to −23% |
| +0% | 30 | 1,376 | −1.7% | −17% to +14% |
| +20% | 30 | 272 | +9.6% | −33% to +53% |
| +30% | 30 | 116 | +58.4% | −8% to +142% |
| +50% | 30 | 24 | +194.6% | +13% to +416% |

Treat this as weak evidence. The profitable buckets are small and most intervals include
losing. The backtest also uses *final* odds, which you can't bet at. That is why the tracker
logs every flagged bet: the live record is the real test.

## Tracker

`tracker/index.html` + `tracker/model.json`. Serve locally with
`cd tracker && python -m http.server`, or use the published artifact (link in the PR/session).

- **Race card**: enter each runner's draw, rating, weight, body weight, record, last 6
  finishes, last 3 starting odds, jockey/trainer season W/rides and the current win odds. It shows
  public %, form-only %, model %, edge and a quarter-Kelly stake (capped at 10% of bankroll per race).
- **Race log**: save a card to a shared log and enter the winner later; it tracks actual vs
  model-expected ROI on flagged overlays.
- **Model evidence**: the numbers above, calibration and the Kelly bankroll curve.

## Reproduce

```bash
pip install -r requirements.txt
./scripts/fetch_data.sh
python -m benter.run      # fits, backtests, writes tracker/model.json
python -m tests.parity    # checks the page's JS reproduces the Python model exactly
```

## Limitations

- 1997–2005 data. Today's market is sharper, with more syndicate money and rebates. Refit on
  recent data before trusting stakes. The pipeline only needs the same CSV columns.
- Jockey/trainer IDs are anonymised in the dataset, so their rates are career-to-date rather
  than the season rates the tracker asks for.
- The tracker's lengths-beaten feature is imputed from finishing positions.
- Win pool only. Benter's real edge also came from exotic pools (quinella, trifecta),
  which this does not model.
