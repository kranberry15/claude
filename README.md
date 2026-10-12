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

Every betting rule below was chosen on the 2002/03 season and only then run on 2003–05.

| Pseudo-R² (Benter's measure) | |
|---|---|
| Public odds | 0.1959 |
| Fundamental model | 0.1176 |
| Benter two-stage blend (α 0.036, β 1.028) | 0.1962 |
| Market-anchored blend | 0.1930 |

The public odds in this era are much sharper than in Benter's 1980s data. The classic
two-stage blend puts almost all its weight on the public.

**Place and quinella place.** The 2nd and 3rd places come from Benter's discounted
Harville formula. The discounts fitted here (0.87, 0.65) are close to the ~0.81, 0.65 his
paper reports, and the resulting place chances are well calibrated.

| Flat HK$1 bets, 2003–05 | Bets | ROI | 90% bootstrap CI |
|---|---|---|---|
| Win, every runner | 19,383 | −27.1% | −31% to −23% |
| Win rule (edge > 20%, odds ≤ 30) | 739 | −14.4% | −37% to +10% |
| Place, every runner | 15,726 | −20.9% | −23% to −19% |
| **Place rule (edge > 0%, place odds ≤ 5)** | 917 | **−6.6%** | −14% to +2% |
| Quinella place | — | not testable: the history has no QPL odds | |

The place pool is the softer pool, and the model's place picks lose far less than the take.
Neither rule shows a reliable profit, though. The backtests also use *final* odds, which you
can't bet at. The live recommender logs every odds snapshot it sees, so once a season of
real pre-race WIN/PLA/QPL prices is collected the rules (QPL included) can be re-tested.

## Tracker

`tracker/index.html` + `tracker/model.json`. Serve locally with
`cd tracker && python -m http.server`, or use the published artifact (link in the PR/session).

- **Race card**: enter each runner's form, the current win and place odds, and QPL odds for any
  pair. It shows win %, place %, the edge on each bet type, the top QPL pairs with their fair odds,
  and quarter-Kelly stakes.
- **Race log**: save a card to a shared log and enter the first three home later; it tracks
  actual vs model-expected ROI separately for win, place and QPL bets.
- **Model evidence**: the numbers above, calibration and the Kelly bankroll curve.

## Scraper and live recommender (`hkjc/`)

HKJC has no public API. Two sources are used:

- **History:** the public results pages at `racing.hkjc.com/en-us/local/information/localresults`,
  one page per race. Each page has the finishing order, weights, draw, margins, times, final win
  odds and every pool's dividends (including QPL).
  `python -m hkjc.results --start 2016-09-01` walks every day, caches raw HTML in
  `data/hkjc/html/`, and writes `data/hkjc/{races,runs,dividends}.csv` at one request a second.
  Ten years takes a few hours the first time.
- **Live odds and race cards:** the GraphQL endpoint behind bet.hkjc.com
  (`info.cld.hkjc.com/graphql/base/`). It only accepts the exact query texts the site itself sends,
  so they are kept verbatim in `hkjc/queries/` (taken from the open-source
  [jeffincn/hkjc-worker](https://github.com/jeffincn/hkjc-worker)). It only serves the current or
  next meeting.

```bash
python -m hkjc.results --start 2016-09-01                       # history (rerun weekly; cached)
python -m benter.run --data data/hkjc --out data/hkjc/model.json  # refit on recent racing
python -m hkjc.live                                              # race day: refreshes every 60 s
python -m hkjc.live --once --race 5 --bankroll 5000
```

For each open race, `hkjc.live` reads the card and the WIN, PLA and QPL odds. It builds each
runner's features from the scraped history and prints every bet that clears the fitted rules,
with a quarter-Kelly stake. All bets in a race are capped together at 10% of bankroll. Every
snapshot is appended to `data/hkjc/odds/<date>_<venue>.jsonl` for later backtesting.

**Not yet run against the live site.** It was written and tested offline, because this
development environment can't reach hkjc.com. The parsers are tested on HKJC-shaped fixtures,
and the live pipeline on a fake feed built from real races (`python -m tests.test_hkjc`). On the
first real run, check that:
1. a results page parses (`python -m hkjc.results --start <a recent race day> --end <same day>`);
2. the live feed's jockey names match the results pages. `hkjc.live` warns if most don't.

Use it for personal research, keep the rate limit, and check HKJC's terms of use.

## Reproduce

```bash
pip install -r requirements.txt
./scripts/fetch_data.sh
python -m benter.run      # fits, backtests, writes tracker/model.json
python -m tests.parity    # the page's JS reproduces the Python model (win + place) exactly
python -m tests.test_hkjc # scraper parser + live pipeline, offline
```

## Limitations

- 1997–2005 data. Today's market is sharper, with more syndicate money and rebates. Refit on
  recent data before trusting stakes. The pipeline only needs the same CSV columns.
- Jockey/trainer IDs are anonymised in the dataset, so their rates are career-to-date rather
  than the season rates the tracker asks for.
- The tracker's lengths-beaten feature is imputed from finishing positions.
- Quinella place uses the same discounted-Harville chances but has no backtest yet.
  Benter's real edge also came from trifecta-type pools, which this does not model.
