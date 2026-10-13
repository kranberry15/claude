# Notes for Claude sessions

The owner is not a programmer. They message in plain English ("give me tonight's
Happy Valley tips") and expect plain-English answers, not code. Do the running yourself.

## Asked for tips

1. `pip install -q -r requirements.txt` (fresh containers lack scipy / lxml).
2. `python -m benter.tips` - refreshes results, fetches the next meeting's card and live
   odds, prints the slip (bankroll HKD 200, quarter-Kelly, HKD 10 units). For specific
   races later in the night just run it again; it only uses races still selling.
3. Reply with the slip as printed (race, post time, bet type, horse number + name, HKD),
   plus the PAPER ONLY warning if it appears. Remind them odds move until the off.
4. Commit and push `data/hkjc/` (the tips file is the paper record) to the working branch.

If it says betting hasn't opened, tell them when to try again (odds usually open the day before).

## Paper record

- `data/hkjc/tips/<date>_<venue>.csv`: last tips shown before each race closed.
- `python -m benter.track` settles them against `data/hkjc/dividends/` and writes
  `data/hkjc/track_record.md`. Read it before discussing how the system is doing, and
  report it honestly, losses included. Small samples prove nothing either way.
- A daily scheduled job downloads new results (`python -m benter.hkjc_data`), runs
  `benter.track`, and commits `data/hkjc/`.

## Model status (2026-10-13)

Full history 2018-09 to 2026-10 (712 meetings, 6,779 races), 24 factors. On 1,060 held-out
races: public odds 0.1883, Benter blend 0.1887, market-anchored 0.1897 (pseudo-R^2). Neither
clears the trust margin (+0.005), the blend puts ~zero/negative weight on form, and a
40-meeting walk-forward replay made 0 bets (benter) / 1 losing bet (anchored). So every slip is
PAPER ONLY. Place tips that do appear come mostly from the place-probability formula, not form,
and are untested. Details: `data/hkjc/model_report.md`. Say this plainly if asked whether to bet.

## Code map

- `benter/hkjc_data.py`: results scraper (racing.hkjc.com, 1 page/s) + upcoming card
  (GraphQL, exact whitelisted query in `benter/queries/race_meetings.graphql`).
- `benter/live_odds.py`: Win / Place / Quinella odds poller.
- `benter/weekend_slip.py`: features, conditional logit, Benter blend, Kelly slip.
- `benter/tips.py`: one-command tips; `--live` keeps watching, `--replay N` paper-tests past meetings.
- Tests: `python -m tests.test_weekend_slip`, `tests.test_live_odds`, `tests.test_hkjc_data`.
- Results pages hold no official ratings, so `rating` = weight carried, in history and card alike.
- Dividends are only stored for meetings downloaded after 2026-10-12.
