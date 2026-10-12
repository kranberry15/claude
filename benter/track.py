"""Paper record: settle every saved tip against the official result.

    python -m benter.track

Tips are saved by benter.tips to data/hkjc/tips/<date>_<venue>.csv (the last
slip shown before each race's pools closed).  Results and dividends come from
benter.hkjc_data.  The record is written to data/hkjc/track_record.md.
"""
from __future__ import annotations

import pandas as pd

from benter.hkjc_data import DATA

TIPS = DATA / "tips"
TIP_COLS = ["saved_at", "race_id", "bet_type", "selection", "name", "model_p", "odds", "edge", "stake_hkd", "trusted"]


def save_tips(slip: pd.DataFrame, day: str, venue: str, races_open: set[str], trusted: bool, saved_at: str):
    """Replace the saved tips of the races still selling with this slip's lines;
    races already closed keep the tips they had when they closed."""
    TIPS.mkdir(parents=True, exist_ok=True)
    path = TIPS / f"{day}_{venue}.csv"
    old = pd.read_csv(path, dtype={"selection": str}) if path.exists() else pd.DataFrame(columns=TIP_COLS)
    new = slip.assign(saved_at=saved_at, trusted=trusted)
    new = new[new["race_id"].isin(races_open)][TIP_COLS] if len(new) else pd.DataFrame(columns=TIP_COLS)
    keep = old[~old["race_id"].isin(races_open)]
    out = pd.concat([d for d in (keep, new) if len(d)] or [pd.DataFrame(columns=TIP_COLS)], ignore_index=True)
    out.sort_values(["race_id", "bet_type", "selection"]).to_csv(path, index=False)
    return path


def settle() -> pd.DataFrame:
    """Every saved tip with its return (None until the result is downloaded)."""
    files = sorted(TIPS.glob("*.csv"))
    if not files:
        return pd.DataFrame(columns=TIP_COLS + ["returned"])
    tips = pd.concat([pd.read_csv(f, dtype={"selection": str}) for f in files], ignore_index=True)
    divs = [pd.read_csv(f, dtype={"selection": str}) for f in sorted((DATA / "dividends").glob("*.csv"))]
    div = pd.concat(divs, ignore_index=True) if divs else pd.DataFrame(columns=["race_id", "bet_type", "selection", "dividend"])
    settled = set(div["race_id"])
    pay = div.groupby(["race_id", "bet_type", "selection"])["dividend"].max()
    tips["returned"] = [
        (r.stake_hkd / 10.0 * pay.get((r.race_id, r.bet_type, r.selection), 0.0)) if r.race_id in settled else None
        for r in tips.itertuples()
    ]
    return tips


def report(tips: pd.DataFrame) -> str:
    done = tips[tips["returned"].notna()]
    lines = ["# Paper record", "", f"Tips saved: {len(tips)}   settled: {len(done)}   waiting for results: {len(tips) - len(done)}", ""]
    if done.empty:
        return "\n".join(lines + ["Nothing settled yet."])
    def row(label, d):
        s, r = d["stake_hkd"].sum(), d["returned"].sum()
        hits = int((d["returned"] > 0).sum())
        return f"| {label} | {len(d)} | {hits} | {s:.0f} | {r:.0f} | {r - s:+.0f} | {(r - s) / s:+.1%} |" if s else ""
    lines += ["| | Bets | Winners | Staked HKD | Returned HKD | Profit HKD | ROI |", "|---|---|---|---|---|---|---|",
              row("All", done)]
    lines += [row(b, d) for b, d in done.groupby("bet_type")]
    lines += [row("Model trusted" if t else "Paper-only", d) for t, d in done.groupby("trusted")]
    lines += ["", "Expected profit from the model's own edges: HKD "
              f"{(done['stake_hkd'] * done['edge']).sum():+.0f} (compare with actual above).", "",
              "## Every settled tip", "", "| Race | Bet | Sel | Odds at tip | Stake | Returned |", "|---|---|---|---|---|---|"]
    lines += [f"| {r.race_id} | {r.bet_type} | {r.selection} | {r.odds} | {r.stake_hkd:.0f} | {r.returned:.0f} |"
              for r in done.itertuples()]
    return "\n".join(l for l in lines if l is not None)


def main():
    text = report(settle())
    (DATA / "track_record.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
