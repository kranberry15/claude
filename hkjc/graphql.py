"""Client for the GraphQL endpoint behind bet.hkjc.com.

HKJC only accepts the exact query texts its own site sends (anything else
returns WHITELIST_ERROR), so the queries live verbatim in queries/*.gql and
must not be edited.  The endpoint only serves the current / next meeting;
history comes from the results pages (hkjc.results).
"""
from __future__ import annotations

from pathlib import Path

from .net import Fetcher

ENDPOINT = "https://info.cld.hkjc.com/graphql/base/"
HEADERS = {"Origin": "https://bet.hkjc.com", "Referer": "https://bet.hkjc.com/"}
_Q = Path(__file__).resolve().parent / "queries"
RACE_MEETINGS = (_Q / "race_meetings.gql").read_text()
POOL_ODDS = (_Q / "pm_pools_odds.gql").read_text()
RESULT_MEETINGS = (_Q / "result_meetings.gql").read_text()


class WhitelistError(RuntimeError):
    pass


class HKJCGraphQL:
    def __init__(self, fetcher: Fetcher | None = None, post=None):
        self.fetcher = fetcher or Fetcher(min_interval=0.5)
        # `post(query, variables) -> dict` can be injected for tests
        self._post = post

    def query(self, query: str, variables: dict) -> dict:
        if self._post:
            res = self._post(query, variables)
        else:
            res = self.fetcher.post_json(ENDPOINT, {"query": query, "variables": variables}, HEADERS)
        errors = "; ".join(e.get("message", "") for e in res.get("errors") or [])
        if "WHITELIST" in errors:
            raise WhitelistError(errors)
        if errors and not res.get("data"):
            raise RuntimeError(errors)
        return res["data"]

    def meeting(self, date: str | None = None, venue: str | None = None) -> dict | None:
        """Race card for a meeting (default: the current local ST/HV meeting)."""
        if not (date and venue):
            data = self.query(RACE_MEETINGS, {})
            local = [m for m in data.get("activeMeetings") or [] if m.get("venueCode") in ("ST", "HV")]
            if not local:
                return None
            date, venue = local[0]["date"], local[0]["venueCode"]
        data = self.query(RACE_MEETINGS, {"date": date, "venueCode": venue})
        ms = data.get("raceMeetings") or []
        return ms[0] if ms else None

    def odds(self, date: str, venue: str, race_no: int, odds_types: list[str]) -> list[dict]:
        """pmPools for one race (race_no=0: every race).  Requested in pairs,
        as the site does; pools can come back in any order."""
        pools = []
        for i in range(0, len(odds_types), 2):
            data = self.query(POOL_ODDS, {"date": date, "venueCode": venue, "raceNo": race_no, "oddsTypes": odds_types[i:i + 2]})
            for m in data.get("raceMeetings") or []:
                pools += m.get("pmPools") or []
        return pools

    def dividends(self, date: str, venue: str, odds_types=("WIN", "PLA", "QIN", "QPL")) -> list[dict]:
        data = self.query(RESULT_MEETINGS, {"date": date, "venueCode": venue, "resultOddsType": list(odds_types),
                                             "foOddsTypes": [], "foFilter": []})
        ms = data.get("raceMeetings") or []
        return (ms[0].get("resPools") or []) if ms else []


def parse_pools(pools: list[dict]) -> dict:
    """{race_no: {"WIN": {no: odds}, "PLA": {...}, "QPL": {(a, b): odds}}}"""
    out: dict[int, dict] = {}
    for p in pools:
        leg = p.get("leg") or {}
        race = (leg.get("races") or [leg.get("number")])[0]
        if race is None:
            continue
        book = out.setdefault(int(race), {}).setdefault(p["oddsType"], {})
        for node in p.get("oddsNodes") or []:
            try:
                odds = float(node["oddsValue"])
            except (TypeError, ValueError):
                continue   # SCR, ---, etc.
            nums = tuple(int(x) for x in str(node["combString"]).replace("-", ",").split(",") if x.strip().isdigit())
            if len(nums) == 1:
                book[nums[0]] = odds
            elif len(nums) == 2:
                book[tuple(sorted(nums))] = odds
    return out
