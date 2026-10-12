"""Offline checks for benter.hkjc_data parsing.   python -m tests.test_hkjc_data

The HTML below is a cut-down, synthetic imitation of a racing.hkjc.com results page.
"""
from datetime import date

from benter import hkjc_data as hd

PAGE = """
<select><option value='{"date":"07/10/2026","venue":""}'>07/10/2026</option>
<option value='{"date":"04/10/2026","venue":""}'>04/10/2026</option></select>
<table><tr><td>Happy Valley:</td></tr></table>
<table><thead><tr><td>RACE 1 (99)</td></tr></thead><tr><td>Class 4 - 1650M - (60-40)</td><td>Going : GOOD</td></tr></table>
<table><thead><tr><th>Pla.</th><th>Horse No.</th><th>Horse</th><th>Jockey</th><th>Trainer</th><th>Act. Wt.</th>
<th>Declar. Horse Wt.</th><th>Dr.</th><th>LBW</th><th>Running Position</th><th>Finish Time</th><th>Win Odds</th></tr></thead>
<tr><td>1</td><td>5</td><td>FAST ONE (K123)</td><td>Z Purton</td><td>J Size</td><td>130</td><td>1100</td><td>3</td><td>---</td><td>1 1 1</td><td>1:39.10</td><td>3.2</td></tr>
<tr><td>2 DH</td><td>2</td><td>SECOND (J456)</td><td>H T Mo</td><td>P C Ng</td><td>120</td><td>1050</td><td>1</td><td>1</td><td>2 2 2</td><td>1:39.27</td><td>8.0</td></tr>
<tr><td>PU</td><td>7</td><td>STOPPED (H789)</td><td>K Teetan</td><td>C S Shum</td><td>118</td><td>1000</td><td>9</td><td>---</td><td></td><td>---</td><td>40</td></tr>
<tr><td>WV</td><td>9</td><td>WITHDRAWN (L001)</td><td>---</td><td>F C Lor</td><td>125</td><td>1020</td><td>4</td><td>---</td><td></td><td>---</td><td>---</td></tr>
</table>
<a href="?racedate=2026/10/07&Racecourse=HV&RaceNo=2">2</a><a href="?racedate=2026/10/07&Racecourse=HV&RaceNo=3">3</a>
"""


def test_parse_race_page():
    p = hd.parse_race_page(PAGE, date(2026, 10, 7))
    assert p["venue"] == "HV" and p["race_no"] == 1 and p["races"] == [1, 2, 3]
    rows = {r["horse_id"]: r for r in p["rows"]}
    assert set(rows) == {"K123", "J456", "H789"}                 # withdrawn runner dropped
    assert rows["K123"]["finish_position"] == 1 and abs(rows["K123"]["finish_time"] - 99.10) < 1e-9
    assert rows["J456"]["finish_position"] == 2                  # dead heat
    assert rows["H789"]["finish_position"] == 99                 # pulled up: ran, no time
    assert rows["K123"]["distance"] == 1650 and rows["K123"]["race_id"] == "2026-10-07-HV-R01"


def test_meeting_dates():
    assert hd.meeting_dates(PAGE) == [date(2026, 10, 7), date(2026, 10, 4)]


def test_card_frame():
    meeting = {"date": "2026-10-14", "venueCode": "HV", "races": [{"no": 1, "distance": 1650, "runners": [
        {"no": "1", "standbyNo": "", "status": "Declared", "name_en": "A", "handicapWeight": "135", "allowance": " 5",
         "horse": {"code": "L045"}, "jockey": {"name_en": "Z Purton"}, "trainer": {"name_en": "J Size"}, "winOdds": "4.5"},
        {"no": "2", "standbyNo": "", "status": "Scratched", "name_en": "B", "handicapWeight": "130", "allowance": "",
         "horse": {"code": "L046"}, "jockey": {"name_en": "H T Mo"}, "trainer": {"name_en": "P C Ng"}, "winOdds": "SCR"},
        {"no": "3", "standbyNo": "1", "status": "Standby", "name_en": "C", "handicapWeight": "120", "allowance": "",
         "horse": {"code": "L047"}, "jockey": {"name_en": "K Teetan"}, "trainer": {"name_en": "F C Lor"}, "winOdds": ""},
    ]}]}
    c = hd.card_frame(meeting)
    assert list(c["horse_no"]) == [1]
    assert c["weight_carried"].iloc[0] == 130 and c["win_odds"].iloc[0] == 4.5


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
