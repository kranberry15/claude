#!/usr/bin/env bash
# Downloads the "Horse Racing in HK" dataset (Graham Daley, Kaggle: gdaley/hkracing),
# 6,349 HKJC races 1997-06 .. 2005-08 with final win odds, into data/.
# Tries the Kaggle CLI first, then falls back to a public GitHub mirror.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
if [[ -f data/races.csv && -f data/runs.csv ]]; then echo "data/ already populated"; exit 0; fi
if command -v kaggle >/dev/null 2>&1; then
  kaggle datasets download -d gdaley/hkracing -p data --unzip && exit 0
fi
tmp=$(mktemp -d)
git clone -q --depth 1 https://github.com/pbovard63/Predicting_Hong_Kong_Horse_Racing_Finishes.git "$tmp/m"
cp "$tmp/m/Data/races.csv" "$tmp/m/Data/runs.csv" data/
rm -rf "$tmp"
echo "fetched data/races.csv data/runs.csv"
