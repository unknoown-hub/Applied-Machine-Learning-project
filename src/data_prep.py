"""
Stage 1 — Data preparation.

Reads PRMN and ACLED raw files, cleans and merges them into a balanced
74-district × 343-week panel, and saves three artefacts:

  data/interim/prmn_panel.parquet   — weekly outflows per ORIGIN district
  data/interim/acled_panel.parquet  — weekly conflict event counts per district
  data/processed/panel.parquet      — merged panel with log(1+outflows) target

Target variable: y = log(1 + outflows_from_district_i_in_week_t)
This matches Z&T's outflow specification — conflict in district i predicts
departure FROM district i, not arrivals.
"""
from __future__ import annotations

import datetime
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT  = Path(__file__).parent.parent
RAW   = ROOT / 'data' / 'raw'
INTER = ROOT / 'data' / 'interim'
PROC  = ROOT / 'data' / 'processed'

# ── Canonical 74-district list (from ACLED admin2 — Z&T definition) ───────────
CANONICAL_DISTRICTS = [
    'Adan Yabaal', 'Afgooye', 'Afmadow', 'Baardheere', 'Badhaadhe', 'Baki',
    'Balcad', 'Banadir', 'Bandarbeyla', 'Baraawe', 'Baydhaba', 'Belet Weyne',
    'Belet Xaawo', 'Berbera', 'Borama', 'Bossaso', "Bu'aale", 'Bulo Burto',
    'Burco', 'Burtinle', 'Buuhoodle', 'Buur Hakaba', 'Cabudwaaq', 'Cadaado',
    'Cadale', 'Caluula', 'Caynabo', 'Ceel Afweyn', 'Ceel Barde', 'Ceel Buur',
    'Ceel Dheer', 'Ceel Waaq', 'Ceerigaabo', 'Dhuusamarreeb', 'Diinsoor',
    'Doolow', 'Eyl', 'Gaalkacyo', 'Galdogob', 'Garbahaarey', 'Garoowe',
    'Gebiley', 'Hargeysa', 'Hobyo', 'Iskushuban', 'Jalalaqsi', 'Jamaame',
    'Jariiban', 'Jilib', 'Jowhar', 'Kismaayo', 'Kurtunwaarey', 'Laas Caanood',
    'Laasqoray', 'Lughaye', 'Luuq', 'Marka', 'Owdweyne', 'Qandala',
    'Qansax Dheere', 'Qardho', 'Qoryooley', 'Rab Dhuure', 'Saakow', 'Sablaale',
    'Sheikh', 'Taleex', 'Tayeeglow', 'Waajid', 'Wanla Weyn', 'Xarardheere',
    'Xudun', 'Xudur', 'Zeylac',
]
CANONICAL_SET = set(CANONICAL_DISTRICTS)

# ── PRMN spelling variants → canonical name ───────────────────────────────────
PRMN_TO_CANONICAL = {
    'Baidoa':   'Baydhaba',
    'Kismayo':  'Kismaayo',
    'Lasqoray': 'Laasqoray',
}

# ── ACLED event_type → short column name ──────────────────────────────────────
ACLED_EVENT_MAP = {
    'Battles':                    'battles',
    'Explosions/Remote violence': 'explosions',
    'Strategic developments':     'strategic_dev',
    'Violence against civilians': 'viol_civ',
}

# ── Panel grid: 343 ISO weeks, 2017-W01 to 2023-W30 ──────────────────────────
PANEL_WEEKS = pd.date_range(start='2017-01-02', periods=343, freq='W-MON')


# ── Week parsing helpers ───────────────────────────────────────────────────────

def _parse_yrweek(yw: int) -> tuple:
    """Decode PRMN's YYYYWW integer to (iso_year, iso_week)."""
    s = str(int(yw))
    return int(s[:4]), int(s[4:])


def _yrweek_to_monday(year: int, week: int) -> pd.Timestamp:
    """Return the Monday of a given ISO year-week, handling week-0 and week-53 edge cases."""
    if week == 0:
        year -= 1
        week = 52
    elif week == 53:
        # Only valid if that year genuinely contains ISO week 53
        last = datetime.date(year, 12, 28)
        if last.isocalendar()[1] < 53:
            year += 1
            week = 1
    return pd.Timestamp(datetime.date.fromisocalendar(year, week, 1))


# ── 1. Load PRMN ───────────────────────────────────────────────────────────────

def load_prmn() -> pd.DataFrame:
    """
    Aggregate PRMN displacement records to weekly outflows per ORIGIN district.

    Uses PreviousDistrict (where people fled FROM), not CurrentMapDistrict.
    Rows with no PreviousDistrict, or whose district is not in the canonical
    74-district list, are dropped.

    Returns columns: district, week_start, outflows.
    """
    df = pd.read_excel(
        RAW / 'UNHCR-PRMN-Displacement-Dataset.xlsx',
        usecols=['Year', 'YrWeek', 'PreviousDistrict', 'TotalIndividuals'],
    )

    # Keep only records within the panel window
    df = df[(df['Year'] >= 2017) & (df['Year'] <= 2023)].copy()
    df = df.dropna(subset=['PreviousDistrict'])

    # Harmonise district names; drop PRMN-only districts (Badhan, Dhahar)
    df['district'] = df['PreviousDistrict'].replace(PRMN_TO_CANONICAL)
    df = df[df['district'].isin(CANONICAL_SET)]

    # Convert YrWeek integer to ISO week start date
    parsed = df['YrWeek'].map(_parse_yrweek)
    df['week_start'] = [_yrweek_to_monday(y, w) for y, w in parsed]

    # Keep only weeks inside the panel grid
    df = df[df['week_start'].isin(set(PANEL_WEEKS))]

    # Sum individuals per (origin district, week)
    agg = (
        df.groupby(['district', 'week_start'], as_index=False)['TotalIndividuals']
        .sum()
        .rename(columns={'TotalIndividuals': 'outflows'})
    )
    return agg


# ── 2. Load ACLED ──────────────────────────────────────────────────────────────

def load_acled() -> pd.DataFrame:
    """
    Aggregate ACLED conflict events to weekly counts per district, one column
    per event type: battles, explosions, strategic_dev, viol_civ.

    Returns columns: district, week_start, battles, explosions, strategic_dev, viol_civ.
    """
    df = pd.read_csv(
        RAW / 'acled_somalia_2017_2023.csv',
        sep=';',
        encoding='utf-8-sig',
        usecols=['event_date', 'event_type', 'admin2'],
    )

    # Keep only the four Z&T event types; drop Protests/Riots
    df = df[df['event_type'].isin(ACLED_EVENT_MAP)].copy()
    df = df.dropna(subset=['admin2'])
    df['event_col'] = df['event_type'].map(ACLED_EVENT_MAP)

    # Parse event_date → ISO week start (Monday)
    df['event_date'] = pd.to_datetime(df['event_date'])
    iso = df['event_date'].dt.isocalendar()
    df['week_start'] = [
        pd.Timestamp(datetime.date.fromisocalendar(int(y), int(w), 1))
        for y, w in zip(iso['year'], iso['week'])
    ]

    # Filter to panel window and canonical districts
    df = df[df['week_start'].isin(set(PANEL_WEEKS))]
    df['district'] = df['admin2']
    df = df[df['district'].isin(CANONICAL_SET)]

    # Count events; pivot so each event type is a column
    counts = (
        df.groupby(['district', 'week_start', 'event_col'])
        .size()
        .reset_index(name='count')
    )
    pivoted = counts.pivot_table(
        index=['district', 'week_start'],
        columns='event_col',
        values='count',
        fill_value=0,
    ).reset_index()
    pivoted.columns.name = None

    # Guarantee all four columns exist even if some event types never appear
    for col in ['battles', 'explosions', 'strategic_dev', 'viol_civ']:
        if col not in pivoted.columns:
            pivoted[col] = 0

    return pivoted[['district', 'week_start', 'battles', 'explosions', 'strategic_dev', 'viol_civ']]


# ── 3. Build the balanced panel ────────────────────────────────────────────────

def build_panel(prmn: pd.DataFrame, acled: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs the full 74-district × 343-week = 25,382-row balanced panel.

    Missing PRMN records → outflows = 0 (Z&T 'incl. zeros' convention).
    Missing ACLED records → conflict counts = 0.
    Target y = log(1 + outflows), matching Z&T's log-transformed count target.
    """
    # Full Cartesian grid — every district appears in every week
    grid = pd.MultiIndex.from_product(
        [CANONICAL_DISTRICTS, PANEL_WEEKS],
        names=['district', 'week_start'],
    ).to_frame(index=False)

    # Merge outflows; weeks with no PRMN record → 0
    panel = grid.merge(prmn, on=['district', 'week_start'], how='left')
    panel['outflows'] = panel['outflows'].fillna(0).astype(int)

    # Merge conflict counts; weeks with no ACLED record → 0
    panel = panel.merge(acled, on=['district', 'week_start'], how='left')
    for col in ['battles', 'explosions', 'strategic_dev', 'viol_civ']:
        panel[col] = panel[col].fillna(0).astype(int)

    # Log-transformed target
    panel['y'] = np.log1p(panel['outflows'])

    # Sequential week index 1–343 for walk-forward splits
    week_map = {w: i + 1 for i, w in enumerate(PANEL_WEEKS)}
    panel['week_num'] = panel['week_start'].map(week_map)

    return panel.sort_values(['district', 'week_start']).reset_index(drop=True)


# ── 4. Sanity assertions ───────────────────────────────────────────────────────

def run_assertions(panel: pd.DataFrame) -> None:
    """Assert key structural properties of the panel; print totals for cross-checking with Z&T."""
    assert panel.shape[0] == 74 * 343, \
        f"Expected {74 * 343} rows, got {panel.shape[0]}"

    assert panel.duplicated(subset=['district', 'week_start']).sum() == 0, \
        "Duplicate (district, week) keys found"

    for col in ['battles', 'explosions', 'strategic_dev', 'viol_civ', 'outflows']:
        assert (panel[col] >= 0).all(), f"Negative values in column '{col}'"

    assert panel['district'].nunique() == 74, \
        f"Expected 74 districts, got {panel['district'].nunique()}"

    assert panel['week_start'].min() == pd.Timestamp('2017-01-02'), \
        f"First week wrong: {panel['week_start'].min()}"
    assert panel['week_start'].max() == pd.Timestamp('2023-07-24'), \
        f"Last week wrong: {panel['week_start'].max()}"

    assert panel['week_num'].min() == 1 and panel['week_num'].max() == 343, \
        "week_num range wrong"

    total_out   = int(panel['outflows'].sum())
    total_acled = int(panel[['battles', 'explosions', 'strategic_dev', 'viol_civ']].sum().sum())
    zero_pct    = (panel['outflows'] == 0).mean() * 100

    print(f"  Total PRMN outflow individuals : {total_out:,}  (Z&T target ≈ 8 million)")
    print(f"  Total ACLED conflict events    : {total_acled:,}  (Z&T target ≈ 19,000)")
    print(f"  Zero-outflow cells             : {zero_pct:.1f}%")
    print(f"  Districts present              : {panel['district'].nunique()}")
    print(f"  Date range                     : {panel['week_start'].min().date()} → {panel['week_start'].max().date()}")
    print("  All assertions passed.")


# ── 5. Entry point ─────────────────────────────────────────────────────────────

def build_and_save() -> pd.DataFrame:
    print("Loading PRMN (origin-district outflows) …")
    prmn = load_prmn()
    prmn.to_parquet(INTER / 'prmn_panel.parquet', index=False)
    print(f"  Saved data/interim/prmn_panel.parquet  [{len(prmn):,} records]")

    print("Loading ACLED (conflict event counts) …")
    acled = load_acled()
    acled.to_parquet(INTER / 'acled_panel.parquet', index=False)
    print(f"  Saved data/interim/acled_panel.parquet  [{len(acled):,} records]")

    print("Building balanced 74×343 panel …")
    panel = build_panel(prmn, acled)

    print("Running sanity checks …")
    run_assertions(panel)

    panel.to_parquet(PROC / 'panel.parquet', index=False)
    print(f"  Saved data/processed/panel.parquet  [{panel.shape[0]} rows × {panel.shape[1]} cols]")
    return panel


if __name__ == '__main__':
    build_and_save()
