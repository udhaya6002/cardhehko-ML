"""
features.py - Parse unit strings, extract brand, create engineered features.

Usage:
    from src.features import engineer_features
    df = engineer_features(df)

Rules (SPEC.md):
    - No transformer is fit here (pure column transforms).
    - Operates on the raw DataFrame before train/test split.
"""

import re
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_numeric(series: pd.Series, pattern: str = r"([\d.]+)") -> pd.Series:
    """Extract the first numeric value from a string column."""
    return (
        series.astype(str)
        .str.extract(pattern, expand=False)
        .astype(float)
    )


def _parse_torque(series: pd.Series) -> pd.DataFrame:
    """
    Parse the torque column which may contain values like:
        '190Nm@ 2000rpm', '23.5 kgm at 1500-2500rpm', etc.
    Returns a DataFrame with columns: torque_nm, max_torque_rpm.
    """
    torque_nm = pd.Series(np.nan, index=series.index, name="torque_nm")
    max_torque_rpm = pd.Series(np.nan, index=series.index, name="max_torque_rpm")

    for i, val in series.items():
        if pd.isna(val) or str(val).strip() == "":
            continue
        s = str(val).lower().strip()

        # Extract Nm value
        nm_match = re.search(r"([\d,.]+)\s*nm", s)
        kgm_match = re.search(r"([\d,.]+)\s*kgm", s)

        if nm_match:
            try:
                torque_nm[i] = float(nm_match.group(1).replace(",", ""))
            except ValueError:
                pass
        elif kgm_match:
            try:
                # 1 kgm ~ 9.80665 Nm
                torque_nm[i] = float(kgm_match.group(1).replace(",", "")) * 9.80665
            except ValueError:
                pass
        else:
            # Try just a bare number (assume Nm)
            bare = re.match(r"^([\d,.]+)", s)
            if bare:
                try:
                    torque_nm[i] = float(bare.group(1).replace(",", ""))
                except ValueError:
                    pass

        # Extract RPM value (take the higher number if a range)
        rpm_match = re.findall(r"([\d,]+)\s*(?:rpm)?", s)
        rpm_values = []
        for r in rpm_match:
            try:
                v = float(r.replace(",", ""))
                if v > 100:  # RPM values are usually > 100
                    rpm_values.append(v)
            except ValueError:
                pass
        if rpm_values:
            max_torque_rpm[i] = max(rpm_values)

    return pd.DataFrame({"torque_nm": torque_nm, "max_torque_rpm": max_torque_rpm})


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering transformations to the raw DataFrame.

    Transformations:
        1. Parse mileage -> mileage_kmpl (float)
        2. Parse engine -> engine_cc (float)
        3. Parse max_power -> max_power_bhp (float)
        4. Parse torque -> torque_nm, max_torque_rpm (float)
        5. Extract brand from name
        6. Create car_age = current_year - year
        7. Create km_per_year = km_driven / car_age
        8. Drop original text columns that have been parsed

    Returns:
        DataFrame with engineered features.
    """
    df = df.copy()

    # 1. Parse mileage (e.g., "23.4 kmpl" or "18.9 km/kg")
    if "mileage" in df.columns:
        df["mileage_kmpl"] = _parse_numeric(df["mileage"])
        df.drop(columns=["mileage"], inplace=True)

    # 2. Parse engine (e.g., "1248 CC")
    if "engine" in df.columns:
        df["engine_cc"] = _parse_numeric(df["engine"])
        df.drop(columns=["engine"], inplace=True)

    # 3. Parse max_power (e.g., "82 bhp")
    if "max_power" in df.columns:
        df["max_power_bhp"] = _parse_numeric(df["max_power"])
        df.drop(columns=["max_power"], inplace=True)

    # 4. Parse torque
    if "torque" in df.columns:
        torque_df = _parse_torque(df["torque"])
        df = pd.concat([df, torque_df], axis=1)
        df.drop(columns=["torque"], inplace=True)

    # 5. Extract brand (first word of name)
    if "name" in df.columns:
        df["brand"] = df["name"].astype(str).str.split().str[0].str.strip().str.title()
        df.drop(columns=["name"], inplace=True)

    # 6. car_age
    current_year = 2024  # Fixed for reproducibility
    if "year" in df.columns:
        df["car_age"] = current_year - df["year"]
        df.drop(columns=["year"], inplace=True)

    # 7. km_per_year
    if "km_driven" in df.columns and "car_age" in df.columns:
        df["km_per_year"] = df["km_driven"] / df["car_age"].replace(0, np.nan)

    return df
