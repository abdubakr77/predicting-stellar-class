"""
Feature engineering exactly matching the report's three pillars:

  1. Colour-colour diagrams      -> 7 colour indices (u-g, g-r, r-i, i-z, u-r, g-i, u-z)
  2. Stellar Locus Geometry      -> perpendicular distance to g-r = 0.1474*(u-g) + 0.3133
  3. Sky-position features       -> cyclic sin/cos encoding of `alpha` + 24x18 spatial bin grid
"""

import numpy as np
import pandas as pd

# --- Stellar locus fit reported for this competition ---
STELLAR_LOCUS_SLOPE = 0.1474
STELLAR_LOCUS_INTERCEPT = 0.3133

N_ALPHA_BINS = 24   # 15 degrees each -> matches the 24 "hour" divisions of right ascension
N_DELTA_BINS = 18   # 10 degrees each across the 180-degree declination range


def add_color_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 7 colour indices used to cancel out distance/extinction effects."""
    df = df.copy()
    df["color_ug"] = df["u"] - df["g"]
    df["color_gr"] = df["g"] - df["r"]
    df["color_ri"] = df["r"] - df["i"]
    df["color_iz"] = df["i"] - df["z"]
    df["color_ur"] = df["u"] - df["r"]
    df["color_gi"] = df["g"] - df["i"]
    df["color_uz"] = df["u"] - df["z"]
    return df


def add_stellar_locus_distance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perpendicular distance of each object from the empirical stellar locus
    line  g-r = 0.1474*(u-g) + 0.3133   in (u-g, g-r) colour space.

    Large distances flag objects that deviate from "normal" point-source
    stellar behaviour -- i.e. distant galaxies and high-redshift quasars.
    Requires add_color_indices() to have been run first (uses color_ug, color_gr).
    """
    df = df.copy()
    if "color_ug" not in df.columns or "color_gr" not in df.columns:
        df = add_color_indices(df)
    m, b = STELLAR_LOCUS_SLOPE, STELLAR_LOCUS_INTERCEPT
    # distance from point (x0, y0) to line y = m*x + b  ==  |m*x0 - y0 + b| / sqrt(m^2 + 1)
    x0, y0 = df["color_ug"], df["color_gr"]
    df["stellar_locus_dist"] = np.abs(m * x0 - y0 + b) / np.sqrt(m**2 + 1)
    # signed version preserves which side of the locus the object falls on
    df["stellar_locus_signed"] = (m * x0 - y0 + b) / np.sqrt(m**2 + 1)
    return df


def add_cyclic_sky_coords(df: pd.DataFrame) -> pd.DataFrame:
    """
    Right ascension (alpha) is a 0-360 degree circular coordinate, so 359.9 deg
    and 0.1 deg are spatially adjacent even though numerically far apart.
    Encode as sin/cos to preserve that circular continuity for tree models.
    """
    df = df.copy()
    alpha_rad = np.deg2rad(df["alpha"])
    df["alpha_sin"] = np.sin(alpha_rad)
    df["alpha_cos"] = np.cos(alpha_rad)
    return df


def add_spatial_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    432-region sky grid: alpha split into 24 bins of 15 deg (matching the
    24-hour right-ascension convention), delta split into 18 bins of 10 deg.
    """
    df = df.copy()
    alpha_bin = np.floor(df["alpha"] / 15.0).clip(0, N_ALPHA_BINS - 1).astype(int)
    # delta spans roughly [-90, 90]; shift to [0, 180] before binning into 18x10-degree bins
    delta_shifted = (df["delta"] + 90.0).clip(0, 180 - 1e-6)
    delta_bin = np.floor(delta_shifted / 10.0).clip(0, N_DELTA_BINS - 1).astype(int)
    df["alpha_bin"] = alpha_bin
    df["delta_bin"] = delta_bin
    df["sky_region"] = alpha_bin * N_DELTA_BINS + delta_bin  # 0..431
    return df


def encode_raw_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode spectral_type / galaxy_population for models that need numeric input (LightGBM/XGBoost).
    CatBoost can consume the raw string columns directly via its cat_features argument."""
    df = df.copy()
    for col, mapping in [
        ("spectral_type", {"O/B": 0, "A/F": 1, "G/K": 2, "M": 3}),
        ("galaxy_population", {"Blue_Cloud": 0, "Red_Sequence": 1}),
    ]:
        if col in df.columns:
            df[f"{col}_enc"] = df[col].map(mapping).fillna(-1).astype(int)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full report-aligned feature-engineering pipeline in order."""
    df = add_color_indices(df)
    df = add_stellar_locus_distance(df)
    df = add_cyclic_sky_coords(df)
    df = add_spatial_bins(df)
    df = encode_raw_categoricals(df)
    return df


def get_feature_columns(df: pd.DataFrame, target_col: str = "class", id_col: str = "id") -> list[str]:
    """All numeric model-ready feature columns, excluding id/target/raw string categoricals."""
    drop_cols = {target_col, id_col, "spectral_type", "galaxy_population"}
    return [c for c in df.columns if c not in drop_cols and df[c].dtype != object and str(df[c].dtype) != "string"]


train_feat = build_features(train_df)
test_feat  = build_features(test_df)

TARGET_COL = "class"
ID_COL = "id"

y = encode_target(train_df)
feature_cols = get_feature_columns(train_feat, TARGET_COL, ID_COL)
cat_features = [c for c in ["sky_region", "alpha_bin", "delta_bin"] if c in feature_cols]

X = train_feat[feature_cols]
X_test = test_feat[feature_cols]

print(f"{len(feature_cols)} features:")
print(feature_cols)
X.head()


