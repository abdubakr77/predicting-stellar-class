import numpy as np
import pandas as pd

# --- Stellar locus fit reported for this competition ---
STELLAR_LOCUS_SLOPE = 0.1474
STELLAR_LOCUS_INTERCEPT = 0.3133


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


def add_redshift_transform(df: pd.DataFrame):
    df = df.copy()
    df['log1p_redshift'] = np.log1p(df['redshift'])
    return df


def add_redshift_interaction(df: pd.DataFrame):
    df = df.copy()
    required_cols = ["color_ug", "color_gr"]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f'{missing} Not Found! Please Make sure that you added all the 7 colour indices from add_color_indices function!')
    
    df["redshift_inter_ug"] = (
        df["redshift"] * df["color_ug"]
    )
    df["redshift_inter_gr"] = (
        df["redshift"] * df["color_gr"]
    )

    df['redshift_x_z'] = df['redshift'] * df['z']
    return df


def add_qso_color_region(df: pd.DataFrame):
    df = df.copy()

    required_cols = ["color_ug", "color_gr"]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f'{missing} Not Found! Please Make sure that you added all the 7 colour indices from add_color_indices function!')

    df['quasar_color_region'] = ((df['color_ug'] < 0.6) & (df['color_gr'] > -0.15) & (df['color_gr'] < 0.7)).astype(int)

    return df


def add_brightness_stat(df:pd.DataFrame):
    df = df.copy()

    filters = ["u", "g", "r", "i", "z"]

    df["mag_mean"] = df[filters].mean(axis=1)
    df["mag_max"] = df[filters].max(axis=1)
    df["mag_min"] = df[filters].min(axis=1)
    df['mag_range'] = df["mag_max"] - df["mag_min"]

    return df


def add_alpha_cyclic(df: pd.DataFrame):
    df = df.copy()

    radians = np.radians(df["alpha"])

    df["alpha_sin"] = np.sin(radians)
    df["alpha_cos"] = np.cos(radians)

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


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full report-aligned feature-engineering pipeline in order."""
    
    # Base astronomy features
    df = add_color_indices(df)
    df = add_redshift_transform(df)

    # Physics-driven interactions
    df = add_redshift_interaction(df)
    df = add_stellar_locus_distance(df)

    # Experimental color features
    df = add_qso_color_region(df)

    # Statistical features
    df = add_brightness_stat(df)

    # Positional features
    df = add_alpha_cyclic(df)

    return df

