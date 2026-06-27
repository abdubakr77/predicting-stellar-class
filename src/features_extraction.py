import numpy as np
import pandas as pd


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

    alpha_radians = np.radians(df["alpha"])
    delta_radians = np.radians(df['delta'])

    df["alpha_sin"] = np.sin(alpha_radians)
    df["alpha_cos"] = np.cos(alpha_radians)
    df['delta_sin'] = np.sin(delta_radians)
    df['delta_cos'] = np.cos(delta_radians)

    return df


def add_stellar_locus_distance(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    if "color_ri" not in df.columns or "color_gr" not in df.columns:
        df = add_color_indices(df)

    df['stellar_locus_dist'] = np.sqrt((df['g_r'] - 0.52)**2 + (df['r_i'] - 0.25)**2)

    df['qso_locus_dist'] = np.sqrt((df['g_r'] - 0.24)**2 + (df['r_i'] - 0.15)**2)

    return df

def add_spectral_type_formula(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['spectral_type'] = pd.cut(
        df['r'] - df['g'],
        [-np.inf, -1, -0.5, 0, np.inf],
        labels=['M', 'G/K', 'A/F', 'O/B']
    ).astype(str)
    return df

def add_galaxy_population_formula(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['galaxy_population'] = pd.cut(
        df['u'] - df['r'],
        [-np.inf, 2.2, np.inf],
        labels=['Blue_Cloud', 'Red_Sequence']
    ).astype(str)
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

