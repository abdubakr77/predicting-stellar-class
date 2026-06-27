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


def add_locus_curvature(df: pd.DataFrame) -> pd.DataFrame:
    """Second-order shape of the stellar locus"""
    df = df.copy()
    required = ["color_ug", "color_gr", "color_ri", "color_iz"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'{missing} not found! Run add_color_indices first.')
    
    df['ug_gr'] = df['color_ug'] * df['color_gr']
    df['gr_ri'] = df['color_gr'] * df['color_ri']
    df['ri_iz'] = df['color_ri'] * df['color_iz']
    return df


def add_redshift_features(df: pd.DataFrame) -> pd.DataFrame:
    """redshift² and zone thresholds"""
    df = df.copy()
    df['redshift_sq']   = df['redshift'] ** 2
    df['redshift_low']  = (df['redshift'] < 0.1).astype(int)
    df['redshift_mid']  = ((df['redshift'] >= 0.1) & 
                           (df['redshift'] < 0.8)).astype(int)
    df['redshift_high'] = (df['redshift'] >= 0.8).astype(int)
    return df


def add_sky_coords_3d(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    alpha_rad = np.radians(df['alpha'])
    delta_rad = np.radians(df['delta'])
    df['sky_x'] = np.cos(delta_rad) * np.cos(alpha_rad)
    df['sky_y'] = np.cos(delta_rad) * np.sin(alpha_rad)
    df['sky_z'] = np.sin(delta_rad)
    return df


def add_redshift_color_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required = ["color_ug", "color_gr"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'{missing} not found! Run add_color_indices first.')
    
    df['z_x_u_g'] = df['redshift'] * df['color_ug']
    df['z_x_g_r'] = df['redshift'] * df['color_gr']
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full report-aligned feature-engineering pipeline in order."""
    
    df = add_color_indices(df)
    df = add_redshift_transform(df)
    df = add_redshift_interaction(df)
    df = add_stellar_locus_distance(df)
    df = add_qso_color_region(df)
    df = add_brightness_stat(df)
    df = add_alpha_cyclic(df)
    df = add_locus_curvature(df)
    df = add_redshift_features(df)
    df = add_sky_coords_3d(df)
    df = add_redshift_color_interactions(df)

    return df

