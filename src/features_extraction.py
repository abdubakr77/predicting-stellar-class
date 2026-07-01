import numpy as np
import pandas as pd


def add_color_indices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the 7 photometric colour indices used to cancel out distance and
    extinction effects (differences between magnitude bands).

    Args:
        df: dataframe with the raw u, g, r, i, z magnitude columns

    Returns:
        df with 7 new color_* columns
    """
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
    """
    Add a log1p transform of redshift.

    Args:
        df: dataframe with a redshift column

    Returns:
        df with log1p_redshift added
    """
    df = df.copy()
    # redshift is heavily right-skewed, log1p compresses the long tail
    df['log1p_redshift'] = np.log1p(df['redshift'])
    return df


def add_redshift_interaction(df: pd.DataFrame):
    """
    Add interaction terms between redshift and the main color indices.

    Args:
        df: dataframe that already has color_ug and color_gr (run
            add_color_indices first)

    Returns:
        df with redshift_inter_ug, redshift_inter_gr, redshift_x_z added
    """
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
    """
    Flag objects that fall in the typical quasar color-color region.

    Args:
        df: dataframe that already has color_ug and color_gr

    Returns:
        df with a binary quasar_color_region column
    """
    df = df.copy()

    required_cols = ["color_ug", "color_gr"]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f'{missing} Not Found! Please Make sure that you added all the 7 colour indices from add_color_indices function!')

    # thresholds based on the known quasar locus in u-g vs g-r space
    df['quasar_color_region'] = ((df['color_ug'] < 0.6) & (df['color_gr'] > -0.15) & (df['color_gr'] < 0.7)).astype(int)

    return df


def add_brightness_stat(df: pd.DataFrame):
    """
    Add summary statistics across the 5 magnitude bands.

    Args:
        df: dataframe with u, g, r, i, z columns

    Returns:
        df with mag_mean, mag_max, mag_min, mag_range added
    """
    df = df.copy()

    filters = ["u", "g", "r", "i", "z"]

    df["mag_mean"] = df[filters].mean(axis=1)
    df["mag_max"] = df[filters].max(axis=1)
    df["mag_min"] = df[filters].min(axis=1)
    df['mag_range'] = df["mag_max"] - df["mag_min"]

    return df


def add_alpha_cyclic(df: pd.DataFrame):
    """
    Encode alpha and delta as sin/cos pairs so the model sees them as
    cyclic angles instead of plain numbers (0 and 360 degrees are the
    same point on the sky, a raw value can't express that).

    Args:
        df: dataframe with alpha and delta columns

    Returns:
        df with alpha_sin, alpha_cos, delta_sin, delta_cos added
    """
    df = df.copy()

    alpha_radians = np.radians(df["alpha"])
    delta_radians = np.radians(df['delta'])

    df["alpha_sin"] = np.sin(alpha_radians)
    df["alpha_cos"] = np.cos(alpha_radians)
    df['delta_sin'] = np.sin(delta_radians)
    df['delta_cos'] = np.cos(delta_radians)

    return df


def add_stellar_locus_distance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Measure how far each object sits from the empirical stellar locus
    and from the typical quasar locus in (g-r, r-i) color space. A large
    distance from the stellar locus usually flags a galaxy or quasar.

    Args:
        df: dataframe, color_gr and color_ri will be added automatically
            if missing

    Returns:
        df with stellar_locus_dist and qso_locus_dist added
    """
    df = df.copy()
    if "color_ri" not in df.columns or "color_gr" not in df.columns:
        df = add_color_indices(df)

    # (0.52, 0.25) is roughly the center of the main-sequence stellar locus
    df['stellar_locus_dist'] = np.sqrt((df['color_gr'] - 0.52)**2 + (df['color_ri'] - 0.25)**2)

    # (0.24, 0.15) is roughly the center of the quasar locus
    df['qso_locus_dist'] = np.sqrt((df['color_gr'] - 0.24)**2 + (df['color_ri'] - 0.15)**2)

    return df


def add_spectral_type_formula(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct the spectral_type bucket from g-r color, using the same
    thresholding the competition data was generated with. Needed when
    appending the original SDSS17 dataset, which doesn't have this column.

    Args:
        df: dataframe with r and g columns

    Returns:
        df with a spectral_type column (M, G/K, A/F, or O/B)
    """
    df = df.copy()
    df['spectral_type'] = pd.cut(
        df['r'] - df['g'],
        [-np.inf, -1, -0.5, 0, np.inf],
        labels=['M', 'G/K', 'A/F', 'O/B']
    ).astype(str)
    return df


def add_galaxy_population_formula(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct the galaxy_population bucket from u-r color, same idea
    as add_spectral_type_formula above.

    Args:
        df: dataframe with u and r columns

    Returns:
        df with a galaxy_population column (Blue_Cloud or Red_Sequence)
    """
    df = df.copy()
    df['galaxy_population'] = pd.cut(
        df['u'] - df['r'],
        [-np.inf, 2.2, np.inf],
        labels=['Blue_Cloud', 'Red_Sequence']
    ).astype(str)
    return df


def add_locus_curvature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add second-order shape features of the stellar locus by multiplying
    adjacent color indices together.

    Args:
        df: dataframe with the 7 color_* columns (run add_color_indices
            first)

    Returns:
        df with ug_gr, gr_ri, ri_iz added
    """
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
    """
    Add a squared redshift term plus binary flags for low/mid/high
    redshift zones (roughly star / overlap / quasar territory).

    Args:
        df: dataframe with a redshift column

    Returns:
        df with redshift_sq, redshift_low, redshift_mid, redshift_high added
    """
    df = df.copy()
    df['redshift_sq']   = df['redshift'] ** 2
    df['redshift_low']  = (df['redshift'] < 0.1).astype(int)
    df['redshift_mid']  = ((df['redshift'] >= 0.1) &
                           (df['redshift'] < 0.8)).astype(int)
    df['redshift_high'] = (df['redshift'] >= 0.8).astype(int)
    return df


def add_sky_coords_3d(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert alpha/delta into 3D unit-sphere coordinates. This avoids the
    wraparound problem at alpha=0/360 and gives the model true spherical
    distances instead of raw angles.

    Args:
        df: dataframe with alpha and delta columns

    Returns:
        df with sky_x, sky_y, sky_z added
    """
    df = df.copy()
    alpha_rad = np.radians(df['alpha'])
    delta_rad = np.radians(df['delta'])
    df['sky_x'] = np.cos(delta_rad) * np.cos(alpha_rad)
    df['sky_y'] = np.cos(delta_rad) * np.sin(alpha_rad)
    df['sky_z'] = np.sin(delta_rad)
    return df


def add_redshift_color_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add redshift x color interaction terms. These matter most in the
    redshift overlap zone where color alone can't separate the classes.

    Args:
        df: dataframe that already has color_ug and color_gr

    Returns:
        df with z_x_u_g and z_x_g_r added
    """
    df = df.copy()
    required = ["color_ug", "color_gr"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'{missing} not found! Run add_color_indices first.')

    df['z_x_u_g'] = df['redshift'] * df['color_ug']
    df['z_x_g_r'] = df['redshift'] * df['color_gr']
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline in order.

    Args:
        df: raw dataframe with the original photometric columns

    Returns:
        df with every engineered feature added
    """
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