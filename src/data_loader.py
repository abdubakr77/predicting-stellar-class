import pandas as pd


def load_data(df_path, include_class=True, is_org_dataset=False):
    """
    Load a raw CSV and prep it for the pipeline.

    Args:
        df_path: path to the CSV file
        include_class: whether to label-encode the target column
        is_org_dataset: True if this is the original SDSS17 dataset,
            which has different metadata columns than the competition data

    Returns:
        pd.DataFrame ready for feature engineering
    """
    df = pd.read_csv(df_path)

    if is_org_dataset:
        # the original dataset carries SDSS metadata columns we don't use
        df.drop(['obj_ID', 'run_ID', 'rerun_ID', 'cam_col', 'field_ID',
                  'spec_obj_ID', 'plate', 'fiber_ID', 'MJD'], inplace=True, axis=1)
    else:
        df.drop('id', inplace=True, axis=1)

        cols = ['spectral_type', 'galaxy_population']

        dummy_data = pd.get_dummies(df[cols], dtype=int)
        # Blue_Cloud is redundant once Red_Sequence is known (they're complementary)
        dummy_data.drop('galaxy_population_Blue_Cloud', inplace=True, axis=1)

        df.drop(cols, inplace=True, axis=1)

        df = pd.concat([df, dummy_data], axis=1)

    if include_class:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df["class"] = le.fit_transform(df["class"])
        # printed so we always know which integer maps to which class
        print(f'Label Inverse Is {le.classes_}')

    return df