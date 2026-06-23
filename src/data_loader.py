import pandas as pd


def load_data(df_path,include_class=True):
    df = pd.read_csv(df_path)
    df.drop('id',inplace=True,axis=1)

    cols = ['spectral_type','galaxy_population']
    
    dummy_data = pd.get_dummies(df[cols])
    dummy_data.drop('galaxy_population_Blue_Cloud',inplace=True,axis=1)

    df.drop(cols,inplace=True,axis=1)

    df = pd.concat([df,dummy_data],axis=1)

    if include_class:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df["class"] = le.fit_transform(df["class"])

    return df