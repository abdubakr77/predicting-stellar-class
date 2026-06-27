import pandas as pd
from features_extraction import add_spectral_type_formula,add_galaxy_population_formula,build_features

def load_data(df_path,include_class=True,is_org_dataset=False):
    df = pd.read_csv(df_path)
    df = build_features(df)

    if is_org_dataset:
        df.drop(['obj_ID','run_ID','rerun_ID','cam_col','field_ID','spec_obj_ID','plate','fiber_ID','MJD'],inplace=True,axis=1)
        df = add_spectral_type_formula(df)
        df = add_galaxy_population_formula(df)
        
    else:
        df.drop('id',inplace=True,axis=1)

    cols = ['spectral_type','galaxy_population']
    
    dummy_data = pd.get_dummies(df[cols],dtype=int)
    dummy_data.drop('galaxy_population_Blue_Cloud',inplace=True,axis=1)

    df.drop(cols,inplace=True,axis=1)

    df = pd.concat([df,dummy_data],axis=1)

    if include_class:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df["class"] = le.fit_transform(df["class"])

    return df