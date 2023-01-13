import datetime
import time
import numpy
import pandas as pd
df = pd.DataFrame({"my_data" : [40,234,452,1000 ] } )
feature_name = "my_data"

def normalize(dict_df, feature_name):
    df = pd.DataFrame(dict_df)
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result[feature_name].values.tolist()
print(normalize(df,"my_data"))


