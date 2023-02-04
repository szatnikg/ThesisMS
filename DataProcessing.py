import pandas as pd
df = pd.DataFrame({"x" : [40,234,452,1000 ],
                   "z": [31,41,411,330],
                   "I": [3103,34,1,3410]} )
feature_name = "x"

class Scaler:
    def __init__(self, features=[]):
        self.features = features
    def normalize(self, dict_df):
        df = pd.DataFrame(dict_df)
        result = df.copy()
        #original = df.copy()

        if not len(self.features):
            loop_container = df.columns
        else: loop_container = self.features

        self.feature_all_max, self.feature_all_min = {}, {}
        for feature_name in loop_container:
            feature_one_max, feature_one_min = [], []
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
            self.feature_all_min[feature_name] = min_value
            self.feature_all_max[feature_name] = max_value

        dict_to_return = {}
        for col in result.columns:
            dict_to_return[col] = result[col].values.tolist()

        return dict_to_return

    def denormalize(self, dict_df):
        df = pd.DataFrame(dict_df)
        original = df.copy()

        if not len(self.features):
            loop_container = df.columns
        else: loop_container = self.features

        for feature_name in loop_container:
            max_value = self.feature_all_max[feature_name]
            min_value = self.feature_all_min[feature_name]
            original[feature_name] = (original[feature_name]* (max_value - min_value) + min_value)

        dict_to_return = {}
        for col in original.columns:
            dict_to_return[col] = original[col].values.tolist()

        return dict_to_return

if __name__ == "__main__":
    scale = Scaler()
    a = scale.normalize(df)
    b = scale.denormalize(a)
    print("normalizing", a)
    print("denormalizing", b)

