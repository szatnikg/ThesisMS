import pandas as pd
df = pd.DataFrame({"x" : [40,234,452,1000 ],
                   "y": [31,41,411,3300],
                   "preds": [3103,34,1,3410]} )

df_nopred = pd.DataFrame({"x" : [40,234,452,1000 ],
                   "y": [31,41,411,3300],
                   } )

class Scaler:
    def __init__(self, features=[]):
        self.features = features

    def normalize(self, dict_df, label_feature_name="y", prediction_feature_name="preds" ):
        df = pd.DataFrame(dict_df)
        result = df.copy()
        #original = df.copy()
        self.prediction_feature_name = prediction_feature_name
        if not len(self.features):
            loop_container = df.columns
        else: loop_container = self.features

        self.feature_all_max, self.feature_all_min = {}, {}
        for feature_name in loop_container:
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
            # if feature_name == label: add an additional one for preds.
            if feature_name == label_feature_name:
                self.feature_all_min[self.prediction_feature_name] = min_value
                self.feature_all_max[self.prediction_feature_name] = max_value
            self.feature_all_min[feature_name] = min_value
            self.feature_all_max[feature_name] = max_value

        dict_to_return = {}
        for col in result.columns:
            dict_to_return[col] = result[col].values.tolist()

        return dict_to_return

    def denormalize(self, dict_df, is_preds_normalized=True):
        df = pd.DataFrame(dict_df)
        original = df.copy()

        if not len(self.features):
            loop_container = df.columns
        else: loop_container = self.features
        if is_preds_normalized:
            loop_container = [self.prediction_feature_name]
        for feature_name in loop_container:
            # check if feature was normalized in the first place
            if feature_name in self.feature_all_max:
                max_value = self.feature_all_max[feature_name]
                min_value = self.feature_all_min[feature_name]
                original[feature_name] = (original[feature_name] * (max_value - min_value) + min_value)

        dict_to_return = {}
        for col in original.columns:
            dict_to_return[col] = original[col].values.tolist()
        print(dict_to_return)
        return dict_to_return

if __name__ == "__main__":
    scale = Scaler()
    a = scale.normalize(df_nopred)
    a["preds"] = [0.9099442651804048, 0.009680258140217073, 0.0, 1.1]
    b = scale.denormalize(a)
    print("normalizing", a)
    print("denormalizing", b)

