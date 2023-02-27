import pandas as pd

class Scaler:
    def __init__(self, features=[]):
        # if features is an empty array, (pandas df also compatible)
        # than all features going to be normalized
        self.features = features
        # this boolean insures to only apply the normalization/denormalization
        # of the instance and do not begin with a fresh start when either method is called.
        # used: when applying to OwnPred_x the same x_train feature-scales.
        self.already_constructed = False

    def normalize(self, dict_df, label_feature_name="y", prediction_feature_name="preds", to_dict=False):

        df = pd.DataFrame(dict_df)
        result = df.copy()
        self.prediction_feature_name = prediction_feature_name
        self.to_dict = to_dict
        if not len(self.features):
            loop_container = df.columns
        else: loop_container = self.features
        if not self.already_constructed:
            self.feature_all_max, self.feature_all_min = {}, {}

        # if already constructed, apply the given feautre_all_max/min for normalization

        for feature_name in loop_container:
            if self.already_constructed:
                max_value = self.feature_all_max[feature_name]
                min_value = self.feature_all_min[feature_name]
            else:
                max_value = df[feature_name].max()
                min_value = df[feature_name].min()

            # Todo give other methods as well, for handling outlying data
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
            # if feature_name == label: create an additional one for preds.
            if feature_name == label_feature_name:
                self.feature_all_min[self.prediction_feature_name] = min_value
                self.feature_all_max[self.prediction_feature_name] = max_value
            self.feature_all_min[feature_name] = min_value
            self.feature_all_max[feature_name] = max_value
        self.already_constructed = True
        if self.to_dict:
            dict_to_return = {}
            for col in result.columns:
                dict_to_return[col] = result[col].values.tolist()
            return dict_to_return
        else:
            return result

    def denormalize(self, dict_df, is_preds_normalized=True):
        df = pd.DataFrame(dict_df)
        original = df.copy()
        # print("denormalizing dataframe: ", original)
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
        if self.to_dict:
            dict_to_return = {}
            for col in original.columns:
                dict_to_return[col] = original[col].values.tolist()
            return dict_to_return
        else:
            return original

if __name__ == "__main__":
    df = pd.DataFrame({"x": [40, 234, 452, 1000],
                       "y": [31, 41, 411, 3300],
                       "preds": [3103, 34, 1, 3410]})

    df_nopred = pd.DataFrame({"x": [40, 234, 452, 1000],
                              "y": [31, 41, 411, 3300],
                              })
    # print(df.iloc[:,-1])
    # print(df.columns[:-1])


    scale = Scaler(df_nopred.columns[::])
    a = scale.normalize(df_nopred)
    a["preds"] = [0.9099442651804048, 0.009680258140217073, 0.0, 1.1]
    b = scale.denormalize(a, is_preds_normalized=False)
    print("normalizing", a)
    print("denormalizing", b)

    print(df.iloc[:2, 1])
    from matplotlib import pyplot as plt
    plt.scatter(df.iloc[:2, 1], df.iloc[:2, 2])
    plt.show()