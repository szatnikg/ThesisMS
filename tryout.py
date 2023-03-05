import pandas as pd

# # define dataset for multivariate timeseriesgeneration
# # dataset = hstack((series, series2))
# # print(dataset)
# # n_features = dataset.shape[1]

# Note:
# resetting and dropping old indexes can be done by:
# my_df.reset_index(inplace=True, drop=True)



if __name__ == "__main__":

    data = pd.read_excel(
        "C:\Egyetem\Diplomamunka\data\TanulokAdatSajat.xlsx")
    # data = GenerateData.genUnNormalizedData(1,300,type="square",step=400)
    # own_pred_data = pandas.read_excel("C:\Egyetem\Diplomamunka\data\TanulokAdatSajat_ownpred.xlsx")

    pred_x = [] # own_pred_data.iloc[::, :-2]
    pred_y = [] #own_pred_data.iloc[::,-2]  #pandas.DataFrame({"y": [j**2 for j in range(460, 560)]})

    # This applies to only OwnPred usage.
    # shuffle order is right for the showTrainTest method, but using pandas.concat,
    # the indexes are applied, so x_test,y_test must be resetted.
    # output = pandas.concat([NN.x_test.reset_index(inplace=False, drop=False),
    #                         NN.y_test.reset_index(inplace=False, drop=False), NN.preds], axis= 1)
    #
    # output.to_csv("result.csv")
