import random
import numpy as np
from pandas import DataFrame as df
from matplotlib import pyplot as plt
dict = {"x":[1,2,3,4,5,6]}
dict_2 = {"z":[1,2,3,4,5,6],
          "k":[10,20,30,40,50,60]}
my_df = df(dict)
my_df2 = df(dict_2)

shuffler = np.random.permutation(len(my_df))

my_df = my_df.iloc[shuffler]
my_df2 = my_df2.iloc[shuffler]
#print(max(my_df2["z"]))

# my_df.reset_index(inplace=True, drop=True)
# print(my_df.columns.tolist())
# print(my_df2)
#
# plt.scatter(my_df2, my_df["x"])
# plt.legend(my_df2.columns)
# plt.show()
