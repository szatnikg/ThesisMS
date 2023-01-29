import math, random
import pandas as pd
from matplotlib import pyplot as plt

def getDataNormalized(data_size, type="linear"):
    # creating data
    my_data = {"x": [j/data_size for j in range(1, data_size)]
               } #
    data_y= []
    for k in range(1, data_size):
        number = k / data_size
        addition = random.choice([0.02])
        if type =="linear":
            data_y.append(number* 1.3 + addition)
        elif type =="square":
            data_y.append((number ** (2)) + addition)
        elif type == "root":

            data_y.append(math.sqrt(number) * 3.2 + addition)
        else:
            data_y.append((math.sin(number)) * 1 + addition)
    my_data["y"] = data_y
    return my_data

def getDataUnnormalized(data_size, type="linear"):
    # creating data
    my_data = {"x": [j for j in range(1, data_size)]
               } #
    data_y= []
    for k in range(1, data_size):
        number = k
        addition = random.choice([0.02,0.05,0.09,0.04])
        if type =="linear":
            data_y.append(number* 1.3 + addition)
        elif type =="square":
            data_y.append((number ** (2)) + addition)
        elif type == "root":

            data_y.append(math.sqrt(number) * 3.2 + addition)
        else:
            data_y.append((math.sin(number)) * 1 + addition)
    my_data["y"] = data_y
    return my_data


def getComplexData(data_size):
    my_data = {}
    x_data = [j for j in range(1, data_size)]
    y_data = []

    e = math.e
    fraction = 463
    even = False
    for y in range(1, data_size):
        additional = random.randint(1, 300)
        some_var = random.randint(1, 3)
        if even:
            complex_y = (((additional + 1000) / fraction - (y * some_var)) * e ** 4)
            even = False
        elif not even:
            complex_y = (((additional + 1000) / fraction + (y * some_var)) * e ** 4)
            even = True

        y_data.append(complex_y)

    my_data["x"] = x_data
    my_data["y"] = y_data
    return my_data

def linearPlot(my_data):
    df = pd.DataFrame(my_data)
    plt.plot(my_data["x"], my_data["y"])
    plt.show()
    #print(df.head(20))

if __name__ == "__main__":
    szamok = getData(650,type="root")
    linearPlot(szamok)