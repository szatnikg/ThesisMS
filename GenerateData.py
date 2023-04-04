import math, random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def genNormalizedData(data_size, type="linear"):
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
        elif type =="polinomial":
            data_y.append((number**2) + (2 * number**3) + (number**4) + addition)
        else:
            data_y.append((math.sin(number)) * 1 + addition)
    my_data["y"] = data_y
    return pd.DataFrame(my_data)

def genUnNormalizedData(data_size_from, data_size_to, type="linear", step=200):
    import DataProcessing
    # creating data
    my_data = {"x": [j for j in range(data_size_from, data_size_to,step)]
               } #
    data_y= []
    for k in range(data_size_from, data_size_to,step):
        number = k
        addition = random.choice([0.2,0.04,0.09,0.44])
        if type =="linear":
            data_y.append(number* 1.3 + addition)
        elif type =="square":
            data_y.append((number ** (2)) +addition)
        elif type == "root":
            data_y.append(math.sqrt(number) * 3.2 + addition)
        elif type =="polinomial":
            data_y.append((number**2) + (2 * number**3) + (number**4) + addition)
        else:
            data_y.append((math.sin(number)) * 1 + addition)
    # normalizing it with min-max scaling
    my_data["y"] = data_y
    return pd.DataFrame(my_data)

def genSinwawe(cycles, resolution, start = 0.0):

    length = np.pi * 2 * cycles
    x_val = np.arange(start, length, length / resolution)
    counter = 0
    x_season = []
    period = 1
    for i in x_val:
        # print("i", i)
        # print("res", resolution / i)
        if i+ (1/(resolution/cycles)) >= np.pi * 2 * period:
            counter = 0
            period += 1

        x_season.append(counter)
        counter += 1
    x_season = np.array(x_season)
    my_wave = np.sin(np.arange(0, length, length / resolution))
    return pd.concat([
                      pd.DataFrame(x_val, columns= ["x"]),
                      (pd.DataFrame(x_season, columns=["season"])),
                      (pd.DataFrame(my_wave, columns= ["y"]))
                      ]
                     , axis = 1)


def genComplexData(data_size):
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
    return pd.DataFrame(my_data)

def linearPlot(my_data):
    df = pd.DataFrame(my_data)
    plt.plot(df["x"], df["y"])
    plt.show()


if __name__ == "__main__":
    df = genComplexData(1000)
    print(df)
    linearPlot(df)

    # import DataProcessing
    # scale = DataProcessing.Scaler(features=[])
    # szamok = genUnNormalizedData(-1, 1, type="else")
    # szamok = genSinwawe(0.5, 100)
    #
    # normaltSzamok = scale.normalize(szamok)
    #
    # linearPlot(normaltSzamok)
    # linearPlot(scale.denormalize(normaltSzamok,is_preds_normalized=False))