import math, random
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

def genUnNormalizedData(data_size, type="linear"):
    import DataProcessing
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
    import DataProcessing
    scale = DataProcessing.Scaler(features=[])
    szamok = genUnNormalizedData(700)
    normaltSzamok = scale.normalize(szamok)

    linearPlot(normaltSzamok)
    linearPlot(scale.denormalize(normaltSzamok,is_preds_normalized=False))