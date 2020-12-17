import pandas as pd
from datetime import datetime

# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.ensemble import RandomForestClassifier

from gds.demo.data.weather_api import get_open_weather_api_data
from gds.demo.data.data_methods import reduce_weather_categories
import pickle


# Load model

# Get live data

# eval model

# pass results back

# df = pd.DataFrame(columns=["A", "B", "C", ])
#
# df = df.append({"A": 3}, ignore_index=True)
# df = df.fillna(0.0)
# print(df)


def evaluate_model(filename=".trained_models/balanced_data_random_tree.pkl"):

    # Load model
    cls = pickle.load(open(filename, 'rb'))

    # Get live data
    input_vec = get_open_weather_api_data()
    input_vec["ward_name"] = input_vec.index
    print(input_vec)
    print(len(input_vec))
    input_vec = pd.get_dummies(input_vec)  # = get_open_weather_api_data()
    print(input_vec)
    print(len(input_vec))
    print(input_vec.keys())

    input_vec[f"hour_{datetime.now().hour}"] = 1.0
    input_vec[f"day_{datetime.now().strftime('%A')}"] = 1.0
    print(input_vec)
    columns = []
    columns += [f"weather_{i}" for i in set(reduce_weather_categories.values())]
    columns += [f"hour_{i}" for i in range(24)]
    columns += [f"day_{i}" for i in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]]

    df = pd.DataFrame(columns=columns)
    df = df.append(input_vec, ignore_index=True)
    df.index = input_vec.index
    df = df.fillna(0.0)
    print(df)
    print(df.keys())
    output = cls.predict(df)
    print(output)
    return pd.DataFrame(output, columns=["data"], index=df.index)
    # eval model
