import pickle
import pandas as pd
from datetime import datetime
from gds.demo.data.weather_api import get_open_weather_api_data
from gds.demo.data.data_methods import reduce_weather_categories


def evaluate_model(filename=".trained_models/balanced_data_random_tree.pkl"):

    # Load model
    cls = pickle.load(open(filename, "rb"))

    # Get live data
    input_vec = get_open_weather_api_data()
    input_vec["ward_name"] = input_vec.index
    input_vec = pd.get_dummies(input_vec)

    input_vec[f"hour_{datetime.now().hour}"] = 1.0
    input_vec[f"day_{datetime.now().strftime('%A')}"] = 1.0

    columns = []
    columns += [f"weather_{i}" for i in set(reduce_weather_categories.values())]
    columns += [f"hour_{i}" for i in range(24)]
    columns += [
        f"day_{i}"
        for i in [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
    ]

    df = pd.DataFrame(columns=columns)
    df = df.append(input_vec, ignore_index=True)
    df.index = input_vec.index
    df = df.fillna(0.0)

    output = cls.predict(df)
    output = pd.DataFrame(output, columns=["data"], index=df.index)

    return output
