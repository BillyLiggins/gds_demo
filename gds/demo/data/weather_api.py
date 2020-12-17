from sodapy import Socrata
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
from gds.demo.utils import singleton
import random
from datetime import datetime
import os

import requests
from gds.demo.data.data_methods import get_casualties_dataset


known_cats = {
    "1 Fine": "1 Fine",  # 800
    "2 Raining": "2 Raining",  # 5** 3**
    "3 Snowing": "3 Snowing",  # 6**
    "4 Fine/High Winds": "1 Fine",  # 741
    "5 Raining/High Winds": "2 Raining",  # 2**
    "6 Snowing/High Winds": "3 Snowing",
    "7 Fog/Mist": "7 Fog/Mist",  # 701
    "8 Other": "8 Other",  # & 7**
    "9 Unknown": "9 Unknown",
}

open_wheather_id_cat_mapping = {
    200: "8 Other",
    201: "8 Other",
    202: "8 Other",
    210: "8 Other",
    211: "8 Other",
    212: "8 Other",
    221: "8 Other",
    230: "8 Other",
    231: "8 Other",
    232: "8 Other",
    300: "2 Raining",
    301: "2 Raining",
    302: "2 Raining",
    310: "2 Raining",
    311: "2 Raining",
    312: "2 Raining",
    313: "2 Raining",
    314: "2 Raining",
    321: "2 Raining",
    500: "2 Raining",
    501: "2 Raining",
    502: "2 Raining",
    503: "2 Raining",
    504: "2 Raining",
    511: "2 Raining",
    520: "2 Raining",
    521: "2 Raining",
    522: "2 Raining",
    531: "2 Raining",
    600: "3 Snowing",
    601: "3 Snowing",
    602: "3 Snowing",
    611: "3 Snowing",
    612: "3 Snowing",
    613: "3 Snowing",
    615: "3 Snowing",
    616: "3 Snowing",
    620: "3 Snowing",
    621: "3 Snowing",
    622: "3 Snowing",
    701: "7 Fog/Mist",
    711: "8 Other",
    721: "8 Other",
    731: "8 Other",
    741: "7 Fog/Mist",
    751: "8 Other",
    761: "8 Other",
    762: "8 Other",
    771: "8 Other",
    781: "8 Other",
    800: "1 Fine",
    801: "8 Other",
    802: "8 Other",
    803: "8 Other",
    804: "8 Other",
}


@singleton
def find_ward_name_loc_mapping():
    accidents_data = get_casualties_dataset()
    accidents_data.dropna(inplace=True)

    fields = {
        "longitude": "float64",
        "latitude": "float64",
        "ward_name": "category",
    }
    accidents_data = accidents_data[fields.keys()]
    accidents_data = accidents_data.astype(fields)

    ward_name_loc_mapping = (
        accidents_data.groupby("ward_name")[["latitude", "longitude"]]
        .aggregate("mean")
        .reset_index()
    )
    ward_name_loc_mapping = ward_name_loc_mapping.set_index("ward_name")
    return ward_name_loc_mapping


def get_open_weather_api_data():
    # Need to get the locs of each ward
    ward_name_loc_mapping = find_ward_name_loc_mapping()

    # call the api for each loc
    output = {}
    for ward_name, row in ward_name_loc_mapping.iterrows():
        url = f"http://api.openweathermap.org/data/2.5/weather"
        params = {
            "lon": row["longitude"],
            "lat": row["latitude"],
            "appid": os.environ.get("OPEN_WEATHER_API_KEY"),
        }
        resp = requests.get(url=url, params=params)
        if resp.status_code == 200:
            resp = resp.json()
            output[ward_name] = open_wheather_id_cat_mapping.get(resp["weather"][0]["id"])

    output = pd.DataFrame.from_dict(output, columns=["weather"], orient='index')
    return output
