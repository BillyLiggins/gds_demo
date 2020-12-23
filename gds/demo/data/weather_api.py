import pandas as pd
from gds.demo.utils import singleton
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
    # accidents_data = get_casualties_dataset()
    # accidents_data.dropna(inplace=True)
    #
    # fields = {
    #     "longitude": "float64",
    #     "latitude": "float64",
    #     "ward_name": "category",
    # }
    # print(accidents_data.keys())  # ["ward_name"].uquine())
    # accidents_data = accidents_data[fields.keys()]
    # accidents_data = accidents_data.astype(fields)
    #
    # ward_name_loc_mapping = (
    #     accidents_data.groupby("ward_name")[["latitude", "longitude"]]
    #     .aggregate("mean")
    #     .reset_index()
    # )
    # ward_name_loc_mapping = ward_name_loc_mapping.set_index("ward_name")

    # Data api changed, this hard coding allows the app to run
    ward_name_loc_mapping = {
        "Belsize": {"latitude": 51.550187, "longitude": -0.166274},
        "Bloomsbury": {"latitude": 51.521896, "longitude": -0.12796},
        "Camden Town with Primrose Hill": {
            "latitude": 51.541257,
            "longitude": -0.153317,
        },
        "Cantelowes": {"latitude": 51.546972, "longitude": -0.133216},
        "Fortune Green": {"latitude": 51.555394, "longitude": -0.196607},
        "Frognal and Fitzjohns": {"latitude": 51.553596, "longitude": -0.181925},
        "Gospel Oak": {"latitude": 51.555472, "longitude": -0.149841},
        "Hampstead Town": {"latitude": 51.555672, "longitude": -0.176203},
        "Haverstock": {"latitude": 51.547857, "longitude": -0.160506},
        "Highgate": {"latitude": 51.571704, "longitude": -0.150124},
        "Holborn and Covent Garden": {"latitude": 51.513031, "longitude": -0.124177},
        "Kentish Town": {"latitude": 51.549095, "longitude": -0.144084},
        "Kilburn": {"latitude": 51.540911, "longitude": -0.196743},
        "King's Cross": {"latitude": 51.532001, "longitude": -0.123323},
        "Regent's Park": {"latitude": 51.531271, "longitude": -0.156969},
        "St Pancras and Somers Town": {"latitude": 51.53109, "longitude": -0.129785},
        "Swiss Cottage": {"latitude": 51.543031, "longitude": -0.175708},
        "West Hampstead": {"latitude": 51.547216, "longitude": -0.191102},
    }
    ward_name_loc_mapping = pd.DataFrame.from_dict(
        ward_name_loc_mapping, orient="index"
    )

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
            output[ward_name] = open_wheather_id_cat_mapping.get(
                resp["weather"][0]["id"]
            )

    output = pd.DataFrame.from_dict(output, columns=["weather"], orient="index")
    return output
