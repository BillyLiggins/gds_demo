import os
from scipy.special import factorial
from scipy.optimize import curve_fit
import geopandas as gpd
from folium.plugins import HeatMap
import folium
from pprint import pprint
import numpy as np
import pandas as pd
from descartes import PolygonPatch
import matplotlib.pyplot as plt
import seaborn as sns
from geopy import distance
import pickle
from tqdm import tqdm
from shapely.geometry import shape

from gds.demo.lighting_utils.lamp_grid_lookup import LampGridLookup
from gds.demo.data.data_methods import (
    get_ward_dataset,
    get_lighting_dataset,
    get_casualties_dataset,
)


plt.style.use("./mplstyles/custom_style.mplstyle")


def style_function(x):
    return {"fillColor": None, "color": "black", "weight": 1.5, "fillOpacity": 0.0}


def accident_locations():
    """
    """

    df = get_casualties_dataset()
    fields = {
        "reference": "string",
        "casualty_age_band": "string",
        "casualty_age": "int32",
        "number_of_casualties": "int32",
        "casualty_severity": "string",
        "mode_of_travel": "string",
        "number_of_vehicles": "int32",
        "day": "string",
        "speed_limit": "string",
        "pedestrian_crossing": "string",
        "light_condition_band": "string",
        "weather": "string",
        "road_surface": "string",
        "special_conditions": "string",
        "carriage_way_hazard": "string",
        "longitude": "float64",
        "latitude": "float64",
        "ward_code": "string",
        "ward_name": "string",
    }
    df = df[fields.keys()]
    df = df.dropna()
    df = df.astype(fields)

    # heat_df = df[(df.casualty_severity == "1 Fatal")]
    heat_df = df
    heat_df = heat_df[["latitude", "longitude"]]
    heat_df = heat_df.dropna(axis=0, subset=["latitude", "longitude"])
    heat_data = [
        [row["latitude"], row["longitude"]] for index, row in heat_df.iterrows()
    ]
    boros = gpd.read_file(
        "dataset/statistical-gis-boundaries-london/ESRI/London_Ward.shp"
    )

    # wards = get_ward_dataset()
    # geom = [shape(i) for i in wards.location]
    # wards_geom = gpd.GeoDataFrame({"geometry": geom})

    m = folium.Map(location=(51.536388, -0.140556), zoom_start=13)
    for datum in heat_data:
        folium.CircleMarker(
            location=datum, radius=1, color="#3186cc", fill=True, fill_color="#3186cc"
        ).add_to(m)

    HeatMap(heat_data, radius=15, blur=5,).add_to(m)
    folium.GeoJson(
        boros[boros.BOROUGH == "Camden"], style_function=style_function
    ).add_to(m)
    m.save("plots/all_accidents_in_camden_map.html")
    return


if __name__ == "__main__":
    os.makedirs("plots/", exist_ok=True)
    accident_locations()
