import streamlit as st
# import streamlit.components.v1 as components
import numpy as np
import pandas as pd
from streamlit_folium import folium_static
import folium
import geopandas as gpd
from folium.plugins import HeatMap
from datetime import datetime
import time
import pickle
# from data_methods import get_casualties_dataset


# @st.cache
# def find_ward_name_loc_mapping():
#     accidents_data = get_casualties_dataset()
#     accidents_data.dropna(inplace=True)
#
#     fields = {
#         "longitude": "float64",
#         "latitude": "float64",
#         "ward_name": "category",
#     }
#     accidents_data = accidents_data[fields.keys()]
#     accidents_data = accidents_data.astype(fields)
#
#     ward_name_loc_mapping = (
#         accidents_data.groupby("ward_name")[["latitude", "longitude"]]
#         .aggregate("mean")
#         .reset_index()
#     )
#     print(ward_name_loc_mapping)
#     print(ward_name_loc_mapping.shape)
#
#     return ward_name_loc_mapping
#
#
# ward_locations = find_ward_name_loc_mapping()


open_wheather_id_cat_mapping = {
    "200": "8 Other",
    "201": "8 Other",
    "202": "8 Other",
    "210": "8 Other",
    "211": "8 Other",
    "212": "8 Other",
    "221": "8 Other",
    "230": "8 Other",
    "231": "8 Other",
    "232": "8 Other",
    "300": "2 Raining",
    "301": "2 Raining",
    "302": "2 Raining",
    "310": "2 Raining",
    "311": "2 Raining",
    "312": "2 Raining",
    "313": "2 Raining",
    "314": "2 Raining",
    "321": "2 Raining",
    "500": "2 Raining",
    "501": "2 Raining",
    "502": "2 Raining",
    "503": "2 Raining",
    "504": "2 Raining",
    "511": "2 Raining",
    "520": "2 Raining",
    "521": "2 Raining",
    "522": "2 Raining",
    "531": "2 Raining",
    "600": "3 Snowing",
    "601": "3 Snowing",
    "602": "3 Snowing",
    "611": "3 Snowing",
    "612": "3 Snowing",
    "613": "3 Snowing",
    "615": "3 Snowing",
    "616": "3 Snowing",
    "620": "3 Snowing",
    "621": "3 Snowing",
    "622": "3 Snowing",
    "701": "7 Fog/Mist",
    "711": "8 Other",
    "721": "8 Other",
    "731": "8 Other",
    "741": "7 Fog/Mist",
    "751": "8 Other",
    "761": "8 Other",
    "762": "8 Other",
    "771": "8 Other",
    "781": "8 Other",
    "800": "1 Fine",
    "801": "8 Other",
    "802": "8 Other",
    "803": "8 Other",
    "804": "8 Other",
}


# def get_current_whether():
#     for index, row in ward_locations.iterrows():
#         print(row["ward_name"])


def _max_width_():
    max_width_str = f"max-width: 10000px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )


def _max_height_():
    max_width_str = f"max-height: 10000px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )


def style_function(x):
    return {"fillColor": None, "color": "blue", "weight": 2, "fillOpacity": 0.0}


def main():
    """"""
    # _max_width_()
    # _max_height_()

    # ward_name_loc_mapping = find_ward_name_loc_mapping()

    st.title("Lists!")
    boros = gpd.read_file(
        "./dataset/statistical-gis-boundaries-london/ESRI/London_Ward.shp"
    )

    st.text(datetime.now())
    traffic_map = folium.Map(location=(51.536388, -0.140556), zoom_start=13)
    # folium.TileLayer('openstreetmap', opacity=0.42).add_to(traffic_map)

    boros = boros[boros.BOROUGH == "Camden"]
    fake_data = boros[["NAME"]]

    update_time = None
    fake_data["data"] = np.random.choice([1, 2, 3], fake_data.shape[0])

    datatyp = st.sidebar.radio("Data", ["live", "fake"])

    if datatyp == "fake":
        update_time = datetime.now()
        fake_data["data"] = np.random.choice([1, 2, 3], fake_data.shape[0])

    if st.sidebar.button("regen date"):
        update_time = datetime.now()
        fake_data["data"] = np.random.choice([1, 2, 3], fake_data.shape[0])

    if update_time:
        st.text(f"Last updated : {update_time}")
    folium.Choropleth(
        geo_data=boros,
        name="choropleth",
        data=fake_data,
        columns=["NAME", "data"],
        key_on="feature.properties.NAME",
        fill_color="BuPu",
        fill_opacity=0.7,
        line_opacity=1,
        bins=3,
        legend_name="Unemployment Rate (%)",
    ).add_to(traffic_map)

    folium.GeoJson(
        boros[boros.BOROUGH == "Camden"], style_function=style_function
    ).add_to(traffic_map)
    folium.LayerControl().add_to(traffic_map)

    folium_static(traffic_map, width=1000, height=700)


if __name__ == "__main__":
    # get_current_whether()
    main()
