import streamlit as st
import numpy as np
import pandas as pd
from streamlit_folium import folium_static
import folium
import geopandas as gpd
from folium.plugins import HeatMap
from datetime import datetime
import time
import pickle
from gds.demo.data.data_methods import find_ward_name_loc_mapping
from gds.demo.model.evaluate import evaluate_model


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
    fake_data_2 = boros[["NAME"]]
    output = evaluate_model("./trained_models/balanced_data_random_tree.pkl")
    # output[
    print(output)
    fake_data = pd.merge(fake_data, output, left_on="NAME", right_index=True)
    fake_data.data += 1
    fake_data["data"] = fake_data.apply(lambda x: x["data"] + 1, axis=1)
    print("fake_data")
    print(fake_data)

    update_time = None
    # fake_data_2["data"] = np.random.choice([1, 2, 3], fake_data_2.shape[0])
    # print("fake_data_2")
    # print(fake_data_2)

    # datatyp = st.sidebar.radio("Data", ["live", "fake"])

    # if datatyp == "fake":
    #     update_time = datetime.now()
    #     fake_data["data"] = np.random.choice([1, 2, 3], fake_data.shape[0])
    #
    # if st.sidebar.button("regen date"):
    #     update_time = datetime.now()
    #     fake_data["data"] = np.random.choice([1, 2, 3], fake_data.shape[0])
    #
    # if update_time:
    #     st.text(f"Last updated : {update_time}")

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
    #


if __name__ == "__main__":
    # get_current_whether()
    main()
