import streamlit as st
import pandas as pd
from streamlit_folium import folium_static
import folium
import geopandas as gpd
from datetime import datetime
from gds.demo.model.evaluate import evaluate_model


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

    st.title("Proof of concept : Camden accident severity prediction")
    boros = gpd.read_file(
        "./dataset/statistical-gis-boundaries-london/ESRI/London_Ward.shp"
    )

    # update St. Pancras and Somers Town name
    boros.loc[
        boros.NAME == "St. Pancras and Somers Town", ["NAME"]
    ] = "St Pancras and Somers Town"
    boros = boros[boros.BOROUGH == "Camden"]

    st.text(f"Last updated : {datetime.now()}",)

    # Take ward names
    model_output = evaluate_model("./trained_models/balanced_data_random_tree.pkl")
    data = boros[["NAME"]]
    data = pd.merge(data, model_output, left_on="NAME", right_index=True)

    m = folium.Map(location=(51.536388, -0.140556), zoom_start=13)
    folium.Choropleth(
        geo_data=boros,
        name="choropleth",
        data=data,
        columns=["NAME", "data"],
        key_on="feature.properties.NAME",
        fill_color="BuPu",
        fill_opacity=0.7,
        line_opacity=1,
        bins=[0, 1, 2, 3],
        legend_name="Predicted result of accident",
    ).add_to(m)

    # folium.LayerControl().add_to(traffic_map)

    folium_static(m, width=1000, height=700)


if __name__ == "__main__":
    main()
