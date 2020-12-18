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
    gen_fake_possion_data,
)


plt.style.use("./mplstyles/custom_style.mplstyle")


def poisson(k, Norm, lamb):
    return Norm * (lamb ** k / factorial(k)) * np.exp(-lamb)


def make_basic_lamp_plots(df_lamp_posts):
    fig, ax = plt.subplots(figsize=(10, 10))
    n_lamp_types = df_lamp_posts.lamp_type_number.unique()
    ax.hist(df_lamp_posts.lamp_type_number, range=(0, n_lamp_types), bins=n_lamp_types)

    # ax.bar(df_lamp_posts.lamp_type, df_lamp_posts.lamp_type_number)  # range=(0, n_lamp_types), bins=n_lamp_types)
    # g = sns.barplot(x="lamp_type", y="lamp_type_number", data=df_lamp_posts, ax=ax)
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig("plots/lamp_type.png")

    fig, ax = plt.subplots()
    ax.hist(df_lamp_posts.wattage, range=(0, 250), bins=50)
    fig.tight_layout()
    fig.savefig("plots/lamp_wattage.png")

    fig, ax = plt.subplots(figsize=(10, 8))
    counts, xedges, yedges, im = ax.hist2d(
        df_lamp_posts.lamp_type_number,
        df_lamp_posts.wattage,
        range=((0, n_lamp_types), (0, 250)),
        bins=(n_lamp_types, 50),
    )
    fig.tight_layout()
    fig.colorbar(im, ax=ax)
    fig.savefig("plots/lamp_type_wattage_corr.png")


def get_processed_lighting_data():
    df_lamp_posts = get_lighting_dataset()
    fields = {
        "longitude": "float64",
        "latitude": "float64",
        "wattage": "float64",
        "lamp_type": "string",
        "ward_name": "string",
        "local_authority_asset_number": "int",
    }

    df_lamp_posts = (
        df_lamp_posts[fields.keys()]
        .dropna()
        .astype(fields)
        .drop_duplicates()
        .set_index("local_authority_asset_number")
    )
    lamp_type_number_mapping = {
        k: v for v, k in enumerate(df_lamp_posts.lamp_type.unique())
    }
    lamp_type_number_mapping_inv = {v: k for k, v in lamp_type_number_mapping.items()}
    df_lamp_posts["lamp_type_number"] = df_lamp_posts.lamp_type.map(
        lamp_type_number_mapping
    )
    return df_lamp_posts


def get_processed_casualties_data():
    df_casualties = get_casualties_dataset()
    fields = {
        "reference": "string",
        "casualty_severity": "string",
        "day": "string",
        "date": "string",
        "time": "string",
        "longitude": "float64",
        "latitude": "float64",
        "ward_name": "string",
    }

    df_casualties = (
        df_casualties[fields.keys()]
        .dropna()
        .astype(fields)
        .drop_duplicates()
        # .set_index("reference")
    )

    casualty_severity_mapping = {
        k: v for v, k in enumerate(df_casualties.casualty_severity.unique())
    }
    casualty_severity_mapping_inv = {v: k for k, v in casualty_severity_mapping.items()}
    df_casualties["casualty_severity_number"] = df_casualties.casualty_severity.map(
        casualty_severity_mapping
    )
    df_casualties["datetime"] = pd.to_datetime(
        df_casualties.date, format="%Y-%m-%dT%H:%M:%S"
    )
    df_casualties["hour"] = pd.DatetimeIndex(df_casualties.datetime).hour
    df_casualties["night"] = (df_casualties.hour > 20) | (df_casualties.hour < 6)
    return df_casualties


def select_highest_severity_rows(df_casualties):
    """ For entries with the same reference only take highest severity entry in the dataset"""

    # This is taking the highest casualty_severity entry in multiple entries
    holder = []
    for g, gg in df_casualties.groupby("reference"):
        if gg.shape[0] > 1:
            holder.append(
                [
                    g,
                    gg.casualty_severity_number[
                        gg.casualty_severity_number != gg.casualty_severity_number.max()
                    ].values.tolist(),
                ]
            )

    for ref, to_be_dropped in holder:
        index = df_casualties[
            (df_casualties.reference == ref)
            & (df_casualties.casualty_severity_number.isin(to_be_dropped))
        ].index
        df_casualties = df_casualties.drop(index=index)

    return df_casualties


def plot_accident_loc_with_lighting(accident_loc, selected_lamps, df_lamp_posts):
    m = folium.Map(location=accident_loc, zoom_start=25)
    for index, row in df_lamp_posts[["latitude", "longitude"]].iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=1,
            color="#3186cc",
            fill=True,
            fill_color="#3186cc",
        ).add_to(m)

    for index, row in selected_lamps[["latitude", "longitude"]].iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=5,
            color="#e00404",
            fill=True,
            fill_color="#e00404",
        ).add_to(m)

    folium.CircleMarker(
        location=accident_loc,
        radius=10,
        color="#3186cc",
        fill=True,
        fill_color="#3186cc",
    ).add_to(m)
    m.save(
        f"plots/accident_lamps/lamp_check_{accident_loc[0]:.3f}_{accident_loc[0]:.3f}.html"
    )


def plot_collected_lighting_info(df, distance_cut_off):
    fig, ax = plt.subplots()
    ax.hist(df["n_lamps"], bins=20)
    ax.set_xlabel(f"Number of lamps with cut of {distance_cut_off} m")
    plt.tight_layout()
    fig.savefig(f"plots/accident_lamps/n_lamps_{distance_cut_off}.png")

    fig, ax = plt.subplots()
    ax.hist(
        df["lamp_type_number_mode"],
        bins=len(df["lamp_type_number_mode"].unique()),
        range=[0, len(df["lamp_type_number_mode"].unique())],
    )
    plt.tight_layout()
    fig.savefig(f"plots/accident_lamps/lamp_type_mode_{distance_cut_off}.png")


def lamp_posts(regenerate_data_cached_to_disk: bool = False):

    distance_cut_off = 50  # meters

    df_lamp_posts = get_processed_lighting_data()
    # Explore the lamp dataset
    make_basic_lamp_plots(df_lamp_posts)

    df_casualties = get_processed_casualties_data()

    # reduce dataset to show only events in the dark
    df_casualties = df_casualties[df_casualties.night]

    df_casualties = select_highest_severity_rows(df_casualties=df_casualties)
    df_casualties = df_casualties.set_index("reference")

    # -------- Now analysis --------

    # We now want to see if lighting around a casualty site has a correlation
    # with the casualty severity of the accident.

    # For each accident calculating the distance from accident site and each
    # lamp post is computational intensive.  Can we grid the data such that we
    # can use the grid to improve the look up efficiency?

    n_lat_bins = 10
    n_lon_bins = 10
    min_lat = min(df_casualties.latitude.min(), df_lamp_posts.latitude.min())
    max_lat = max(df_casualties.latitude.max(), df_lamp_posts.latitude.max())
    min_lon = min(df_casualties.longitude.min(), df_lamp_posts.longitude.min())
    max_lon = max(df_casualties.longitude.max(), df_lamp_posts.longitude.max())

    # only do the lamp grid mapping if cached results don't exist or if
    # requested from the command line
    if regenerate_data_cached_to_disk or not os.path.isfile(
        "output/lamp_grid_lookup.pkl"
    ):
        LampGrid = LampGridLookup(
            n_lat_bins, min_lat, max_lat, n_lon_bins, min_lon, max_lon,
        )
        for index, row in tqdm(
            df_lamp_posts.iterrows(),
            total=df_lamp_posts.shape[0],
            desc="Generating lamp grid mapping",
        ):

            LampGrid.get_collection_at_x_y(
                x=row["latitude"], y=row["longitude"]
            ).add_lamp_id(lamp_id=index)
        pickle.dump(LampGrid, open("output/lamp_grid_lookup.pkl", "wb"))
    else:
        LampGrid = pickle.load(open("output/lamp_grid_lookup.pkl", "rb"))

    # Now for each accident calculate information about the lighting in the
    # surrounding area and cache to disk. Only do the computation if cached
    # results don't exist or if requested from the command line
    if regenerate_data_cached_to_disk or not os.path.isfile(
        "output/lighting_features.pkl"
    ):

        results = pd.DataFrame(
            columns=["lamp_type_number_mode", "wattage_mean", "wattage_mode"]
        )

        for i, (index, row) in enumerate(
            tqdm(
                df_casualties[["latitude", "longitude"]].iterrows(),
                total=df_casualties.shape[0],
                desc="accident looping",
            )
        ):
            accident_loc = (row["latitude"], row["longitude"])
            # Get the lamp ids found in the same "bin" as the accident location
            lamp_ids = LampGrid.get_collection_at_x_y(
                x=row["latitude"], y=row["longitude"]
            ).collection_to_list()

            def do_distance_calc(row: pd.Series, accident_loc: tuple):
                light_loc = (row["latitude"], row["longitude"])
                return distance.distance(light_loc, accident_loc).m

            selected_lamps = df_lamp_posts[(df_lamp_posts.index.isin(lamp_ids))].copy()
            if selected_lamps.index.any():
                selected_lamps["distance_from_accident"] = selected_lamps.apply(
                    lambda x: do_distance_calc(row=x, accident_loc=accident_loc), axis=1
                )
                selected_lamps = selected_lamps[
                    selected_lamps.distance_from_accident < distance_cut_off
                ]

                if selected_lamps.index.any():
                    # Make plot for sanity checking takes time so commented by
                    # default
                    # if i % 100 == 0:
                    #     plot_accident_loc_with_lighting(
                    #         accident_loc, selected_lamps, df_lamp_posts
                    #     )

                    # Now we can report information about the lamps that are close to the accident
                    out_dict = {
                        "casualty_index": index,
                        "lamp_type_number_mode": int(
                            selected_lamps.lamp_type_number.mode()[0]
                        ),
                        "wattage_mean": selected_lamps.wattage.mean(),
                        "wattage_mode": selected_lamps.wattage.mode()[0],
                        "n_lamps": selected_lamps.shape[0],
                    }
                else:
                    out_dict = {
                        "casualty_index": index,
                        "lamp_type_number_mode": pd.NA,
                        "wattage_mean": pd.NA,
                        "wattage_mode": pd.NA,
                    }
            else:
                out_dict = {
                    "casualty_index": index,
                    "lamp_type_number_mode": pd.NA,
                    "wattage_mean": pd.NA,
                    "wattage_mode": pd.NA,
                }

            results = results.append(out_dict, ignore_index=True)
        results.to_pickle("output/lighting_features.pkl")
    else:
        results = pd.read_pickle("output/lighting_features.pkl")

    # Now need to join with df_casualties
    df = pd.merge(df_casualties, results, left_index=True, right_on="casualty_index")
    df.dropna(inplace=True)

    # Need to cast to int
    df["lamp_type_number_mode"] = df["lamp_type_number_mode"].astype(int)

    print("correlation between lighting and casualty severity:")
    print(df.corr())
    print("correlation between lamp_type and casualty severity:")
    print(df[["casualty_severity_number", "lamp_type_number_mode"]].corr())

    plot_collected_lighting_info(df, distance_cut_off)

    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Is lighting correlated to accident severity ."
    )
    parser.add_argument(
        "-r",
        dest="regen",
        type=bool,
        nargs="?",
        const=True,
        default=False,
        help="Regenerate data cached to disk",
    )

    args = parser.parse_args()

    os.makedirs("plots/", exist_ok=True)
    os.makedirs("plots/accident_lamps/", exist_ok=True)
    os.makedirs("dataset/", exist_ok=True)
    os.makedirs("output/", exist_ok=True)
    lamp_posts(regenerate_data_cached_to_disk=args.regen)
