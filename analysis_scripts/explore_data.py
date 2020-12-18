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


DAY_MAPPING = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6,
}
DAY_MAPPING_INV = {v: k for k, v in DAY_MAPPING.items()}


def poisson(k, Norm, lamb):
    return Norm * (lamb ** k / factorial(k)) * np.exp(-lamb)


def make_plotting_dirs():
    os.makedirs("plots/", exist_ok=True)
    os.makedirs("plots/by_day/", exist_ok=True)
    os.makedirs("plots/by_hour/", exist_ok=True)
    os.makedirs("plots/day_year/", exist_ok=True)


def make_basic_lamp_plots(df_lamp_posts):
    fig, ax = plt.subplots(figsize=(10, 10))
    n_lamp_types = df_lamp_posts.lamp_type_number.unique()
    ax.hist(df_lamp_posts.lamp_type_number, range=(0, n_lamp_types), bins=n_lamp_types)
    # ax.bar(df_lamp_posts.lamp_type, df_lamp_posts.lamp_type_number)  # range=(0, n_lamp_types), bins=n_lamp_types)
    # g = sns.barplot(x="lamp_type", y="lamp_type_number", data=df_lamp_posts, ax=ax)
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yscale("log")
    fig.tight_layout()
    make_plotting_dirs()
    fig.savefig("plots/lamp_type.png")

    fig, ax = plt.subplots()
    ax.hist(df_lamp_posts.wattage, range=(0, 250), bins=50)
    fig.tight_layout()
    make_plotting_dirs()
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
    make_plotting_dirs()
    fig.savefig("plots/lamp_type_wattage_corr.png")


def lamp_posts(regenerate_data_cached_to_disk: bool = False):

    distance_cut_off = 20
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

    # Explore the lamp dataset
    # plot_lamp_sites_on_map(df_lamp_posts)
    make_basic_lamp_plots(df_lamp_posts)

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

    # reduce dataset to show only events in the dark
    df_casualties = df_casualties[df_casualties.night]

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

    df_casualties = df_casualties.set_index("reference")

    # -------- Now analysis --------

    # We now want to see if lighting around a casualty site has a correlation
    # with the number of incidents or the casualty severity of the accident.

    # Can we grid the data such that we can use the grid to improve the look up
    # efficiency.

    n_lat_bins = 10
    n_lon_bins = 10
    min_lat = min(df_casualties.latitude.min(), df_lamp_posts.latitude.min())
    max_lat = max(df_casualties.latitude.max(), df_lamp_posts.latitude.max())
    min_lon = min(df_casualties.longitude.min(), df_lamp_posts.longitude.min())
    max_lon = max(df_casualties.longitude.max(), df_lamp_posts.longitude.max())

    os.makedirs("output/", exist_ok=True)
    if regenerate_data_cached_to_disk or not os.path.isfile(
        "output/lamp_grid_lookup.pkl"
    ):
        LampGrid = LampGridLookup(
            n_lat_bins, min_lat, max_lat, n_lon_bins, min_lon, max_lon,
        )
        for index, row in tqdm(df_lamp_posts.iterrows(),
                               total=df_lamp_posts.shape[0],
                               desc="Generating lamp grid mapping",
                               ):

            LampGrid.get_collection_at_x_y(
                x=row["latitude"], y=row["longitude"]
            ).add_lamp_id(lamp_id=index)
        pickle.dump(LampGrid, open("output/lamp_grid_lookup.pkl", "wb"))
    else:
        LampGrid = pickle.load(open("output/lamp_grid_lookup.pkl", "rb"))

    os.makedirs("output/", exist_ok=True)
    if regenerate_data_cached_to_disk or not os.path.isfile(
        "output/lighting_features.pkl"
    ):

        results = pd.DataFrame(
            columns=["lamp_type_number_mode", "wattage_mean", "wattage_mode"]
        )

        for index, row in tqdm(
            df_casualties[["latitude", "longitude"]].iterrows(),
            total=df_casualties.shape[0],
            desc="accident looping",
        ):
            accident_loc = (row["latitude"], row["longitude"])
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
                    # Make plot for sanity checking
                    # if i == 10:
                    #     traffic_map = folium.Map(location=accident_loc, zoom_start=13)
                    #     print(selected_lamps)
                    #
                    #     for index, row in df_lamp_posts[
                    #         ["latitude", "longitude"]
                    #     ].iterrows():
                    #         folium.CircleMarker(
                    #             location=[row["latitude"], row["longitude"]],
                    #             radius=1,
                    #             color="#3186cc",
                    #             fill=True,
                    #             fill_color="#3186cc",
                    #         ).add_to(traffic_map)
                    #     for index, row in selected_lamps[
                    #         ["latitude", "longitude"]
                    #     ].iterrows():
                    #         folium.CircleMarker(
                    #             location=[row["latitude"], row["longitude"]],
                    #             radius=5,
                    #             color="#e00404",
                    #             fill=True,
                    #             fill_color="#e00404",
                    #         ).add_to(traffic_map)
                    #     folium.CircleMarker(
                    #         location=accident_loc,
                    #         radius=10,
                    #         color="#3186cc",
                    #         fill=True,
                    #         fill_color="#3186cc",
                    #     ).add_to(traffic_map)
                    #     traffic_map.save("lamp_check.html")

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

    df["lamp_type_number_mode"] = df["lamp_type_number_mode"].astype(int)

    print(df.corr())
    print(df[["casualty_severity_number", "lamp_type_number_mode"]].corr())

    fig, ax = plt.subplots()
    ax.hist(df["n_lamps"], bins=20)
    plt.show()
    fig, ax = plt.subplots()
    ax.hist(
        df["lamp_type_number_mode"],
        bins=len(df["lamp_type_number_mode"].unique()),
        range=[0, len(df["lamp_type_number_mode"].unique())],
    )
    # ax.set_xticklabels("")
    #
    # ax.set_xticklabels(
    #     # labels=[
    #     #     lamp_type_number_mapping_inv[ind] if ind in lamp_type_number_mapping_inv else "" for ind in
    #     #     range(len(df['lamp_type_number_mode'].unique()))
    #     # ],
    #     labels=[str(i) for i in
    #             np.arange(len(df['lamp_type_number_mode'].unique()))
    #             ],
    #     rotation=40,
    #     ha="right",
    #     # minor=True,
    # )
    plt.tight_layout()
    plt.show()

    return


def get_clean_casualties_dataset(fake_date: bool = False):

    if fake_date:
        df = gen_fake_possion_data()
    else:
        df = get_casualties_dataset()

    fields = {
        "reference": "string",
        # "casualty_severity": "string",
        "date": "string",
        # "time": "string",
        # "longitude": "float64",
        # "latitude": "float64",
        # "ward_code": "string",
        # "ward_name": "string",
    }
    df = df[fields.keys()]
    print(f"before na drop : {df.shape}")
    df = df.dropna()
    print(f"after na drop : {df.shape}")
    df = df.astype(fields)
    # df = df.set_index("reference")
    print("_" * 80)
    print(df)
    print(f"before duplicated drop : {df.shape}")
    df = df.drop_duplicates()
    print(f"after duplicated drop : {df.shape}")
    return df


def possion_in_time():

    df = get_clean_casualties_dataset()

    df["datetime"] = pd.to_datetime(df.date, format="%Y-%m-%dT%H:%M:%S")
    df["date"] = pd.DatetimeIndex(df.datetime).date
    df["day"] = pd.DatetimeIndex(df.datetime).strftime("%A")
    df["year"] = pd.DatetimeIndex(df.datetime).year.astype(int)
    df["hour"] = pd.DatetimeIndex(df.datetime).hour
    df["month"] = pd.DatetimeIndex(df.date).month
    df["month_number"] = df.apply(lambda x: f"{x.month}{x.year}", axis=1)
    month_number_mapping = {k: v for v, k in enumerate(df.month_number.unique())}
    month_number_mapping_inv = {v: k for k, v in month_number_mapping.items()}
    df["month_number"] = df.month_number.map(month_number_mapping)

    df["day_number"] = df.day.map(DAY_MAPPING)

    date_number_mapping = {k: v for v, k in enumerate(sorted(df.date.unique()))}
    date_number_mapping_inv = {v: k for k, v in date_number_mapping.items()}
    df["date_number"] = df.date.map(date_number_mapping)

    holder = []
    for g, gg in df.groupby("reference"):
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
        index = df[
            (df.reference == ref) & (df.casualty_severity_number.isin(to_be_dropped))
        ].index
        df = df.drop(index=index)

    df = df.set_index("reference")

    # events per month
    dff = (
        df.groupby(["month_number", "date_number"])["date_number"]
        .count()
        .reset_index(name="counts")
    )
    dfff = [
        [month, dff[dff.month_number == month].counts.mean()]
        for month in dff.month_number.unique()
    ]
    dfff = pd.DataFrame.from_records(dfff, columns=["month_number", "mean_counts"])
    fig, ax = plt.subplots(nrows=2)
    ax[0].plot(dfff.month_number, dfff.mean_counts)
    ax[1].hist(dfff.mean_counts)
    plt.tight_layout()
    make_plotting_dirs()
    fig.savefig("plots/mean_accidents_per_month.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    counts, xedges, yedges, im = ax.hist2d(
        df.day_number, df.hour, bins=(7, 24), range=((0, 7), (0, 24))
    )

    ax.set_xlabel("Day")
    ax.set_ylabel("Hour")
    print(xedges)
    ax.set_xticks(xedges, minor=True)
    ax.set_xticklabels("")

    ax.set_xticklabels(
        labels=[
            DAY_MAPPING_INV[ind] if ind in DAY_MAPPING_INV else "" for ind in xedges
        ],
        rotation=40,
        ha="right",
        minor=True,
    )
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    make_plotting_dirs()
    fig.savefig("plots/accidents_per_day_hour.png")

    fig, ax = plt.subplots()
    counts, xedges, yedges, im = ax.hist2d(
        df.hour, df.day_number + 0.5, bins=(24, 7), range=((0, 24), (0, 7))
    )
    ax.set_xlabel("Hour")
    ax.set_ylabel("Day")
    plt.tight_layout()
    make_plotting_dirs()
    fig.savefig("plots/accidents_per_day_hour_transpose.png")
    # plt.show()

    ax.cla()

    # for g, gg in df.groupby(["day_number", "hour"]):
    #     print(g)
    #     print(gg[["day_number", "hour", "date_number"]])
    #     print(gg["date_number"].count())
    #     return

    # df_by_day = (
    #     df.groupby(["day_number", "date_number"])["date_number"]
    #     .count()
    #     .reset_index(name="counts")
    # )
    # min_bin, max_bin = (1, 15)
    # entries, bin_edges = np.histogram(
    #     df_by_day[(df_by_day.day_number != 5) & (df_by_day.day_number != 6)].counts,
    #     # df_by_day.counts,
    #     bins=max_bin - min_bin,
    #     range=[min_bin, max_bin],
    # )
    # min_bin, max_bin = (1, 15)
    # entries, bin_edges = np.histogram(
    #     df_by_day[(df_by_day.day_number != 5) & (df_by_day.day_number != 6)].counts,
    #     # df_by_day.counts,
    #     bins=max_bin - min_bin,
    #     range=[min_bin, max_bin],
    # )

    slice_by_day(df)
    slice_by_hour(df)
    slice_by_day_and_year(df)
    return

    df_by_hour = (
        df.groupby(["hour", "date_number"])["date_number"]
        .count()
        .reset_index(name="counts")
    )
    df_by_hour_year = (
        df.groupby(["hour", "year", "date_number"])["date_number"]
        .count()
        .reset_index(name="counts")
    )

    df_by_day_year = (
        df.groupby(["day_number", "year", "date_number"])["date_number"]
        .count()
        .reset_index(name="counts")
    )

    # for hour_number in range(24):
    #     # for year in range(2005, 2016):
    #     min_bin, max_bin = (1, 7)
    #     entries, bin_edges = np.histogram(
    #         df_by_hour[df_by_hour.hour == hour_number].counts,
    #         bins=max_bin - min_bin,
    #         range=[min_bin, max_bin],
    #     )
    #
    #     print(df.date[df.date_number == df_by_hour.date_number.max()].unique())
    #     print(df.date[df.date_number == df_by_hour.date_number.min()].unique())
    #     number_of_days_in_data_set = (
    #         df.date[df.date_number == df_by_hour.date_number.max()].unique()[0]
    #         - df.date[df.date_number == df_by_hour.date_number.min()].unique()[0]
    #     ).days
    #     print("number_of_days_in_data_set 1: ", number_of_days_in_data_set)
    #
    #     number_of_days_in_data_set_ = (df.date.max() - df.date.min()).days
    #     print("number_of_days_in_data_set 2: ", number_of_days_in_data_set_)
    #
    #     print("number_of_days_in_data_set : ", number_of_days_in_data_set)
    #     print(
    #         "df_by_hour[df_by_hour.hour == hour_number].date_number.unique().shape[0] : ",
    #         df_by_hour[df_by_hour.hour == hour_number].date_number.unique().shape[0],
    #     )
    #     number_of_days_with_zero_accidents = (
    #         number_of_days_in_data_set
    #         - df_by_hour[df_by_hour.hour == hour_number].date_number.unique().shape[0]
    #     )
    #     entries = np.array([number_of_days_with_zero_accidents] + entries.tolist())
    #     bin_edges = np.array([0] + bin_edges.tolist())
    #
    #     # entries, bin_edges = np.histogram(
    #     #     df_by_hour_year[(df_by_hour_year.year == year) & (df_by_hour_year.hour == hour_number)].counts,
    #     #     bins=max_bin - min_bin,
    #     #     range=[min_bin, max_bin],
    #     # )
    #     #
    #     # number_of_days_in_data_set = (df.date.max() - df.date.min()).days
    #     # print("number_of_days_in_data_set : ", number_of_days_in_data_set)
    #     # print(
    #     #     "df_by_hour[df_by_hour.hour == hour_number].date_number.unique().shape[0] : ",
    #     #     df_by_hour[df_by_hour.hour == hour_number].date_number.unique().shape[0],
    #     # )
    #     # number_of_days_with_zero_accidents = (
    #     #     number_of_days_in_data_set
    #     #     - df_by_hour[df_by_hour.hour == hour_number].date_number.unique().shape[0]
    #     # )
    #     # entries = np.array([number_of_days_with_zero_accidents] + entries.tolist())
    #     # bin_edges = np.array([0] + bin_edges.tolist())
    #
    #     parameters, cov_matrix = curve_fit(
    #         poisson,
    #         bin_edges[:-1],
    #         entries,
    #         p0=[200, 3],
    #         bounds=[[1, 0.0001], [100000, 6]],
    #     )
    #     print("hist:")
    #     print(list(zip(bin_edges[:-1], entries)))
    #     print("parameters:")
    #     print(parameters)
    #     print("cov_matrix:")
    #     print(cov_matrix)
    #
    #     # min_bin, max_bin = (1, 15)
    #     # entries_all, bin_edges_all = np.histogram(
    #     #     df.groupby(["day_number", "date_number"])["date_number"].count().values,
    #     #     bins=max_bin - min_bin,
    #     #     range=[min_bin, max_bin],
    #     # )
    #     # print(list(zip(bin_edges_all[:-1], entries_all)))
    #     # entries_all = np.array([number_of_days_with_zero_accidents] + entries_all.tolist())
    #     # bin_edges_all = np.array([0] + bin_edges_all.tolist())
    #     ax.errorbar(
    #         bin_edges[:-1], entries, fmt="r.", label=r"Data", yerr=np.sqrt(entries),
    #     )
    #     x = np.linspace(0, bin_edges[-1], 100)
    #     ax.plot(x, poisson(x, *parameters), "r-", lw=2, label=r"Poisson fit")
    #     ax.set_ylim(0.009, 10000)
    #     ax.set_yscale("log")
    #     # fig.savefig(f"plots/{hour_number}_{year}.png")
    #     fig.savefig(f"plots/{hour_number}.png")
    #     ax.cla()
    return


def calc_chisq(x, parameters, function, data):
    Nexp = function(x, *parameters)
    r = data - Nexp
    stdev = np.std(data)
    chisq = np.sum((r / stdev) ** 2)
    ndf = len([entry for entry in data if entry]) - 1
    print("chisq =", chisq, "ndf =", ndf)
    return chisq, ndf


def slice_by_hour(df):
    min_bin, max_bin = (1, 15)
    bin_range = (1, 15)
    n_bins = max_bin - min_bin
    # heat_map_date = pd.DataFrame(columns=["year", "day", "chisq", "lambda"])

    for indices, group in df.groupby(["hour"]):
        print(indices)
        hour = indices
        print(group)
        counts = group.groupby(["date_number"])["date_number"].count()
        entries, bin_edges = np.histogram(counts, bins=n_bins, range=bin_range,)

        number_of_days_in_data_set = (group.date.max() - group.date.min()).days
        number_of_days_with_zero_accidents = (
            number_of_days_in_data_set - group.date_number.unique().shape[0]
        )

        entries = np.array([number_of_days_with_zero_accidents] + entries.tolist())
        bin_edges = np.array([0] + bin_edges.tolist())
        parameters, cov_matrix = curve_fit(
            poisson, bin_edges[:-1], entries, bounds=[[1, 0.0001], [100000, 10]],
        )
        print("hist:")
        print(list(zip(bin_edges[:-1], entries)))
        print("parameters:")
        print(parameters)
        print("cov_matrix:")
        print(cov_matrix)

        chisq, ndf = calc_chisq(
            x=bin_edges[:-1], parameters=parameters, function=poisson, data=entries
        )
        fig, ax = plt.subplots()
        ax.errorbar(
            bin_edges[:-1], entries, fmt="r.", label=r"Data", yerr=np.sqrt(entries),
        )
        x = np.linspace(0, bin_edges[-1], 100)
        ax.plot(x, poisson(x, *parameters), "r-", lw=2, label=r"Poisson fit")
        result_string = (
            f"Result :\n"
            f"$\lambda$ : {parameters[1]:.2} $\pm$ {np.sqrt(cov_matrix[1][1]):.2}\n"
            f"$\chi^2/ndf$ : {chisq:.2}/{ndf}\n"
        )
        ax.text(7, 20, result_string)
        ax.set_ylim(0.9, 5000)
        # ax.set_yscale("log")
        fig.tight_layout()
        make_plotting_dirs()
        fig.savefig(f"plots/by_hour/{hour}.png")
        ax.cla()
        plt.clf()
    return


def slice_by_day(df):
    min_bin, max_bin = (1, 15)
    bin_range = (1, 15)
    n_bins = max_bin - min_bin
    heat_map_date = pd.DataFrame(columns=["year", "day", "chisq", "lambda"])

    for indices, group in df.groupby(["day_number"]):
        day = indices
        counts = group.groupby(["date_number"])["date_number"].count()
        entries, bin_edges = np.histogram(counts, bins=n_bins, range=bin_range,)

        number_of_days_in_data_set = (group.date.max() - group.date.min()).days
        number_of_days_in_data_set = number_of_days_in_data_set / 7.0
        number_of_days_with_zero_accidents = (
            number_of_days_in_data_set - group.date_number.unique().shape[0]
        )

        entries = np.array([number_of_days_with_zero_accidents] + entries.tolist())
        bin_edges = np.array([0] + bin_edges.tolist())
        parameters, cov_matrix = curve_fit(
            poisson, bin_edges[:-1], entries, bounds=[[1, 0.0001], [100000, 10]],
        )
        print("hist:")
        print(list(zip(bin_edges[:-1], entries)))
        print("parameters:")
        print(parameters)
        print("cov_matrix:")
        print(cov_matrix)

        chisq, ndf = calc_chisq(
            x=bin_edges[:-1], parameters=parameters, function=poisson, data=entries
        )
        fig, ax = plt.subplots()
        ax.errorbar(
            bin_edges[:-1], entries, fmt="r.", label=r"Data", yerr=np.sqrt(entries),
        )
        x = np.linspace(0, bin_edges[-1], 100)
        ax.plot(x, poisson(x, *parameters), "r-", lw=2, label=r"Poisson fit")
        result_string = (
            f"Result :\n"
            f"$\lambda$ : {parameters[1]:.2} $\pm$ {np.sqrt(cov_matrix[1][1]):.2}\n"
            f"$\chi^2/ndf$ : {chisq:.2}/{ndf}\n"
        )
        ax.text(7, 200, result_string)
        ax.set_ylim(0.9, 300)
        # ax.set_yscale("log")
        make_plotting_dirs()
        fig.savefig(f"plots/by_day/{day}.png")
        ax.cla()
        plt.clf()
    return


def slice_by_day_and_year(df):

    min_bin, max_bin = (1, 15)
    bin_range = (1, 15)
    n_bins = max_bin - min_bin
    heat_map_date = pd.DataFrame(columns=["year", "day", "chisq", "lambda"])

    for indices, group in df.groupby(["day_number", "year"]):
        print(indices)
        day = indices[0]
        year = indices[1]
        counts = group.groupby(["date_number"])["date_number"].count()
        entries, bin_edges = np.histogram(counts, bins=n_bins, range=bin_range,)

        number_of_days_in_data_set = np.busday_count(
            f"{year}", f"{year+1}", weekmask=[1 if i == day else 0 for i in range(7)],
        )
        number_of_days_with_zero_accidents = (
            number_of_days_in_data_set - group.date_number.unique().shape[0]
        )

        entries = np.array([number_of_days_with_zero_accidents] + entries.tolist())
        bin_edges = np.array([0] + bin_edges.tolist())
        parameters, cov_matrix = curve_fit(
            poisson,
            bin_edges[:-1],
            entries,
            # p0=[200, 3],
            bounds=[[1, 0.0001], [100000, 10]],
        )
        print("hist:")
        print(list(zip(bin_edges[:-1], entries)))
        print("parameters:")
        print(parameters)
        print("cov_matrix:")
        print(cov_matrix)

        chisq, ndf = calc_chisq(
            x=bin_edges[:-1], parameters=parameters, function=poisson, data=entries
        )
        fig, ax = plt.subplots()
        ax.errorbar(
            bin_edges[:-1], entries, fmt="r.", label=r"Data", yerr=np.sqrt(entries),
        )
        x = np.linspace(0, bin_edges[-1], 100)
        ax.plot(x, poisson(x, *parameters), "r-", lw=2, label=r"Poisson fit")
        result_string = (
            f"Result :\n"
            f"$\lambda$ : {parameters[1]:.2} $\pm$ {np.sqrt(cov_matrix[1][1]):.2}\n"
            f"$\chi^2/ndf$ : {chisq:.2}/{ndf}\n"
        )
        ax.text(7, 20, result_string)
        ax.set_ylim(0, 32)
        # ax.set_yscale("log")
        make_plotting_dirs()
        fig.savefig(f"plots/day_year/{year}_{day}.png")
        ax.cla()
        plt.clf()
        heat_map_date = heat_map_date.append(
            {"year": year, "day": day, "chisq": chisq, "lambda": parameters[1]},
            ignore_index=True,
        )
    plotting_grid = (
        heat_map_date[["year", "day", "lambda"]].set_index(["year", "day"]).unstack()
    )
    plotting_grid.columns = [f"{DAY_MAPPING_INV[j]}" for i, j in plotting_grid.columns]
    plotting_grid.reset_index()
    plotting_grid.index = plotting_grid.index.astype(int)

    fig, ax = plt.subplots(figsize=(20, 20))
    g = sns.heatmap(plotting_grid, annot=True, ax=ax, fmt=".1f")
    g.set_yticklabels(g.get_yticklabels(), rotation=0)
    g.set_xticklabels(g.get_xticklabels(), rotation=45)
    # ax.set_title("Harvest of local farmers (in tons/year)")
    fig.tight_layout()
    make_plotting_dirs()
    fig.savefig("plots/time_heat_map.png")
    ax.cla()

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=plotting_grid, ax=ax)

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    make_plotting_dirs()
    fig.savefig("plots/time_series.png")
    return
    # #     return
    # #     year = indices[0]
    # #     day = indices[0]
    # #     entries, bin_edges = np.histogram(
    # #         group.counts,
    # #         bins=n_bins,
    # #         range=bin_range,
    # #     )
    # #
    # #     # number_of_days_in_data_set = 52  # Fix this
    # #     number_of_days_in_data_set = np.busday_count(
    # #         f"{year}",
    # #         f"{year+1}",
    # #         weekmask=[
    # #             1 if i == day else 0
    # #             for i in range(len(df_by_day_year.day_number.unique()))
    # #         ],
    # #     )
    # #     number_of_days_with_zero_accidents = (
    # #         number_of_days_in_data_set
    # #         - df_by_day_year[
    # #             (df_by_day_year.day_number == day) & (df_by_day_year.year == year)
    # #         ]
    # #         .date_number.unique()
    # #         .shape[0]
    # #     )
    # #
    # #     entries = np.array([number_of_days_with_zero_accidents] + entries.tolist())
    # #     bin_edges = np.array([0] + bin_edges.tolist())
    # #
    # #     # entries, bin_edges = np.histogram(
    # #     #     df_by_hour_year[(df_by_hour_year.year == year) & (df_by_hour_year.hour == hour_number)].counts,
    # #     #     bins=max_bin - min_bin,
    # #     #     range=[min_bin, max_bin],
    # #     # )
    # #     #
    # #     # number_of_days_in_data_set = (df.date.max() - df.date.min()).days
    # #     # print("number_of_days_in_data_set : ", number_of_days_in_data_set)
    # #     # print(
    # #     #     "df_by_hour[df_by_hour.hour == hour_number].date_number.unique().shape[0] : ",
    # #     #     df_by_hour[df_by_hour.hour == hour_number].date_number.unique().shape[0],
    # #     # )
    # #     # number_of_days_with_zero_accidents = (
    # #     #     number_of_days_in_data_set
    # #     #     - df_by_hour[df_by_hour.hour == hour_number].date_number.unique().shape[0]
    # #     # )
    # #     # entries = np.array([number_of_days_with_zero_accidents] + entries.tolist())
    # #     # bin_edges = np.array([0] + bin_edges.tolist())
    # #
    # #     parameters, cov_matrix = curve_fit(
    # #         poisson,
    # #         bin_edges[:-1],
    # #         entries,
    # #         p0=[200, 3],
    # #         bounds=[[1, 0.0001], [100000, 10]],
    # #     )
    # #     print("hist:")
    # #     print(list(zip(bin_edges[:-1], entries)))
    # #     print("parameters:")
    # #     print(parameters)
    # #     print("cov_matrix:")
    # #     print(cov_matrix)
    # #
    # #     def chisq(x, parameters, function, data):
    # #         Nexp = function(bin_edges[:-1], *parameters)
    # #         r = data - Nexp
    # #         stdev = np.std(data)
    # #         chisq = np.sum((r / stdev) ** 2)
    # #         ndf = len([entry for entry in entries if entry]) - 1
    # #         print("chisq =", chisq, "ndf =", ndf)
    # #         return chisq, ndf
    # #
    # #     chisq, ndf = chisq(
    # #         x=bin_edges[:-1], parameters=parameters, function=poisson, data=entries
    # #     )
    # #     fig, ax = plt.subplots()
    # #     ax.errorbar(
    # #         bin_edges[:-1], entries, fmt="r.", label=r"Data", yerr=np.sqrt(entries),
    # #     )
    # #     x = np.linspace(0, bin_edges[-1], 100)
    # #     ax.plot(x, poisson(x, *parameters), "r-", lw=2, label=r"Poisson fit")
    # #
    # #     result_string = f"Result :\n"
    # #     result_string = (
    # #         f"$\lambda$ : {parameters[1]:.2} $\pm$ {np.sqrt(cov_matrix[1][1]):.2}\n"
    # #     )
    # #     result_string += f"$\chi^2/ndf$ : {chisq:.2}/{ndf}\n"
    # #     ax.text(7, 20, result_string)
    # #     ax.set_ylim(0, 32)
    # #     # ax.set_yscale("log")
    # #     make_plotting_dirs()
    # #     fig.savefig(f"plots/day_year/{year}_{day}.png")
    # #     ax.cla()
    # #     heat_map_date = heat_map_date.append(
    # #         {"year": year, "day": day, "chisq": chisq, "lambda": parameters[1]},
    # #         ignore_index=True,
    # #     )
    #
    # for day in df_by_day_year.day_number.unique():
    #     for year in df_by_day_year.year.unique():
    #         min_bin, max_bin = (1, 15)
    #         entries, bin_edges = np.histogram(
    #             df_by_day_year[
    #                 (df_by_day_year.day_number == day) & (df_by_day_year.year == year)
    #             ].counts,
    #             bins=max_bin - min_bin,
    #             range=[min_bin, max_bin],
    #         )
    #
    #         # number_of_days_in_data_set = 52  # Fix this
    #         number_of_days_in_data_set = np.busday_count(
    #             f"{year}",
    #             f"{year+1}",
    #             weekmask=[
    #                 1 if i == day else 0
    #                 for i in range(len(df_by_day_year.day_number.unique()))
    #             ],
    #         )
    #         number_of_days_with_zero_accidents = (
    #             number_of_days_in_data_set
    #             - df_by_day_year[
    #                 (df_by_day_year.day_number == day) & (df_by_day_year.year == year)
    #             ]
    #             .date_number.unique()
    #             .shape[0]
    #         )
    #
    #         entries = np.array([number_of_days_with_zero_accidents] + entries.tolist())
    #         bin_edges = np.array([0] + bin_edges.tolist())
    #
    #         # entries, bin_edges = np.histogram(
    #         #     df_by_hour_year[(df_by_hour_year.year == year) & (df_by_hour_year.hour == hour_number)].counts,
    #         #     bins=max_bin - min_bin,
    #         #     range=[min_bin, max_bin],
    #         # )
    #         #
    #         # number_of_days_in_data_set = (df.date.max() - df.date.min()).days
    #         # print("number_of_days_in_data_set : ", number_of_days_in_data_set)
    #         # print(
    #         #     "df_by_hour[df_by_hour.hour == hour_number].date_number.unique().shape[0] : ",
    #         #     df_by_hour[df_by_hour.hour == hour_number].date_number.unique().shape[0],
    #         # )
    #         # number_of_days_with_zero_accidents = (
    #         #     number_of_days_in_data_set
    #         #     - df_by_hour[df_by_hour.hour == hour_number].date_number.unique().shape[0]
    #         # )
    #         # entries = np.array([number_of_days_with_zero_accidents] + entries.tolist())
    #         # bin_edges = np.array([0] + bin_edges.tolist())
    #
    #         parameters, cov_matrix = curve_fit(
    #             poisson,
    #             bin_edges[:-1],
    #             entries,
    #             p0=[200, 3],
    #             bounds=[[1, 0.0001], [100000, 10]],
    #         )
    #         print("hist:")
    #         print(list(zip(bin_edges[:-1], entries)))
    #         print("parameters:")
    #         print(parameters)
    #         print("cov_matrix:")
    #         print(cov_matrix)
    #
    #         def chisq(x, parameters, function, data):
    #             Nexp = function(bin_edges[:-1], *parameters)
    #             r = data - Nexp
    #             stdev = np.std(data)
    #             chisq = np.sum((r / stdev) ** 2)
    #             ndf = len([entry for entry in entries if entry]) - 1
    #             print("chisq =", chisq, "ndf =", ndf)
    #             return chisq, ndf
    #
    #         chisq, ndf = chisq(
    #             x=bin_edges[:-1], parameters=parameters, function=poisson, data=entries
    #         )
    #         fig, ax = plt.subplots()
    #         ax.errorbar(
    #             bin_edges[:-1], entries, fmt="r.", label=r"Data", yerr=np.sqrt(entries),
    #         )
    #         x = np.linspace(0, bin_edges[-1], 100)
    #         ax.plot(x, poisson(x, *parameters), "r-", lw=2, label=r"Poisson fit")
    #
    #         result_string = f"Result :\n"
    #         result_string = (
    #             f"$\lambda$ : {parameters[1]:.2} $\pm$ {np.sqrt(cov_matrix[1][1]):.2}\n"
    #         )
    #         result_string += f"$\chi^2/ndf$ : {chisq:.2}/{ndf}\n"
    #         ax.text(7, 20, result_string)
    #         ax.set_ylim(0, 32)
    #         # ax.set_yscale("log")
    #         make_plotting_dirs()
    #         fig.savefig(f"plots/day_year/{year}_{day}.png")
    #         ax.cla()
    #         heat_map_date = heat_map_date.append(
    #             {"year": year, "day": day, "chisq": chisq, "lambda": parameters[1]},
    #             ignore_index=True,
    #         )
    #
    # # plotting_grid = pd.DataFrame(heat_map_date)
    # plotting_grid = (
    #     heat_map_date[["year", "day", "lambda"]].set_index(["year", "day"]).unstack()
    # )
    # plotting_grid.columns = [f"{DAY_MAPPING_INV[j]}" for i, j in plotting_grid.columns]
    # plotting_grid.reset_index()
    # plotting_grid.index = plotting_grid.index.astype(int)
    # print(plotting_grid)
    #
    # fig, ax = plt.subplots(figsize=(20, 20))
    # g = sns.heatmap(plotting_grid, annot=True, ax=ax, fmt=".1f")
    # g.set_yticklabels(g.get_yticklabels(), rotation=0)
    # g.set_xticklabels(g.get_xticklabels(), rotation=45)
    # # ax.set_title("Harvest of local farmers (in tons/year)")
    # fig.tight_layout()
    # make_plotting_dirs()
    # fig.savefig("plots/time_heat_map.png")
    # ax.cla()
    # fig, ax = plt.subplots()
    # sns.lineplot(data=plotting_grid, ax=ax)
    # # sns.scatterplot(data=plotting_grid, ax=ax)
    #
    # # Shrink current axis by 20%
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # # Put a legend to the right of the current axis
    # ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    # make_plotting_dirs()
    # fig.savefig("plots/time_series.png")


def main():
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
    print(df)

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

    wards = get_ward_dataset()
    print(boros)
    print(wards)
    geom = [shape(i) for i in wards.location]
    wards_geom = gpd.GeoDataFrame({'geometry': geom})
    print(wards_geom)
    # BLUE = '#6699cc'
    # fig, ax = plt.subplots(figsize=(20, 16))
    # import geoplot
    # # wards_geom.geometry.plot(ax=ax)
    # geoplot.polyplot(wards_geom.geometry, figsize=(8, 4), ax=ax)

    # ax.add_patch(PolygonPatch(wards_geom.geometry, fc=BLUE, ec=BLUE, alpha=0.5, zorder=2))

    # df.plot(column='NUTS_ID', cmap='Purples', ax=ax)
    # df.geometry.boundary.plot(color=None, edgecolor='k', linewidth=2, ax=ax)
    # plt.show()

    # wards = gpd.read_file(
    #     "dataset/statistical-gis-boundaries-london/ESRI/London_Ward.shp"
    # )

    traffic_map = folium.Map(location=(51.536388, -0.140556), zoom_start=13)
    # for datum in heat_data:
    #     folium.CircleMarker(
    #         location=datum, radius=1, color="#3186cc", fill=True, fill_color="#3186cc"
    #     ).add_to(traffic_map)
    # folium.CircleMarker(
    #     location=heat_data, radius=1, color="#3186cc", fill=True, fill_color="#3186cc"
    # ).add_to(traffic_map)

    style = {"fillColor": "#228B22", "color": "#228B22"}

    def style_function(x):
        return {"fillColor": None, "color": "black", "weight": 1.5, "fillOpacity": 0.0}

    HeatMap(heat_data, radius=15, blur=5,).add_to(traffic_map)
    print(boros[boros.BOROUGH == "Camden"])
    folium.GeoJson(boros[boros.BOROUGH == "Camden"], style_function=style_function).add_to(traffic_map)
    # folium.GeoJson(wards_geom, style_function=style_function).add_to(traffic_map)
    traffic_map.save("cycalist_traffic_map.html")
    print("_____________")
    return

    df = df.set_index("reference")
    speed_limit_mapping = {k: v for v, k in enumerate(df.speed_limit.unique())}
    speed_limit_mapping_inv = {v: k for k, v in speed_limit_mapping.items()}

    df["speed_rank"] = df.speed_limit.map(speed_limit_mapping)
    print(df.corr())
    # print(dff.head())
    # print(dff.keys())

    # print(df.head())
    # print(df.keys())

    print(df.describe())
    # print(df.head())
    # print(df.keys())

    fig, ax = plt.subplots()
    ax.cla()
    n_bins = 6
    entries, bin_edges = np.histogram(df.speed_rank, bins=n_bins, range=[0, n_bins])
    center = (bin_edges[:-1] + bin_edges[1:]) / 2
    # parameters, cov_matrix = curve_fit(poisson, center, entries)
    # print "Parameters : ", parameters
    df.speed_rank.hist(ax=ax, range=[0, n_bins], bins=n_bins)
    # ax.plot(center, poisson(center, *parameters), 'r-', lw=2, label=r"Poisson fit : $\nu$ = {:.2f}".format(parameters[0]))

    # print(df.speed_rank.unique())
    # ax.set_xlim(0, len(df.speed_limit.unique()))
    ax.set_yscale("log")
    print(
        bin_edges,
        [
            speed_limit_mapping_inv[ind] if ind in speed_limit_mapping_inv else ""
            for ind in bin_edges
        ],
    )
    ax.set_xticks(bin_edges)
    ax.set_xticklabels(
        labels=[
            speed_limit_mapping_inv[ind] if ind in speed_limit_mapping_inv else ""
            for ind in bin_edges
        ],
        rotation=40,
        ha="right",
    )
    print(speed_limit_mapping)

    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    # main()
    # get_clean_casualties_dataset()
    possion_in_time()
    # lamp_posts(regenerate_data_cached_to_disk=False)
