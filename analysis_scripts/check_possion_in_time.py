import os
from scipy.special import factorial
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from gds.demo.data.data_methods import (
    get_casualties_dataset,
    gen_fake_possion_data,
)


plt.style.use("./mplstyles/custom_style.mplstyle")

# plt.style.use("ggplot")

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


def calc_chisq(x, parameters, function, data):
    exp = function(x, *parameters)
    r = data - exp
    stdev = np.std(data)
    chisq = np.sum((r / stdev) ** 2)
    ndf = len([entry for entry in data if entry]) - 1
    # print("chisq =", chisq, "ndf =", ndf)
    return chisq, ndf


def slice_by_hour(df):
    min_bin, max_bin = (1, 15)
    bin_range = (1, 15)
    n_bins = max_bin - min_bin

    for indices, group in df.groupby(["hour"]):
        hour = indices
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
        # print("hist:")
        # print(list(zip(bin_edges[:-1], entries)))
        # print("parameters:")
        # print(parameters)
        # print("cov_matrix:")
        # print(cov_matrix)

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
        # print("hist:")
        # print(list(zip(bin_edges[:-1], entries)))
        # print("parameters:")
        # print(parameters)
        # print("cov_matrix:")
        # print(cov_matrix)

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
        # print("hist:")
        # print(list(zip(bin_edges[:-1], entries)))
        # print("parameters:")
        # print(parameters)
        # print("cov_matrix:")
        # print(cov_matrix)

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
            {
                "year": year,
                "day": day,
                "chisq": chisq,
                "lambda": parameters[1],
                "lambda_err": cov_matrix[1][1],
            },
            ignore_index=True,
        )
    plotting_grid = (
        heat_map_date[["year", "day", "lambda"]].set_index(["year", "day"]).unstack()
    )
    plotting_grid.columns = [f"{DAY_MAPPING_INV[j]}" for i, j in plotting_grid.columns]
    plotting_grid.reset_index()
    plotting_grid.index = plotting_grid.index.astype(int)

    plotting_grid_err = (
        heat_map_date[["year", "day", "lambda_err"]]
        .set_index(["year", "day"])
        .unstack()
    )

    # plotting_grid_err.columns = [f"{DAY_MAPPING_INV[j]}" for i, j in plotting_grid.columns]
    # plotting_grid_err.reset_index()
    # plotting_grid_err.index = plotting_grid.index.astype(int)

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


def get_clean_time_data(fake_date: bool = False):

    if fake_date:
        df = gen_fake_possion_data()
    else:
        df = get_casualties_dataset()

    fields = {
        "reference": "string",
        "date": "string",
    }
    df = df[fields.keys()]
    df = df.dropna()
    df = df.astype(fields)
    # As we are only interesting in single accidents not the number of causalities
    # print(f"before duplicated drop : {df.shape}")
    df = df.drop_duplicates()
    # print(f"after duplicated drop : {df.shape}")

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

    df = df.set_index("reference")

    return df


def possion_in_time():

    df = get_clean_time_data()

    # # events per month
    # dff = (
    #     df.groupby(["month_number", "date_number"])["date_number"]
    #     .count()
    #     .reset_index(name="counts")
    # )
    # dfff = [
    #     [month, dff[dff.month_number == month].counts.mean()]
    #     for month in dff.month_number.unique()
    # ]
    # dfff = pd.DataFrame.from_records(dfff, columns=["month_number", "mean_counts"])
    # fig, ax = plt.subplots(nrows=2)
    # ax[0].plot(dfff.month_number, dfff.mean_counts)
    # ax[1].hist(dfff.mean_counts)
    # plt.tight_layout()
    # make_plotting_dirs()
    # fig.savefig("plots/mean_accidents_per_month.png")

    # fig, ax = plt.subplots(figsize=(15, 8))
    # counts, xedges, yedges, im = ax.hist2d(
    #     df.hour, df.day_number + 0.5, bins=(24, 7), range=((0, 24), (0, 7))
    # )
    # ax.set_xlabel("Hour of day")
    # # ax.set_ylabel("Day")
    # ax.set_yticks(yedges[:-1] + 0.5)
    # ax.set_yticklabels(
    #     labels=[
    #         DAY_MAPPING_INV[ind] if ind in DAY_MAPPING_INV else ""
    #         for ind in yedges[:-1]
    #     ],
    #     ha="right",
    # )
    #
    # plt.tight_layout()
    # cbar = fig.colorbar(im, ax=ax)
    # cbar.set_label("Number of accidents")
    # make_plotting_dirs()
    # fig.savefig("plots/accidents_per_day_hour_transpose.png")
    #
    # fig, ax = plt.subplots(figsize=(15, 8))
    #
    # dff = df.groupby(["day_number"])["date_number"].count().reset_index(name="counts")
    # dff = {row["day_number"]: row["counts"] for _, row in dff.iterrows()}
    # print(type(dff))
    # print(dff)
    # df["day_number_norm"] = df.apply(
    #     lambda row: 1.0 / float(dff[row["day_number"]]), axis=1
    # )
    # # df["day_number_norm"] = df.apply(lambda row: 1.0 / row["day_number"], axis=1)
    # print(df)
    # counts, xedges, yedges, im = ax.hist2d(
    #     df.hour,
    #     df.day_number + 0.5,
    #     weights=df.day_number_norm,
    #     bins=(24, 7),
    #     range=((0, 24), (0, 7)),
    # )
    # ax.set_xlabel("Hour of day")
    # # ax.set_ylabel("Day")
    # ax.set_yticks(yedges[:-1] + 0.5)
    # ax.set_yticklabels(
    #     labels=[
    #         DAY_MAPPING_INV[ind] if ind in DAY_MAPPING_INV else ""
    #         for ind in yedges[:-1]
    #     ],
    #     ha="right",
    # )
    #
    # plt.tight_layout()
    # cbar = fig.colorbar(im, ax=ax)
    # cbar.set_label("Probability of accidents normalised by day")
    # make_plotting_dirs()
    # fig.savefig("plots/accidents_per_day_hour_transpose_norm_by_day.png")
    #
    # ax.cla()

    slice_by_day(df)
    slice_by_hour(df)
    slice_by_day_and_year(df)
    return


if __name__ == "__main__":
    possion_in_time()
