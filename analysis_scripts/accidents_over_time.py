import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.seasonal import seasonal_decompose
from fbprophet import Prophet

from gds.demo.data.data_methods import (
    get_casualties_dataset,
)

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

plt.style.use("ggplot")


def get_time_data():

    df = get_casualties_dataset()

    fields = {
        "reference": "string",
        "date": "string",
        "casualty_severity": "string",
    }
    df = df[fields.keys()]
    df = df.dropna()
    df = df.astype(fields)
    df = df.drop_duplicates()

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

    return df


def plot_collected_accidents():
    plt.style.use("./mplstyles/custom_style.mplstyle")
    df = get_time_data()
    fig, ax = plt.subplots(figsize=(15, 8))
    counts, xedges, yedges, im = ax.hist2d(
        df.hour, df.day_number + 0.5, bins=(24, 7), range=((0, 24), (0, 7))
    )
    ax.set_xlabel("Hour of day")
    # ax.set_ylabel("Day")
    ax.set_yticks(yedges[:-1] + 0.5)
    ax.set_yticklabels(
        labels=[
            DAY_MAPPING_INV[ind] if ind in DAY_MAPPING_INV else ""
            for ind in yedges[:-1]
        ],
        ha="right",
    )

    plt.tight_layout()
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Number of accidents")
    fig.savefig("plots/accidents_per_day_hour_transpose.png")

    fig, ax = plt.subplots(figsize=(15, 8))

    dff = df.groupby(["day_number"])["date_number"].count().reset_index(name="counts")
    dff = {row["day_number"]: row["counts"] for _, row in dff.iterrows()}
    df["day_number_norm"] = df.apply(
        lambda row: 1.0 / float(dff[row["day_number"]]), axis=1
    )
    counts, xedges, yedges, im = ax.hist2d(
        df.hour,
        df.day_number + 0.5,
        weights=df.day_number_norm,
        bins=(24, 7),
        range=((0, 24), (0, 7)),
    )
    ax.set_xlabel("Hour of day")
    # ax.set_ylabel("Day")
    ax.set_yticks(yedges[:-1] + 0.5)
    ax.set_yticklabels(
        labels=[
            DAY_MAPPING_INV[ind] if ind in DAY_MAPPING_INV else ""
            for ind in yedges[:-1]
        ],
        ha="right",
    )

    plt.tight_layout()
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Probability of accidents normalised by day")
    fig.savefig("plots/accidents_per_day_hour_transpose_norm_by_day.png")


def plot_accidents_over_time():
    plt.style.use("./mplstyles/custom_style.mplstyle")
    register_matplotlib_converters()
    df = get_time_data()

    accidents_data_month_total = df.groupby("date")[["month"]].count().reset_index()
    accidents_data_month_total = accidents_data_month_total.set_index("date")
    accidents_data_month_total.index = pd.to_datetime(accidents_data_month_total.index)
    accidents_data_month_total = (
        accidents_data_month_total["month"].resample("MS").sum()
    )
    accidents_data_month_total.rename("accidents_data_month_total")

    accidents_data_month_fatal = (
        df[df.casualty_severity == "1 Fatal"]
        .groupby("date")[["month"]]
        .count()
        .reset_index()
    )
    accidents_data_month_fatal = accidents_data_month_fatal.set_index("date")
    accidents_data_month_fatal.index = pd.to_datetime(accidents_data_month_fatal.index)
    accidents_data_month_fatal = (
        accidents_data_month_fatal["month"].resample("MS").sum()
    )
    accidents_data_month_fatal.rename("accidents_data_month_fatal")

    accidents_data_month_serious = (
        df[df.casualty_severity == "2 Serious"]
        .groupby("date")[["month"]]
        .count()
        .reset_index()
    )
    accidents_data_month_serious = accidents_data_month_serious.set_index("date")
    accidents_data_month_serious.index = pd.to_datetime(
        accidents_data_month_serious.index
    )
    accidents_data_month_serious = (
        accidents_data_month_serious["month"].resample("MS").sum()
    )
    accidents_data_month_serious.rename("accidents_data_month_serious")

    accidents_data_month_slight = (
        df[df.casualty_severity == "3 Slight"]
        .groupby("date")[["month"]]
        .count()
        .reset_index()
    )
    accidents_data_month_slight = accidents_data_month_slight.set_index("date")
    accidents_data_month_slight.index = pd.to_datetime(
        accidents_data_month_slight.index
    )
    accidents_data_month_slight = (
        accidents_data_month_slight["month"].resample("MS").sum()
    )
    accidents_data_month_slight.rename("accidents_data_month_slight")

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot_date(
        accidents_data_month_total.index,
        y=accidents_data_month_total,
        linestyle="solid",
        marker="None",
        label="Total",
    )
    ax.plot_date(
        accidents_data_month_fatal.index,
        y=accidents_data_month_fatal,
        linestyle="solid",
        marker="None",
        label="Fatal",
    )
    ax.plot_date(
        accidents_data_month_serious.index,
        y=accidents_data_month_serious,
        linestyle="solid",
        marker="None",
        label="Serious",
    )
    ax.plot_date(
        accidents_data_month_slight.index,
        y=accidents_data_month_slight,
        linestyle="solid",
        marker="None",
        label="Slight",
    )

    ax.set_ylabel("Number of accidents per month")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    fig.savefig("plots/accidents_rate_over_time.png")

    plt.style.use("ggplot")
    result = seasonal_decompose(accidents_data_month_total)
    fig = result.plot()
    fig.savefig("plots/seasonal_decompose_total.png")

    fig, ax = plt.subplots(figsize=(14, 8))
    # run a facebook like forecast
    model = Prophet()
    train_df = pd.DataFrame()
    train_df["y"] = accidents_data_month_total
    train_df["ds"] = accidents_data_month_total.index
    model.fit(train_df)
    future = model.make_future_dataframe(10, freq='M', include_history=True)
    forecast = model.predict(future)
    model.plot(forecast, ax=ax)
    ax.set_ylabel("Total number of accidents per month")
    ax.set_xlabel("Year")
    fig.savefig('plots/10_month_projection_total.png')


if __name__ == "__main__":
    import os
    os.makedirs("plots/", exist_ok=True)
    plot_accidents_over_time()
    plot_collected_accidents()
