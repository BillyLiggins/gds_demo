from sodapy import Socrata
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
from gds.demo.utils import singleton
import random
from datetime import datetime
# from functools import lru_cache

# @lru_cache(maxsize=10)
@singleton
def get_socrata_client():
    return Socrata("opendata.camden.gov.uk", None)


# @lru_cache(maxsize=10)
@singleton
def get_ward_dataset():
    results = get_socrata_client().get("cvwp-5m5j", limit=25000)
    return pd.DataFrame.from_records(results)


# @lru_cache(maxsize=10)
@singleton
def get_lighting_dataset():
    results = get_socrata_client().get("dfq3-8wzu", limit=11700)
    return pd.DataFrame.from_records(results)

# @lru_cache(maxsize=10)
@singleton
def get_casualties_dataset():
    results = get_socrata_client().get("puar-wf4h", limit=25000)
    return pd.DataFrame.from_records(results)


def nextTime(rateParameter):
    return -np.log(1.0 - random.random()) / rateParameter


def gen_fake_possion_data():

    time = []
    date = []
    day = []
    ref = []
    date_now = datetime(day=1, month=1, year=2005)
    rate = 0.00004
    for ev in range(10000):
        date_now += relativedelta(seconds=nextTime(rate))
        # if random.random() < 0.1:
        #     rate = random.random() * 0.002
        ref.append(ev)
        date.append(date_now.isoformat())
        time.append(date_now.strftime("%H.%M"))
        day.append(date_now.strftime("%A"))

    df = pd.DataFrame(data={"reference": ref, "date": date, "day": day, "time": time})
    return df


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
    print(ward_name_loc_mapping)
    print(ward_name_loc_mapping.shape)

    return ward_name_loc_mapping
