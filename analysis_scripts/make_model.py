import os
import pickle
import numpy as np
import pandas as pd
from time import time
from pprint import pprint

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from gds.demo.data.data_methods import get_casualties_dataset, reduce_weather_categories


def train_predict(learner, X_train, y_train, X_test, y_test):
    """
    train learner on training data and pull out some important metrics
    """
    results = {}

    start = time()  # Get start time
    learner = learner.fit(X_train, y_train)
    end = time()  # Get end time

    results["train_time"] = end - start

    start = time()  # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train)
    end = time()  # Get end time

    results["pred_time"] = end - start
    results["acc_train"] = accuracy_score(y_train, predictions_train)
    results["acc_test"] = accuracy_score(y_test, predictions_test)
    results["f_train"] = fbeta_score(
        y_train, predictions_train, beta=0.5, average="micro"
    )
    results["f_test"] = fbeta_score(y_test, predictions_test, beta=0.5, average="micro")
    print("{} trained on {} samples.".format(learner.__class__.__name__, len(y_test)))
    return results


def evaluate(results):
    """
    Visualization code to display results of various learners.
    """
    subplot_titles = (
        "train_time",
        "acc_train",
        "f_train",
        "pred_time",
        "acc_test",
        "f_test",
    )
    # fig = tools.make_subplots(rows=2, cols=3, subplot_titles=subplot_titles)
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 10))
    ax = [item for sublist in ax for item in sublist]

    result_df = pd.DataFrame.from_dict(results)
    x = result_df.index.to_list()

    for j, metric in enumerate(
        ["train_time", "acc_train", "f_train", "pred_time", "acc_test", "f_test"]
    ):
        result_df.T[metric].plot.bar(x=result_df.T[metric].index, y=metric, ax=ax[j])
        ax[j].set_xticklabels(ax[j].get_xticklabels(), rotation=45, ha="right")
        ax[j].set_title(metric)
        if j not in [0, 3]:
            ax[j].set_ylim(0, 1.0)
            ax[j].plot([-1.0, 4.5], [0.33, 0.33], "k--")
            ax[j].plot([-1.0, 4.5], [0.5, 0.5], "k--")

    fig.tight_layout()
    fig.savefig("plots/Model_Evaluation_Result.png")


def feature_plot(importances, X_train):
    "Plot the top 20 features in the model and their cumulative sum"
    fig, ax = plt.subplots()

    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:20]]
    values = importances[indices][:20]
    cumulative_weight = np.cumsum(values)

    ax.bar(columns, values)
    ax.scatter(columns, cumulative_weight)

    ax.set_xticklabels(columns, rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig("plots/Model_Feature_Importances.png")


def group_severity(row):
    if row["casualty_severity"] in ["1 Fatal", "2 Serious"]:
        return 1
    else:
        return 0


def data_two_category_all_data(data):
    """slice data such that on two severity categories exist, use all of the data"""
    data["casualty_severity"] = data.apply(lambda x: group_severity(x), axis=1)
    severity_raw = data["casualty_severity"]
    features_raw = data.drop("casualty_severity", axis=1)
    data_manip_identifer = "two_category_all_data"
    return severity_raw, features_raw, data_manip_identifer


def data_two_category_balanced_data(data):
    """ Slice data such that on two severity categories exist, balance the data """
    data["casualty_severity"] = data.apply(lambda x: group_severity(x), axis=1)

    df_fatal = data[data.casualty_severity == 1]
    n_fatal = df_fatal.shape[0]
    df_slight = data[data.casualty_severity == 0].sample(n=n_fatal)
    data = pd.concat([df_slight, df_fatal])
    data = data.sample(frac=1)

    severity_raw = data["casualty_severity"]
    features_raw = data.drop("casualty_severity", axis=1)
    data_manip_identifer = "two_category_balanced_data"
    return severity_raw, features_raw, data_manip_identifer


def train_supervised_model():
    accidents_data = get_casualties_dataset()
    accidents_data.dropna(inplace=True)
    data = accidents_data.copy()

    fields = {
        "reference": "string",
        "casualty_severity": "category",
        "weather": "category",
        "ward_name": "category",
        "date": "string",
    }
    data = data[fields.keys()]
    data = data.astype(fields)
    data = data.set_index("reference")

    data["datetime"] = pd.to_datetime(data.date, format="%Y-%m-%dT%H:%M:%S")
    data["hour"] = pd.DatetimeIndex(data.datetime).hour
    data["hour"] = data["hour"].astype("category")
    data["day"] = pd.DatetimeIndex(data.datetime).strftime("%A")
    data["day"] = data["day"].astype("category")

    data.drop(["date", "datetime"], axis=1, inplace=True)

    data.weather = data.weather.map(reduce_weather_categories)

    # Pick the data you want to train on
    # severity_raw, features_raw, data_manip_identifer = data_two_category_all_data(data)
    severity_raw, features_raw, data_manip_identifer = data_two_category_balanced_data(
        data
    )

    filted_data = pd.get_dummies(features_raw).astype(np.float64)

    X_train, X_test, y_train, y_test = train_test_split(
        filted_data, severity_raw, test_size=0.2, random_state=0,
    )
    clf_A = DecisionTreeClassifier(random_state=9)
    clf_B = AdaBoostClassifier(random_state=9)
    clf_C = GradientBoostingClassifier(random_state=9)
    clf_D = RandomForestClassifier(random_state=9)

    results = {}
    for clf in [clf_A, clf_B, clf_C, clf_D]:
        clf_name = clf.__class__.__name__
        clf_name = clf_name.replace("Classifier", "")
        results[clf_name] = train_predict(clf, X_train, y_train, X_test, y_test)

    pprint(results)
    evaluate(results)

    best_clf = RandomForestClassifier(
        criterion="gini",
        max_depth=15,
        max_features="auto",
        min_samples_split=100,
        n_estimators=50,
        random_state=9,
    )

    clf = clf_D
    # Get the estimator
    best_clf = best_clf.fit(X_train, y_train)
    # Make predictions using the unoptimized and model
    predictions = (clf.fit(X_train, y_train)).predict(X_test)
    best_predictions = best_clf.predict(X_test)

    # Report the before-and-afterscores
    print("Unoptimized model\n------")
    print(
        "Accuracy score on testing data: {:.4f}".format(
            accuracy_score(y_test, predictions)
        )
    )
    print(
        "F-score on testing data: {:.4f}".format(
            fbeta_score(y_test, predictions, beta=0.5, average="micro")
        )
    )
    print("\nOptimized Model\n------")
    print(
        "Final accuracy score on the testing data: {:.4f}".format(
            accuracy_score(y_test, best_predictions)
        )
    )
    print(
        "Final F-score on the testing data: {:.4f}".format(
            fbeta_score(y_test, best_predictions, beta=0.5, average="micro")
        )
    )

    feature_plot(best_clf.feature_importances_, X_train)

    pickle.dump(
        best_clf, open(f"trained_models/{data_manip_identifer}_random_tree.pkl", "wb")
    )
    return


if __name__ == "__main__":
    os.makedirs("trained_models/", exist_ok=True)
    os.makedirs("plots/", exist_ok=True)
    train_supervised_model()
