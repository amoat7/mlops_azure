# Import libraries

import argparse
import glob
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
import mlflow
from sklearn.model_selection import train_test_split


# define functions
def main(args):
    # enable autologging
    mlflow.sklearn.autolog()
    # read data
    df = get_csvs_df(args.training_data)

    # split data
    x_train, x_test, y_train, y_test = split_data(df)

    # train model
    train_model(args.reg_rate, x_train, x_test, y_train, y_test)


def get_csvs_df(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")
    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(f"No CSV files found in provided data path: {path}")
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)


# function to split data
def split_data(df):
    x_data, y_data = (
        df[
            [
                "Pregnancies",
                "PlasmaGlucose",
                "DiastolicBloodPressure",
                "TricepsThickness",
                "SerumInsulin",
                "BMI",
                "DiabetesPedigree",
                "Age",
            ]
        ].values,
        df["Diabetic"].values,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.30, random_state=0
    )
    return (x_train, x_test, y_train, y_test)


def train_model(reg_rate, x_train, x_test, y_train, y_test):
    # train model
    LogisticRegression(C=1 / reg_rate, solver="liblinear").fit(x_train, y_train)


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest="training_data", type=str)
    parser.add_argument("--reg_rate", dest="reg_rate", type=float, default=0.01)

    # parse args
    args = parser.parse_args()

    # return args
    return args


# run script
if __name__ == "__main__":
    # add space in logs
    print("\n\n")
    print("*" * 60)

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")
