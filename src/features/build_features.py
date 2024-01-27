import pandas as pd


train_df = pd.read_csv("../../data/raw/train_data.csv")
val_df = pd.read_csv("../../data/raw/val_data.csv")


# Essential feautures

column_list = [
    "battery_output_lag1",
    "battery_output_lag2",
    "battery_output_lag48",
    "windspeed_100mNewcastle upon Tyne_weather",
    "is_dayNewcastle upon Tyne_weather",
]
label = ["battery_output"]
# feature selection for train_data

train_df["battery_output_lag1"] = train_df["battery_output"].shift(1)
train_df["battery_output_lag2"] = train_df["battery_output"].shift(2)
train_df["battery_output_lag48"] = train_df["battery_output"].shift(48)

mean_value = train_df["battery_output"].mean()
train_df["battery_output_lag1"] = train_df["battery_output_lag1"].fillna(mean_value)
train_df["battery_output_lag2"] = train_df["battery_output_lag2"].fillna(mean_value)
train_df["battery_output_lag48"] = train_df["battery_output_lag48"].fillna(mean_value)

train_df = train_df[column_list + label]

train_df = train_df.ffill().bfill()


# feature selection for val_data


val_df["battery_output_lag1"] = 0
val_df["battery_output_lag2"] = 0
val_df["battery_output_lag48"] = 0


val_df = val_df[column_list]

val_df = val_df.ffill().bfill()

# To csv

train_df.to_csv("../../data/interim/data.csv", index=False)
val_df.to_csv("../../data/interim/val.csv", index=False)


X = pd.read_pickle("../../data/processed/X_train.pkl")
pd.read_pickle("../../data/processed/X_test.pkl")
pd.read_pickle("../../data/processed/val.pkl")

X.loc[:, "windspeed_100mNewcastle upon Tyne_weather"]
X["windspeed_100mNewcastle upon Tyne_weather"]

pd.read_csv("../../data/raw/val_data.csv")

train_df[31562:]
