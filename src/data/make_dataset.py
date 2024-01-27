import pandas as pd


def load_data(file_path):
    """
    Load CSV data from the specified file path.
    """
    return pd.read_csv(file_path)


def preprocess_datetime(df, column_name, date_format=None):
    """
    Convert the specified column to datetime format in the DataFrame.
    """
    df[column_name] = pd.to_datetime(df[column_name], format=date_format)
    return df


def set_datetime_index(df, column_name):
    """
    Set the specified column as the datetime index in the DataFrame.
    """
    df = df.set_index(column_name)
    return df


def main():
    # Load data
    train_df = load_data("../../data/raw/eso-battery-forecasting/train_data.csv")
    val_df = load_data("../../data/raw/eso-battery-forecasting/val_data.csv")

    # Display information about the datasets
    # print("Train Data Info:")
    # train_df.info(verbose=True)
    # print("\nval Data Info:")
    # val_df.info(verbose=True)

    # Preprocess datetime columns
    train_df = preprocess_datetime(train_df, "UTC_Settlement_DateTime")
    val_df = preprocess_datetime(
        val_df, "UTC_Settlement_DateTime", date_format="%d/%m/%Y %H:%M"
    )

    # Set datetime columns as index
    # train_df = set_datetime_index(train_df, "UTC_Settlement_DateTime")
    # val_df = set_datetime_index(val_df, "UTC_Settlement_DateTime")

    return train_df, val_df


train_df, val_df = main()

train_df.to_csv("../../data/raw/train_data.csv", index=False)

val_df.to_csv("../../data/raw/val_data.csv", index=False)


if __name__ == "__main__":
    train_data, val_data = main()
