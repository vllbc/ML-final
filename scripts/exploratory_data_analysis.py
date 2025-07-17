import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import os

PROCESSED_TRAIN_DATA_PATH = "data/processed_train.csv"
EDA_PLOTS_DIR = "plots/eda/"


def analyze_core_variable(df, output_dir):
    print("Analyzing core variable: Global_active_power...")
    plt.figure(figsize=(15, 7))
    df["Global_active_power"].plot()
    plt.title("Global Active Power Over Time")
    plt.xlabel("Date")
    plt.ylabel("Global Active Power")
    plt.savefig(f"{output_dir}/1_gap_time_series.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(df["Global_active_power"], kde=True)
    plt.title("Distribution of Global Active Power")
    plt.savefig(f"{output_dir}/2_gap_distribution.png")
    plt.close()

    decomposition = sm.tsa.seasonal_decompose(
        df["Global_active_power"], model="additive", period=365
    )
    fig = decomposition.plot()
    fig.set_size_inches(15, 12)
    plt.savefig(f"{output_dir}/3_gap_decomposition.png")
    plt.close()
    print("... Done.")


def analyze_variable_correlations(df, output_dir):
    print("Analyzing variable correlations...")
    plt.figure(figsize=(12, 10))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix of All Variables")
    plt.savefig(f"{output_dir}/4_correlation_heatmap.png")
    plt.close()

    plt.figure(figsize=(8, 8))
    sns.scatterplot(x="Global_intensity", y="Global_active_power", data=df)
    plt.title("Global Active Power vs. Global Intensity")
    plt.savefig(f"{output_dir}/5_scatter_power_vs_intensity.png")
    plt.close()

    plt.figure(figsize=(8, 8))
    sns.scatterplot(x="Voltage", y="Global_active_power", data=df)
    plt.title("Global Active Power vs. Voltage")
    plt.savefig(f"{output_dir}/6_scatter_power_vs_voltage.png")
    plt.close()
    print("... Done.")


def analyze_sub_metering(df, output_dir):
    print("Analyzing sub-metering data...")
    plt.figure(figsize=(15, 8))
    sub_metering_cols = [
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
        "sub_metering_remainder",
    ]
    df[sub_metering_cols].plot.area(stacked=True, figsize=(15, 8), linewidth=0.5)
    plt.title("Power Consumption by Sub-metering")
    plt.xlabel("Date")
    plt.ylabel("Power (Watt-hour)")
    plt.legend(title="Sub-metering")
    plt.savefig(f"{output_dir}/7_sub_metering_stacked_area.png")
    plt.close()
    print("... Done.")


def analyze_periodicity(df, output_dir):
    print("Analyzing periodicity...")
    df_copy = df.copy()
    df_copy["month"] = df_copy.index.month
    df_copy["weekday"] = df_copy.index.day_name()

    plt.figure(figsize=(12, 6))
    sns.boxplot(x="month", y="Global_active_power", data=df_copy)
    plt.title("Monthly Power Consumption Pattern")
    plt.savefig(f"{output_dir}/8_monthly_boxplot.png")
    plt.close()

    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="weekday", y="Global_active_power", data=df_copy, order=weekday_order)
    plt.title("Weekly Power Consumption Pattern")
    plt.savefig(f"{output_dir}/9_weekly_boxplot.png")
    plt.close()

    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    sm.graphics.tsa.plot_acf(df["Global_active_power"], lags=40, ax=axes[0])
    sm.graphics.tsa.plot_pacf(df["Global_active_power"], lags=40, ax=axes[1])
    plt.savefig(f"{output_dir}/10_acf_pacf_plots.png")
    plt.close()
    print("... Done.")


def analyze_external_variables(df, output_dir):
    print("Analyzing external variables...")
    fig, ax1 = plt.subplots(figsize=(15, 7))

    color = "tab:blue"
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Global Active Power", color=color)
    ax1.plot(df.index, df["Global_active_power"], color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("RR", color=color)
    ax2.plot(df.index, df["RR"], color=color, alpha=0.6)
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("Global Active Power vs. RR Variable")
    fig.tight_layout()
    plt.savefig(f"{output_dir}/11_power_vs_rr.png")
    plt.close()
    print("... Done.")


def main():
    print("Starting Exploratory Data Analysis...")
    os.makedirs(EDA_PLOTS_DIR, exist_ok=True)

    df = pd.read_csv(PROCESSED_TRAIN_DATA_PATH, index_col="DateTime", parse_dates=True)

    analyze_core_variable(df, EDA_PLOTS_DIR)
    analyze_variable_correlations(df, EDA_PLOTS_DIR)
    analyze_sub_metering(df, EDA_PLOTS_DIR)
    analyze_periodicity(df, EDA_PLOTS_DIR)
    analyze_external_variables(df, EDA_PLOTS_DIR)

    print("\nExploratory Data Analysis complete.")
    print(f"All plots saved in '{EDA_PLOTS_DIR}' directory.")


if __name__ == "__main__":
    main()
