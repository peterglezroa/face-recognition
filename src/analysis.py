"""
analysis.py
-----------
Script to analyze the results from the run_tests script
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RESULTS_PATH = "../results.csv"

def main():
    # Set seaborn theme
    sns.set_theme(style="whitegrid")

    base_df = pd.read_csv(RESULTS_PATH)

    # Drop Date col
    base_df = base_df.drop(columns=["Date"])

    # Define types of cols
    id_vars = ["Dataset", "# People"]
    time_cols = [col for col in base_df if col.endswith("time")]

    # Melt for better analysis
    df = base_df.drop(columns=time_cols)
    val_vars = [col for col in df if col not in id_vars]
    df = base_df.melt(id_vars=id_vars, value_vars=val_vars)
    df["cluster alg"] = df["variable"].str.split(' ').str[-1]
    print(df)

    ax = sns.catplot(kind="box", data=df, x="cluster alg", y="value")
    plt.show()

    # Melt to analyse agglomerative vs kmeans
    
    # Melt to analyse single, cc, mvc

if __name__ == "__main__":
    main()
