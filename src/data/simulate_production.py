# src/data/simulate_production.py

import pandas as pd
import numpy as np
import os

OUTPUT_DIR = "data/production"


def simulate_day(reference_df, reference_y, day):
    prod = reference_df.sample(frac=0.3, random_state=day)
    prod_y = reference_y.loc[prod.index]

    if day == 2:
        prod["AGE"] += 7
    if day == 3:
        prod["LIMIT_BAL"] *= 0.65
    if day == 4:
        prod["PAY_0"] += np.random.choice([0, 1, 2], size=len(prod))

    return prod, prod_y


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True) 

    # IMPORTANT: preserve index
    reference = pd.read_csv("data/reference.csv", index_col=0)

    reference_y = pd.read_csv("data/reference_target.csv")
    reference_y.index = reference.index
    reference_y = reference_y.squeeze()

    for day in range(1, 5):
        prod, prod_y = simulate_day(reference, reference_y, day)

        prod.to_csv(f"{OUTPUT_DIR}/day_{day:02d}.csv", index=True)
        prod_y.to_csv(
            f"{OUTPUT_DIR}/day_{day:02d}_labels.csv",
            index=True
        )

        print(f"Generated day_{day:02d}.csv + labels")


if __name__ == "__main__":
    main()
