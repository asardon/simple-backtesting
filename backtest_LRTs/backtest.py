import sys
import os
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import (
    plot_best_strategy_vs_hold,
    backtest_reinvesting_strategy_shifted_starts,
    get_historical_price_data,
    get_eth_vol_data,
    plot_vol_surface,
)


def calculate_final_nav_for_strikes(
    strikes, tenor, price_df, eth_call_spline, vol_shift=0.0, stop_once_converted=False
):
    results = []
    for rel_strike in strikes:
        (
            final_nav,
            avg_final_return,
            _,
            _,
            avg_final_notional_units,
        ) = backtest_reinvesting_strategy_shifted_starts(
            price_df, tenor, rel_strike, eth_call_spline, vol_shift, stop_once_converted
        )

        result_entry = {
            "Tenor": tenor,
            "Strike (%)": rel_strike * 100,
            "Final NAV": final_nav,
            "Avg. Final Return (%)": avg_final_return * 100,
            "Avg. Notional Units": avg_final_notional_units,
        }
        results.append(result_entry)

        # Plot strategy and save image
        filename = (
            f"7d_{int(rel_strike * 100)}_stop_once_converted_{stop_once_converted}.png"
        )
        plot_best_strategy_vs_hold(
            price_df,
            tenor,
            rel_strike,
            eth_call_spline,
            vol_shift,
            stop_once_converted,
            filename,
        )

    # Convert results to DataFrame for sorting and display
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="Avg. Notional Units", ascending=False)

    return results_df


def print_results_table(df):
    df_sorted = df.sort_values(by="Avg. Notional Units", ascending=False).round(3)

    headers = [
        "Tenor",
        "Strike (%)",
        "Final NAV",
        "Avg. Final Return (%)",
        "Avg. Notional Units",
    ]
    col_widths = [
        max(len(str(x)), len(header))
        for x, header in zip(df_sorted.max().astype(str), headers)
    ]

    print(f"+{'-+-'.join('-' * w for w in col_widths)}+")

    header_row = " | ".join(
        f"{header:<{col_widths[i]}}" for i, header in enumerate(headers)
    )
    print(f"| {header_row} |")
    print(f"+{'-+-'.join('-' * w for w in col_widths)}+")

    for _, row in df_sorted.iterrows():
        row_data = " | ".join(
            f"{str(row[col]):<{col_widths[i]}}" for i, col in enumerate(headers)
        )
        print(f"| {row_data} |")

    print(f"+{'-+-'.join('-' * w for w in col_widths)}+")


if __name__ == "__main__":
    # Load price data (Ethereum example)
    price_df = get_historical_price_data(
        coin_id="ethereum", vs_currency="usd", days=365
    )

    vol_shift = 0.0

    # Get the ETH volatility data and generate the spline
    df = get_eth_vol_data()
    eth_call_spline = plot_vol_surface(df, vol_shift, filename="vol_surface.png")

    # Define the strikes to be tested
    strikes = [
        1.0,
        1.05,
        1.1,
        1.15,
        1.2,
        1.25,
        1.3,
        1.35,
        1.4,
        1.5,
        1.55,
        1.6,
        1.65,
        1.7,
        1.75,
        1.8,
    ]
    tenor = 7  # Fixed tenor (7 days)
    stop_once_converted = False
    results_df = calculate_final_nav_for_strikes(
        strikes, tenor, price_df, eth_call_spline, vol_shift, stop_once_converted
    )

    # Print the results sorted by Final NAV
    print_results_table(results_df)
