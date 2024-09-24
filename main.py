import requests
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.interpolate import SmoothBivariateSpline
from scipy import optimize
from datetime import datetime, date
import matplotlib.pyplot as plt
import seaborn as sns

RISK_FREE_RATE = 0.05


def plot_heatmap_strategy_performance(df, title, filename="heatmap_plot.png"):
    heatmap_data = df.pivot(
        index="Strike (%)", columns="Tenor", values="Avg. Final Return (%)"
    )

    plt.figure(figsize=(15, 12))
    sns.heatmap(
        heatmap_data,
        cmap="viridis",
        annot=True,
        fmt=".1f",
        cbar_kws={"label": "Avg. Final Return (%)"},
    )

    plt.title(title)
    plt.xlabel("Tenor (days)")
    plt.ylabel("Strike (%)")

    plt.savefig(filename, bbox_inches="tight")
    print(f"Saved heatmap to {filename}")


def backtest_reinvesting_strategy(
    df, tenor, rel_strike, call_spline, vol_shift=0.0, stop_once_converted=True
):
    tenor = int(tenor)
    moneyness = 1.0 / rel_strike
    implied_vol = float(call_spline.ev(moneyness, tenor)) + vol_shift
    option_premium = black_scholes_call(
        100, 100 * rel_strike, tenor / 365.0, RISK_FREE_RATE, implied_vol
    )
    rel_option_premium = option_premium / 100.0
    cumulative_nav = []
    conversions = []
    price_data = []
    strike_prices = []
    notional_units_over_time = []

    strategy_stop = False
    current_tenor_day = 0
    start_price = df["price"].values[0]

    start_nav = df["price"].values[0] + df["price"].values[0] * rel_option_premium
    notional_units = 1
    end_nav = None

    cumulative_nav.append((df["date"].iloc[0], start_nav))
    notional_units_over_time.append((df["date"].iloc[0], notional_units))

    for i in range(1, len(df)):
        current_price = df["price"].iloc[i]
        current_date = df["date"].iloc[i]
        price_data.append((current_date, current_price))

        if not strategy_stop:
            if current_tenor_day == 0:
                start_price = current_price
                strike_price = current_price * rel_strike
                strike_prices.append((current_date, strike_price))

            # No premium is added here, reinvestment happens only at the end of the period
            cumulative_nav.append((current_date, start_nav))
            strike_prices.append((current_date, strike_price))
            notional_units_over_time.append((current_date, notional_units))

            if current_tenor_day == tenor:
                # End of tenor, process the option
                if current_price >= strike_price:
                    # Option is exercised, add strike price + premium for the next period
                    end_nav = (
                        notional_units * start_price * rel_option_premium
                        + notional_units * start_price * rel_strike
                    )
                    conversions.append((current_date, strike_price, end_nav))
                    strategy_stop = True if stop_once_converted else False
                else:
                    # Option is not exercised, underlying value adjusts to price movement
                    end_nav = (
                        notional_units * start_price * rel_option_premium
                        + notional_units * current_price
                    )

                notional_units = end_nav / current_price
                start_nav = end_nav

                current_tenor_day = 0  # Reset tenor period

            else:
                current_tenor_day += 1
        else:
            # Strategy stops after conversion, so no further NAV changes
            cumulative_nav.append((current_date, start_nav))
            notional_units_over_time.append((current_date, notional_units))

    final_nav = cumulative_nav[-1][1]
    final_return = final_nav / df["price"].values[0] - 1  # Calculate total return
    return (
        final_nav,
        final_return,
        cumulative_nav,
        conversions,
        price_data,
        strike_prices,
        implied_vol,
        rel_option_premium,
        notional_units_over_time,
    )


def backtest_reinvesting_strategy_shifted_starts(
    df, tenor, rel_strike, call_spline, vol_shift=0.0, stop_once_converted=True
):
    total_final_nav = 0
    total_final_returns = 0
    total_final_notional_units = 0
    num_shifts = tenor
    for start_shift in range(num_shifts):
        shifted_df = df.iloc[start_shift:].reset_index(drop=True)
        (
            final_nav,
            final_return,
            _,
            _,
            _,
            _,
            implied_vol,
            rel_option_premium,
            notional_units_over_time,
        ) = backtest_reinvesting_strategy(
            shifted_df, tenor, rel_strike, call_spline, vol_shift, stop_once_converted
        )
        total_final_nav += final_nav
        total_final_returns += final_return
        total_final_notional_units += notional_units_over_time[-1][1]
    avg_final_nav = total_final_nav / num_shifts
    avg_final_return = total_final_returns / num_shifts
    avg_final_notional_units = total_final_notional_units / num_shifts
    return (
        avg_final_nav,
        avg_final_return,
        implied_vol,
        rel_option_premium,
        avg_final_notional_units,
    )


def compare_strategies(df, strike_thresholds, tenors, call_spline, vol_shift=0.0):
    results_stop = []
    results_continue = []
    buy_and_hold_final_nav = df["price"].iloc[-1]
    for tenor in tenors:
        for rel_strike in strike_thresholds:
            for stop_once_converted in [True, False]:
                (
                    avg_final_nav,
                    avg_final_return,
                    implied_vol,
                    rel_option_premium,
                    avg_final_notional_units,
                ) = backtest_reinvesting_strategy_shifted_starts(
                    df, tenor, rel_strike, call_spline, vol_shift, stop_once_converted
                )
                result_entry = {
                    "Tenor": tenor,
                    "Strike (%)": rel_strike * 100,
                    "Avg. Final Return (%)": avg_final_return * 100,
                    "Avg. Final NAV": avg_final_nav,
                    "Buy and Hold NAV": buy_and_hold_final_nav,
                    "Stop @ Conversion": stop_once_converted,
                    "IV (%)": implied_vol * 100,
                    "Option Premium (%)": rel_option_premium * 100,
                    "Avg. Notional Units": avg_final_notional_units,
                }
                if stop_once_converted:
                    results_stop.append(result_entry)
                else:
                    results_continue.append(result_entry)
    return (
        pd.DataFrame(results_stop),
        pd.DataFrame(results_continue),
        buy_and_hold_final_nav,
    )


def plot_vol_surface(res, vol_shift, filename="vol_surface.png"):
    # Create grid data for moneyness and years_to_expiry
    moneyness_grid = np.linspace(res["moneyness"].min(), res["moneyness"].max(), 100)
    years_to_expiry_grid = np.linspace(
        res["days_to_expiry"].min(), res["days_to_expiry"].max(), 100
    )
    M, Y = np.meshgrid(moneyness_grid, years_to_expiry_grid)

    # Fit the SmoothBivariateSpline to the data
    spline = SmoothBivariateSpline(
        res["moneyness"], res["days_to_expiry"], res["implied_vol"]
    )

    # Evaluate the spline on the grid
    vol_surface = spline.ev(M.flatten(), Y.flatten()).reshape(M.shape)

    # Shift the vol surface by vol_shift to create the second plane
    vol_surface_shifted = vol_surface + vol_shift

    # Plot the surface
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot original volatility surface
    surf = ax.plot_surface(
        M,
        Y,
        vol_surface,
        cmap="viridis",
        edgecolor="none",
        alpha=0.7,
        label="Original Vol Surface",
    )

    # Plot shifted volatility surface
    surf_shifted = ax.plot_surface(
        M,
        Y,
        vol_surface_shifted,
        cmap="plasma",
        edgecolor="none",
        alpha=0.6,
        label="Shifted Vol Surface",
    )

    # Add actual data points
    ax.scatter(
        res["moneyness"],
        res["days_to_expiry"],
        res["implied_vol"],
        color="r",
        marker="o",
    )

    # Labels and title
    ax.set_xlabel("Moneyness")
    ax.set_ylabel("Days to Expiry")
    ax.set_zlabel("Implied Volatility")
    ax.set_title("ETH Call Vol Surface with Shifted Plane")
    ax.view_init(elev=20, azim=30)

    # Save the plot
    plt.savefig(filename, bbox_inches="tight")
    print(f"Vol surface with shift saved to {filename}")

    return spline


def plot_best_strategy_vs_hold(
    df,
    tenor,
    rel_strike,
    call_spline,
    vol_shift=0.0,
    stop_once_converted=False,
    filename=None,
):
    (
        final_nav,
        _,
        cumulative_nav,
        conversions,
        price_data,
        strike_prices,
        _,
        _,
        notional_units_over_time,  # Include notional units over time from backtest
    ) = backtest_reinvesting_strategy(
        df, tenor, rel_strike, call_spline, vol_shift, stop_once_converted
    )

    # Split cumulative NAV data into dates and values
    nav_dates, nav_values = zip(*cumulative_nav)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 12), gridspec_kw={"height_ratios": [10, 2]}
    )

    # Plot NAV and spot price on the first subplot (ax1)
    ax1.plot(
        df["date"],
        df["price"],
        label="Spot Price",
        color="blue",
        linestyle="--",
        alpha=0.7,
    )
    ax1.plot(
        nav_dates,
        nav_values,
        label="Cumulative NAV (Call Writing)",
        color="green",
        linewidth=2,
    )
    strike_dates, strike_values = zip(*strike_prices)
    ax1.plot(
        strike_dates,
        strike_values,
        label="Strike Price",
        color="red",
        linestyle=":",
        linewidth=2,
    )
    for conv_date, strike_price, _ in conversions:
        ax1.scatter(conv_date, strike_price, color="red", marker="x", s=100)
    ax1.set_ylabel("NAV / Spot Price")
    ax1.set_title(
        f"Strategy (Tenor: {tenor} days, Strike: {rel_strike * 100}%), Stop once converted: {stop_once_converted}"
    )
    ax1.legend()
    ax1.grid(True)

    # Plot notional units over time on the second subplot (ax2)
    notional_dates, notional_units = zip(*notional_units_over_time)

    ax2.plot(
        notional_dates,
        notional_units,
        label="Notional Units",
        color="orange",
        linewidth=2,
    )
    ax2.set_ylabel("Notional Units")
    ax2.set_xlabel("Date")
    ax2.legend()
    ax2.grid(True)

    # Save the combined plot with both subplots
    title = (
        f"Best Stop @ Conversion Call Writing Strategy (Tenor: {tenor} days, Strike: {rel_strike * 100}%) vs. Buy and Hold"
        if stop_once_converted
        else f"Best Non-Stop Call Writing Strategy (Tenor: {tenor} days, Strike: {rel_strike * 100}%) vs. Buy and Hold"
    )
    plt.title(title)
    filename = (
        f"best_strategy_stop_at_conversion_{stop_once_converted}.png"
        if filename is None
        else filename
    )
    plt.savefig(
        filename,
        bbox_inches="tight",
    )


def plot_strategy_vs_spot(df, cumulative_nav, conversions, price_data, strike_prices):
    dates = df["date"].values
    spot_prices = df["price"].values
    nav_dates, nav_values = zip(*cumulative_nav)
    strike_dates, strike_values = zip(*strike_prices)
    plt.figure(figsize=(14, 8))
    plt.plot(
        dates, spot_prices, label="Spot Price", color="blue", linestyle="--", alpha=0.7
    )
    plt.plot(
        nav_dates,
        nav_values,
        label="Cumulative NAV (Call Writing)",
        color="green",
        linewidth=2,
    )
    plt.plot(
        strike_dates,
        strike_values,
        label="Strike Price",
        color="red",
        linestyle=":",
        linewidth=2,
    )
    for conv_date, strike_price, _ in conversions:
        plt.scatter(conv_date, strike_price, color="red", marker="x", s=100)
    plt.title("Strategy NAV vs. Spot Price and Strike Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)


def print_results_table(df, title):
    df_sorted = df.sort_values(by="Avg. Notional Units", ascending=False).round(3)

    # Set column headers
    headers = [
        "Tenor",
        "Strike (%)",
        "Avg. Final Return (%)",
        "Avg. Final NAV",
        "Buy and Hold NAV",
        "IV (%)",
        "Option Premium (%)",
        "Avg. Notional Units",
    ]

    # Calculate the maximum width for each column
    col_widths = [
        max(len(str(x)), len(header))
        for x, header in zip(df_sorted.max().astype(str), headers)
    ]

    # Print the title
    print(f"\n{title}")
    print(f"+{'-+-'.join('-' * w for w in col_widths)}+")

    # Print the header row
    header_row = " | ".join(
        f"{header:<{col_widths[i]}}" for i, header in enumerate(headers)
    )
    print(f"| {header_row} |")
    print(f"+{'-+-'.join('-' * w for w in col_widths)}+")

    # Print each row of the DataFrame
    for _, row in df_sorted.iterrows():
        row_data = " | ".join(
            f"{str(row[col]):<{col_widths[i]}}" for i, col in enumerate(headers)
        )
        print(f"| {row_data} |")

    # Print the bottom border of the table
    print(f"+{'-+-'.join('-' * w for w in col_widths)}+")


def get_historical_price_data(coin_id="apecoin", vs_currency="usd", days=365):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": days, "interval": "daily"}
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.drop(columns=["timestamp"], inplace=True)
    return df


def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def implied_volatility(S, K, T, r, real_price):
    def min_func(sigma, S, K, T, r, real_price):
        theor_price = black_scholes_call(S, K, T, r, sigma)
        return (theor_price - real_price) ** 2

    sigma_guesses = [0.05, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    best_result = None
    best_fun_value = float("inf")

    for sigma_0 in sigma_guesses:
        res = optimize.minimize(
            min_func,
            args=(S, K, T, r, real_price),
            x0=[sigma_0],
            bounds=[(0.01, None)],
        )

        if res["success"] and res["fun"] < best_fun_value:
            best_fun_value = res["fun"]
            best_result = res

    if best_result is not None:
        return best_result["x"][0]

    return None


def get_eth_vol_data(ccy="eth", iv_fitting_tol=0.1):
    filename = f"{ccy}_option_data.xlsx"

    def days_until_expiry(date_str):
        expiry_date = datetime.strptime(date_str, "%d%b%y").date()
        today = date.today()
        return (expiry_date - today).days

    try:
        call_df = pd.read_excel(filename)
        print(f"Data loaded from {filename}")
        return call_df
    except FileNotFoundError:
        print(f"{filename} not found. Fetching data from Deribit...")

        url = f"https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency={ccy}&kind=option"

        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            call_data = []

            for option in data["result"]:
                _, expiry, strike, option_type = option["instrument_name"].split("-")
                is_call = option_type == "C"

                S = float(option["underlying_price"])
                K = float(strike)

                option_market_price = float(option["mark_price"]) * S
                dte = days_until_expiry(expiry)
                moneyness = S / K

                mark_iv = option["mark_iv"] / 100.0
                if is_call and dte > 15:
                    T = dte / 365.0
                    sigma = implied_volatility(
                        S, K, T, RISK_FREE_RATE, option_market_price
                    )
                    iv_vol_diff = abs(sigma / mark_iv - 1)
                    if sigma and iv_vol_diff < iv_fitting_tol:
                        call_data.append(
                            {
                                "moneyness": moneyness,
                                "strike": K,
                                "days_to_expiry": dte,
                                "market_price": option_market_price,
                                "implied_vol": sigma,
                            }
                        )

            call_df = pd.DataFrame(call_data)

            call_df = call_df.groupby(
                ["moneyness", "days_to_expiry"], as_index=False
            ).mean()

            call_df.sort_values(by=["moneyness", "days_to_expiry"], inplace=True)
            call_df.to_excel(filename)
            print(f"Data saved to {filename}")
            return call_df

        except requests.RequestException as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    price_df = get_historical_price_data(
        coin_id="ethereum", vs_currency="usd", days=365
    )

    vol_shift = 0.0

    df = get_eth_vol_data()
    eth_call_spline = plot_vol_surface(df, vol_shift, filename="vol_surface.png")

    strike_thresholds = [1.0, 1.05, 1.1, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.19, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5]
    tenors = [
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        20,
        25,
        30,
        35,
        40,
        45,
        50,
        55,
        60,
        65,
        70,
        75,
        80,
        85,
        90,
    ]

    results_stop, results_continue, buy_and_hold_final_nav = compare_strategies(
        price_df, strike_thresholds, tenors, eth_call_spline, vol_shift
    )

    print_results_table(results_stop, "Strategy Results (Stop After Conversion)")
    print_results_table(results_continue, "Strategy Results (No Stop After Conversion)")

    best_strategy_stop = results_stop.loc[results_stop["Avg. Notional Units"].idxmax()]
    best_strategy_continue = results_continue.loc[
        results_continue["Avg. Notional Units"].idxmax()
    ]

    plot_heatmap_strategy_performance(
        results_stop,
        title="Strategies stop after Conversion",
        filename="strategies_plot_stop_at_conversion.png",
    )
    plot_heatmap_strategy_performance(
        results_continue,
        title="Strategies no-stop after Conversion",
        filename="strategies_plot_continue_at_conversion.png",
    )

    print("\nPlotting the best strategy (Stop After Conversion)...")
    plot_best_strategy_vs_hold(
        price_df,
        best_strategy_stop["Tenor"],
        best_strategy_stop["Strike (%)"] / 100,
        eth_call_spline,
        vol_shift,
        True,
    )

    print("\nPlotting the best strategy (No Stop After Conversion)...")
    plot_best_strategy_vs_hold(
        price_df,
        best_strategy_continue["Tenor"],
        best_strategy_continue["Strike (%)"] / 100,
        eth_call_spline,
        vol_shift,
        False,
    )
