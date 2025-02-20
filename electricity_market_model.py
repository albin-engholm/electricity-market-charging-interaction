# -*- coding: utf-8 -*-
"""
Electricity Market Model with Dynamic Monthly Capacity Factors
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
# =============================================================================
# 1. Data Loading and Preprocessing
# =============================================================================


def load_data(file_path):
    """Load and preprocess the energy market data."""
    df = pd.read_excel(file_path, sheet_name="Förb + prod i Sverige", skiprows=6)
    df.rename(columns={
        "Tid": "Timestamp",
        "Total förbrukning": "Load Profile [MWh]",
        "Vindkraft": "Wind Power [MWh]",
        "Vattenkraft": "Hydro Power [MWh]",
        "Kärnkraft": "Nuclear Power [MWh]",
        "Övr.värmekraft": "Other Thermal Power [MWh]",
        "Ospec. prod.": "Unspecified Power [MWh]",
        "Solkraft": "Solar Power [MWh]",
        "Energilager": "Battery Storage [MWh]",
        "Total produktion": "Total Production [MWh]",
        "Import/export": "Net Import Export [MWh]"
    }, inplace=True)

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["Load Profile [MWh]"] = abs(df["Load Profile [MWh]"])  # Ensure positive load values

    # Descriptive DataFrame name
    energy_market_df = df[["Timestamp", "Load Profile [MWh]"]].copy()

    return energy_market_df


# Load data
file_path = "./data/2024_svk_se.xlsx"
energy_market_df = load_data(file_path)

# =============================================================================
# 2. Define Generation Assets and Apply Monthly Capacity Factors
# =============================================================================

# Define power generation assets
generation_assets = pd.DataFrame({
    "Type": ["Hydro", "Nuclear", "Wind", "Solar", "Biomass CHP", "Gas", "Coal", "Oil"],
    "Installed Capacity (MW)": [16400, 6900, 16300, 4000, 6600, 1500, 500, 200],
    "Capacity Factor (%)": [45, 85, 35, 12, 60, 30, 40, 10],
    "Marginal Cost": [5, 10, 0, 3, 20, 50, 100, 200]
})

# Apply capacity factor for initial effective capacity
generation_assets["Effective Capacity (MW)"] = generation_assets["Installed Capacity (MW)"] * \
    (generation_assets["Capacity Factor (%)"] / 100)

# Sort assets by merit order
generation_assets = generation_assets.sort_values(by="Marginal Cost").reset_index(drop=True)

# Monthly Capacity Factors for Wind, Solar, and Hydro
monthly_capacity_factors = {
    'Wind':    [0.40, 0.42, 0.38, 0.35, 0.30, 0.28, 0.25, 0.30, 0.35, 0.38, 0.40, 0.42],
    'Solar':   [0.03, 0.05, 0.15, 0.20, 0.25, 0.30, 0.28, 0.25, 0.18, 0.10, 0.05, 0.02],
    'Hydro':   [0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.75, 0.65, 0.60, 0.55, 0.50, 0.50]
}

# Apply Monthly Capacity Factors to energy_market_df


def apply_monthly_capacity_factors(df, generation_assets, monthly_capacity_factors):
    df = df.copy()
    df['Month'] = df['Timestamp'].dt.month

    for energy_type in ['Wind', 'Solar', 'Hydro']:
        installed_capacity = generation_assets.loc[generation_assets['Type']
                                                   == energy_type, 'Installed Capacity (MW)'].values[0]
        df[f'{energy_type} Power (MWh)'] = df['Month'].apply(
            lambda m: monthly_capacity_factors[energy_type][m-1]) * installed_capacity

    df.drop(columns='Month', inplace=True)
    return df


# Apply capacity factors
energy_market_df = apply_monthly_capacity_factors(energy_market_df, generation_assets, monthly_capacity_factors)

# =============================================================================
# 3. Define Helper Functions
# =============================================================================


def calculate_clearing_price(row):
    """
    Determine the market clearing price using dynamic hourly capacities.
    """
    supplied = 0
    demand = row["Total Load [MWh]"]

    # Sort by merit order (ascending marginal cost)
    merit_order = generation_assets.sort_values(by="Marginal Cost").reset_index(drop=True)

    for _, asset in merit_order.iterrows():
        # ✅ Use dynamic capacity for Wind, Solar, Hydro
        if asset["Type"] in ["Wind", "Solar", "Hydro"]:
            available_capacity = row[f"{asset['Type']} Power (MWh)"]
        else:
            # Use static effective capacity for other sources (e.g., Nuclear, Gas)
            available_capacity = asset["Effective Capacity (MW)"]

        supplied += available_capacity

        if supplied >= demand:
            return asset["Marginal Cost"]

    # If demand exceeds supply, use the highest marginal cost
    return merit_order["Marginal Cost"].max()


def calculate_operating_profit(row):
    """
    Calculate operating profit for each energy asset per hour based on dynamic capacities.
    """
    supplied = 0
    profits = {}
    demand = row["Total Load [MWh]"]

    merit_order = generation_assets.sort_values(by="Marginal Cost").reset_index(drop=True)

    for _, asset in merit_order.iterrows():
        if asset["Type"] in ["Wind", "Solar", "Hydro"]:
            available_capacity = row[f"{asset['Type']} Power (MWh)"]
        else:
            available_capacity = asset["Effective Capacity (MW)"]

        supply_from_asset = min(available_capacity, demand - supplied)
        revenue = supply_from_asset * row["Clearing Price (SEK/MWh)"]
        cost = supply_from_asset * asset["Marginal Cost"]
        profits[asset["Type"]] = revenue - cost

        supplied += supply_from_asset

        if supplied >= demand:
            break

    return profits

# =============================================================================
# 4. Simulation Engine
# =============================================================================


def run_simulation(charging_profile, energy_market_df, scenario_name="Default"):
    """
    Run electricity market simulation with dynamic capacities.
    """
    df = energy_market_df.copy()

    num_days = df['Timestamp'].dt.date.nunique()
    df['Truck Charging Demand [MWh]'] = np.tile(charging_profile, num_days)[:len(df)]
    df['Total Load [MWh]'] = df['Load Profile [MWh]'] + df['Truck Charging Demand [MWh]']

    # Apply dynamic clearing prices and profits
    df["Clearing Price (SEK/MWh)"] = df.apply(calculate_clearing_price, axis=1)
    profit_results = df.apply(calculate_operating_profit, axis=1).apply(pd.Series)
    profit_results.columns = [f"Profit ({col})" for col in profit_results.columns]
    df = pd.concat([df, profit_results], axis=1)

    average_clearing_price = df["Clearing Price (SEK/MWh)"].mean()
    total_profits = df[[col for col in df.columns if "Profit" in col]].sum()

    df['Scenario'] = scenario_name

    return {
        "scenario": scenario_name,
        "df": df,
        "average_clearing_price": average_clearing_price,
        "total_profits": total_profits
    }

# =============================================================================
# 5. Scenario Testing and Visualization
# =============================================================================


def run_scenarios(scenarios, energy_market_df):
    """Run multiple scenarios and return results."""
    results = {}
    for scenario_name, charging_profile in scenarios.items():
        print(f"Running scenario: {scenario_name}")
        res = run_simulation(charging_profile, energy_market_df, scenario_name=scenario_name)
        results[scenario_name] = res
    return results


# =============================================================================
# 6. Define and Run Scenarios
# =============================================================================
# Example charging profiles
# Base profile factors (24-hour pattern)
base_charging_profile = np.array([
    0.60, 0.60, 0.60, 0.50,
    0.45, 0.40, 0.30, 0.20,
    0.10, 0.50, 0.75, 1.00,
    0.80, 0.25, 0.10, 0.10,
    0.25, 0.25, 0.40, 0.45,
    0.50, 0.50, 0.55, 0.60
])

night_charging_profile = np.array([
    1, 1, 1, 1,
    1, 1, .5, .5,
    0, 0, 0, 0,
    0, 0, 0, 0,
    .5, .5, 1, 1,
    1, 1, 1, 1
])

annual_charging_energy_target = 5280000  # Assumed annual total charged energy(5.28 GWh)

annual_scaling_base = annual_charging_energy_target/sum(base_charging_profile*365)
annual_scaling_night = annual_charging_energy_target/sum(night_charging_profile*365)

base_charging_profile = base_charging_profile*annual_scaling_base
annual_charging_energy_base = sum(base_charging_profile*365)
print(f"Total annual charging energy base: {annual_charging_energy_base}")

night_charging_profile = night_charging_profile*annual_scaling_night
annual_charging_energy_night = sum(night_charging_profile*365)
print(f"Total annual charging energy night: {annual_charging_energy_night}")

# Compute total daily energy from the base profile
base_total_daily = base_charging_profile.sum()
average_power = base_total_daily / 24.0
uniform_charging_profile = np.full(24, average_power)


# No charging demand scenario
no_demand_profile = base_charging_profile * 0

# Assemble scenarios into a dictionary
scenarios = {
    "Base": base_charging_profile,
    "Night": night_charging_profile,
    "No charging": no_demand_profile,
    "Uniform ": uniform_charging_profile,
}


# Run Scenarios
results = run_scenarios(scenarios, energy_market_df)

# =============================================================================
# 7. Visualize Results
# =============================================================================


def visualize_results(results):
    """
    Create visualizations based on the simulation results.

    Parameters:
      - results: dictionary mapping scenario names to simulation results.
    """
    # Combine all scenario DataFrames
    df_list = [res["df"] for res in results.values()]
    combined_df = pd.concat(df_list)

    # 5.1: Average Clearing Price Across Scenarios
    plt.figure(figsize=(8, 5))
    scenario_names = list(results.keys())
    avg_prices = [results[name]["average_clearing_price"] for name in scenario_names]
    sns.barplot(x=scenario_names, y=avg_prices, palette="viridis")
    plt.xlabel("Scenario")
    plt.ylabel("Average Clearing Price (SEK/MWh)")
    plt.title("Average Clearing Price Across Scenarios")
    plt.grid(axis='y', linestyle="--", alpha=0.7)
    plt.show()

    # 5.2: Total Annual Profits per Energy Asset Across Scenarios
    profit_data = {}
    for scenario_name, res in results.items():
        profits = res["total_profits"]
        # Clean asset names by removing the "Profit (" prefix and ")" suffix
        profits.index = [name.replace("Profit (", "").replace(")", "") for name in profits.index]
        profit_data[scenario_name] = profits
    profit_df = pd.DataFrame(profit_data).T
    profit_df.plot(kind="bar", figsize=(10, 6))
    plt.xlabel("Scenario")
    plt.ylabel("Total Annual Profit (SEK)")
    plt.title("Total Annual Profits per Energy Asset Across Scenarios")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle="--", alpha=0.7)
    plt.show()

    # 5.3: Clearing Price Over Time per Scenario
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=combined_df, x="Timestamp", y="Clearing Price (SEK/MWh)",
                 hue="Scenario", palette="tab10", linewidth=0.2)
    plt.xlabel("Timestamp")
    plt.ylabel("Clearing Price (SEK/MWh)")
    plt.title("Clearing Price Over Time per Scenario")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

    # 5.4: Total Load Over Time per Scenario
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=combined_df, x="Timestamp", y="Total Load [MWh]", hue="Scenario",
                 palette="tab10", linewidth=0.2)
    plt.xlabel("Timestamp")
    plt.ylabel("Total Load (MWh)")
    plt.title("Total Load Over Time per Scenario")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

    # 5.5: Load Duration Diagram per Scenario (Total Load)
    plt.figure(figsize=(12, 6))
    for scenario_name, res in results.items():
        df = res["df"]
        # Sort total load values in descending order
        sorted_load = df["Total Load [MWh]"].sort_values(ascending=False).reset_index(drop=True)
        # Create a normalized time index (percentage of hours)
        time_fraction = (sorted_load.index + 1) / len(sorted_load) * 100
        plt.plot(time_fraction, sorted_load, label=scenario_name)
    plt.xlabel("Percentage of time load is exceeded (%)")
    plt.ylabel("Total Load (MWh)")
    plt.title("Load Duration Diagram per Scenario (Total Load)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 5.6: Average Hourly Demand Profiles per Scenario for Total Load and Truck Charging Demand
    # Create an 'Hour' column based on the Timestamp
    combined_df['Hour'] = combined_df['Timestamp'].dt.hour
    # Group by Scenario and Hour, and compute the average Total Load
    avg_total_profile = combined_df.groupby(['Scenario', 'Hour'])['Total Load [MWh]'].mean().reset_index()
    # Group by Scenario and Hour, and compute the average Truck Charging Demand
    avg_charging_profile = combined_df.groupby(['Scenario', 'Hour'])[
        'Truck Charging Demand [MWh]'].mean().reset_index()

    # Plot both profiles in subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    sns.lineplot(data=avg_total_profile, x='Hour',
                 y='Total Load [MWh]', hue='Scenario', palette="tab10", marker="o", ax=axes[0])
    axes[0].set_title("Average Hourly Total Load Profile per Scenario")
    axes[0].set_ylabel("Total Load (MWh)")
    axes[0].set_xticks(range(0, 24))
    axes[0].grid(True)

    sns.lineplot(data=avg_charging_profile, x='Hour',
                 y='Truck Charging Demand [MWh]', hue='Scenario', palette="tab10", marker="o", ax=axes[1])
    axes[1].set_title("Average Hourly Truck Charging Demand Profile per Scenario")
    axes[1].set_xlabel("Hour of the Day")
    axes[1].set_ylabel("Truck Charging Demand (MWh)")
    axes[1].set_xticks(range(0, 24))
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    # 5.7: Clearing Price Over Time per Scenario
    plt.figure(figsize=(12, 6))
    sns.displot(data=combined_df,  x="Clearing Price (SEK/MWh)",
                hue="Scenario",  palette="tab10", kind="kde")
    plt.xlabel("Clearing Price (SEK/MWh)")
    plt.title("Clearing price distribution per Scenario")
    plt.grid(True)
    plt.show()

    # 5.8: Total Load Over Time per Scenario
    plt.figure(figsize=(12, 6))
    sns.displot(data=combined_df, x="Total Load [MWh]", hue="Scenario",
                palette="tab10", linewidth=1, kind="kde")
    plt.xlabel("Total Load (MWh)")
    plt.title("Total Load Histogram per Scenario")
    plt.grid(True)
    plt.show()


visualize_results(results)
