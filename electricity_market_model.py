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
    "Installed Capacity (MW)": [16400, 6900, 16300, 4000, 6600, 1500, 500, 200],  # Based on SVK statistics for 2024
    # Average capacity factors, only used for non-volatile sources
    "Capacity Factor (%)": [45, 85, 35, 12, 60, 30, 40, 10],
    "Marginal Cost": [15, 90, 5, 5, 150, 900, 950, 1800]  # Guesstimates on average Marginal Costs
})
generation_assets["CO2 Tax (SEK/MWh)"] = generation_assets["Type"].map({
    "Gas": 540, "Coal": 1140, "Oil": 1200
}).fillna(0)

# Add CO₂ tax to marginal cost
generation_assets["Marginal Cost"] += generation_assets["CO2 Tax (SEK/MWh)"]
# Apply capacity factor for initial effective capacity
generation_assets["Effective Capacity (MW)"] = generation_assets["Installed Capacity (MW)"] * \
    (generation_assets["Capacity Factor (%)"] / 100)

# Sort assets by merit order
generation_assets = generation_assets.sort_values(by="Marginal Cost").reset_index(drop=True)

# Monthly Capacity Factors for Wind, Solar, and Hydro, plus some manual calibration to get "reasonable" energy mix in output matching Sweden 2024
monthly_capacity_factors = {
    'Wind':  [x * 0.7 for x in [0.40, 0.42, 0.38, 0.35, 0.30, 0.28, 0.25, 0.30, 0.35, 0.38, 0.40, 0.42]],
    'Solar': [x * .5 for x in [0.03, 0.05, 0.15, 0.20, 0.25, 0.30, 0.28, 0.25, 0.18, 0.10, 0.05, 0.02]],
    'Hydro': [x * 0.65 for x in [0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.75, 0.65, 0.60, 0.55, 0.50, 0.50]]

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
    Calculate operating profit per energy asset per hour and record dispatched energy.
    """
    supplied = 0
    profits = {}
    dispatched_energy = {}
    demand = row["Total Load [MWh]"]

    merit_order = generation_assets.sort_values(by="Marginal Cost").reset_index(drop=True)

    for _, asset in merit_order.iterrows():
        if asset["Type"] in ["Wind", "Solar", "Hydro"]:
            available_capacity = row[f"{asset['Type']} Power (MWh)"]
        else:
            available_capacity = asset["Effective Capacity (MW)"]

        # Dispatch up to available capacity or remaining demand
        supply_from_asset = min(available_capacity, demand - supplied)

        # ✅ Prefix "Profit (" to match later DataFrame operations
        profit_key = f"Profit ({asset['Type']})"
        dispatched_key = f"Dispatched ({asset['Type']})"

        # Calculate profit
        revenue = supply_from_asset * row["Clearing Price (SEK/MWh)"]
        cost = supply_from_asset * asset["Marginal Cost"]
        profits[profit_key] = revenue - cost

        # ✅ Record dispatched energy
        dispatched_energy[dispatched_key] = supply_from_asset

        supplied += supply_from_asset

        if supplied >= demand:
            break

    # Combine profits and dispatched energy
    combined_result = {**profits, **dispatched_energy}
    return combined_result


# =============================================================================
# 4. Simulation Engine
# =============================================================================


def run_simulation(charging_profile, energy_market_df, scenario_name="Default"):
    """
    Run electricity market simulation and calculate energy mix.
    """
    df = energy_market_df.copy()

    num_days = df['Timestamp'].dt.date.nunique()
    df['Truck Charging Demand [MWh]'] = np.tile(charging_profile, num_days)[:len(df)]
    df['Total Load [MWh]'] = df['Load Profile [MWh]'] + df['Truck Charging Demand [MWh]']

    # Apply clearing prices and profits
    df["Clearing Price (SEK/MWh)"] = df.apply(calculate_clearing_price, axis=1)
    profit_results = df.apply(calculate_operating_profit, axis=1).apply(pd.Series)
    df = pd.concat([df, profit_results], axis=1)

    # ✅ Calculate average clearing price
    average_clearing_price = df["Clearing Price (SEK/MWh)"].mean()

    # ✅ Correct profit column filtering
    profit_cols = [col for col in df.columns if col.startswith("Profit")]
    total_profits = df[profit_cols].sum()

    # ✅ Calculate total dispatched energy for energy mix
    dispatched_cols = [col for col in df.columns if col.startswith("Dispatched")]
    total_dispatched = df[dispatched_cols].sum()
    total_energy = total_dispatched.sum()

    # Calculate energy mix as percentages
    energy_mix = (total_dispatched / total_energy) * 100

    df['Scenario'] = scenario_name

    return {
        "scenario": scenario_name,
        "df": df,
        "average_clearing_price": average_clearing_price,
        "total_profits": total_profits,
        "energy_mix": energy_mix
    }


# =============================================================================
# 5. Scenario Testing and Visualization
# =============================================================================


def run_scenarios(scenarios, energy_market_df):
    """Run multiple scenarios and return results, including energy mix."""
    results = {}
    for scenario_name, charging_profile in scenarios.items():
        print(f"Running scenario: {scenario_name}")
        res = run_simulation(charging_profile, energy_market_df, scenario_name=scenario_name)
        results[scenario_name] = res

        # ✅ Display Energy Mix
        print(f"\nEnergy Mix for {scenario_name}:")
        print(res["energy_mix"].round(2).astype(str) + " %")
        print("\n")

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

extreme_case_profile = np.array([
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    1, 0, 0, 0,
    0, 0, 0, 0
])


annual_charging_energy_target = 5280000  # Assumed annual total charged energy(5.28 GWh)

annual_scaling_base = annual_charging_energy_target/sum(base_charging_profile*365)
annual_scaling_night = annual_charging_energy_target/sum(night_charging_profile*365)
annual_scaling_extreme = annual_charging_energy_target/sum(extreme_case_profile*365)

base_charging_profile = base_charging_profile*annual_scaling_base
annual_charging_energy_base = sum(base_charging_profile*365)
print(f"Total annual charging energy base: {annual_charging_energy_base}")

night_charging_profile = night_charging_profile*annual_scaling_night
annual_charging_energy_night = sum(night_charging_profile*365)
print(f"Total annual charging energy night: {annual_charging_energy_night}")

extreme_case_profile = extreme_case_profile*annual_scaling_extreme
annual_charging_energy_extreme = sum(extreme_case_profile*365)
print(f"Total annual charging energy extreme: {annual_charging_energy_extreme}")


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
    "Extreme": extreme_case_profile
}


# Run Scenarios
results = run_scenarios(scenarios, energy_market_df)
#%% 
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

    # ================================
    # 1. Average Clearing Price Across Scenarios
    # ================================
    plt.figure(figsize=(8, 5))
    scenario_names = list(results.keys())
    avg_prices = [results[name]["average_clearing_price"] for name in scenario_names]
    sns.barplot(x=scenario_names, y=avg_prices, hue=scenario_names, palette="viridis", legend=False)
    plt.xlabel("Scenario")
    plt.ylabel("Average Clearing Price (SEK/MWh)")
    plt.title("Average Clearing Price Across Scenarios")
    plt.grid(axis='y', linestyle="--", alpha=0.7)
    plt.show()

    # ================================
    # 2. Total Annual Profits per Energy Asset Across Scenarios
    # ================================

    # Prepare profit data
    profit_data = {}
    for scenario_name, res in results.items():
        profits = res.get("total_profits", pd.Series())

        # ✅ Clean asset names by removing "Profit (" prefix and ")" suffix
        if not profits.empty:
            profits.index = [name.replace("Profit (", "").replace(")", "") for name in profits.index]
            profit_data[scenario_name] = profits
        else:
            print(f"⚠️ No profit data found for scenario: {scenario_name}")

    # Convert to DataFrame
    profit_df = pd.DataFrame(profit_data).T

    # ✅ Convert non-numeric to NaN and handle missing values
    profit_df = profit_df.apply(pd.to_numeric, errors='coerce')

    # ✅ Debug: Print profit_df if the error persists
    print("Profit DataFrame for Plotting:")
    print(profit_df.head())

    # ✅ Check for empty or non-numeric data
    if profit_df.dropna(axis=1, how='all').empty:
        print("⚠️ No numeric data to plot for profits.")
    else:
        # Plot if valid data exists
        profit_df.plot(kind="bar", figsize=(10, 6))
        plt.xlabel("Scenario")
        plt.ylabel("Total Annual Profit (SEK)")
        plt.title("Total Annual Profits per Energy Asset Across Scenarios")
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle="--", alpha=0.7)
        plt.show()

    # ================================
    # 3. Clearing Price Over Time per Scenario
    # ================================
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=combined_df, x="Timestamp", y="Clearing Price (SEK/MWh)",
                 hue="Scenario", palette="tab10", linewidth=0.5)
    plt.xlabel("Timestamp")
    plt.ylabel("Clearing Price (SEK/MWh)")
    plt.title("Clearing Price Over Time per Scenario")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

    # ================================
    # 4. Total Load Over Time per Scenario
    # ================================
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=combined_df, x="Timestamp", y="Total Load [MWh]", hue="Scenario",
                 palette="tab10", linewidth=0.5)
    plt.xlabel("Timestamp")
    plt.ylabel("Total Load (MWh)")
    plt.title("Total Load Over Time per Scenario")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

    # ================================
    # 5. Load Duration Diagram per Scenario (Total Load)
    # ================================
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

    # ================================
    # 6. Average Hourly Demand Profiles per Scenario
    # ================================
    combined_df['Hour'] = combined_df['Timestamp'].dt.hour

    # Group by Scenario and Hour
    avg_total_profile = combined_df.groupby(['Scenario', 'Hour'])['Total Load [MWh]'].mean().reset_index()
    avg_charging_profile = combined_df.groupby(['Scenario', 'Hour'])[
        'Truck Charging Demand [MWh]'].mean().reset_index()

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    sns.lineplot(data=avg_total_profile, x='Hour', y='Total Load [MWh]', hue='Scenario',
                 palette="tab10", marker="o", ax=axes[0])
    axes[0].set_title("Average Hourly Total Load Profile per Scenario")
    axes[0].set_ylabel("Total Load (MWh)")
    axes[0].set_xticks(range(0, 24))
    axes[0].grid(True)

    sns.lineplot(data=avg_charging_profile, x='Hour', y='Truck Charging Demand [MWh]', hue='Scenario',
                 palette="tab10", marker="o", ax=axes[1])
    axes[1].set_title("Average Hourly Truck Charging Demand Profile per Scenario")
    axes[1].set_xlabel("Hour of the Day")
    axes[1].set_ylabel("Truck Charging Demand (MWh)")
    axes[1].set_xticks(range(0, 24))
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    # ================================
    # 7. Clearing Price Distribution per Scenario
    # ================================
    plt.figure(figsize=(12, 6))
    sns.displot(data=combined_df, x="Clearing Price (SEK/MWh)", hue="Scenario",
                palette="tab10", kind="kde")
    plt.xlabel("Clearing Price (SEK/MWh)")
    plt.title("Clearing Price Distribution per Scenario")
    plt.grid(True)
    plt.show()

    # ================================
    # 8. Total Load Histogram per Scenario
    # ================================
    plt.figure(figsize=(12, 6))
    sns.displot(data=combined_df, x="Total Load [MWh]", hue="Scenario",
                palette="tab10", kind="kde")
    plt.xlabel("Total Load (MWh)")
    plt.title("Total Load Histogram per Scenario")
    plt.grid(True)
    plt.show()

    # ================================
    # 9. Energy Mix per Scenario (New)
    # ================================
    energy_mix_data = {}
    for scenario_name, res in results.items():
        energy_mix = res["energy_mix"]
        energy_mix.index = [name.replace("Dispatched (", "").replace(")", "") for name in energy_mix.index]
        energy_mix_data[scenario_name] = energy_mix

    energy_mix_df = pd.DataFrame(energy_mix_data).T

    # Plot Energy Mix as stacked bar chart
    energy_mix_df.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='tab20')
    plt.xlabel("Scenario")
    plt.ylabel("Percentage of Total Energy (%)")
    plt.title("Energy Mix Per Scenario")
    plt.legend(title='Energy Source', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # ================================
    # 10. Load duration for charging
    # ================================

    plt.figure(figsize=(12, 6))
    for scenario_name, res in results.items():
        df = res["df"]

        # Sort charging demand in descending order
        sorted_charging = df["Truck Charging Demand [MWh]"].sort_values(ascending=False).reset_index(drop=True)

        # Create a normalized time index (percentage of hours)
        time_fraction = (sorted_charging.index + 1) / len(sorted_charging) * 100

        plt.plot(time_fraction, sorted_charging, label=scenario_name)

    plt.xlabel("Percentage of time charging demand is exceeded (%)")
    plt.ylabel("Truck Charging Demand (MWh)")
    plt.title("Load Duration Diagram for Charging Demand")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ================================
    # 11. boxplot clearing price
    # ================================

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=combined_df, y="Clearing Price (SEK/MWh)", hue="Scenario",
                x="Scenario",
                palette="tab10")
    plt.xlabel("Clearing Price (SEK/MWh)")
    plt.title("Clearing Price Distribution per Scenario")
    plt.grid(True)
    plt.show()
    

visualize_results(results)
