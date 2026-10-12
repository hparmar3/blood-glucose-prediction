#hparmar3:11am:20260422:project.py
import pandas as pd

#Q1
"""
DATA IMPORT
"""
# Import Dataset
dex = pd.read_csv("Clarity_Export_Parmar_Henna_2026-04-13_013311.csv", header=0, skiprows=range(1, 21))
dex.head()




"""
ESTIMATING CARBS AND INSULIN DOSAGE & DATA CLEANING

This section of code estimates carbs and insulin dosage for data analysis. Most
of the time I don't track the amount of carbs I eat or insulin I give, so based
on the rate of change of my blood sugars and my "insulin-to-carb-ratio", 
I estimated each amount. I adjusted the times until it looked like reasonable 
numbers and times based on what I usually eat.

I also cleaned some data and got rid of columns I wasn't using.

This was adapated from a pervious project.
"""
# Estimate Carbs and Insulin
import numpy as np

# Filter the Columns for EGV events and include 'Insulin Value (u)'
glucose_df = dex[dex['Event Type'] == "EGV"].copy()
glucose_df = glucose_df[['Timestamp (YYYY-MM-DDThh:mm:ss)', 'Glucose Value (mg/dL)']]
glucose_df.columns = ['Timestamp', 'Glucose']

# Convert 'Glucose' and 'Insulin Value (u)' to numeric, coercing errors
glucose_df['Timestamp'] = pd.to_datetime(glucose_df['Timestamp'])
glucose_df['Glucose'] = pd.to_numeric(glucose_df['Glucose'], errors='coerce')

# Compute Rate of Change and Rolling Differences
glucose_df['delta'] = glucose_df['Glucose'].diff()
glucose_df['delta_30min'] = glucose_df['Glucose'].diff(6) # 6 readings ~ 30 min (5 min intervals)
glucose_df['roc'] = glucose_df['delta'] / 5 # rate of change mg/dL per minute

# Detect Meal Events (Rapid Rise)
glucose_df['meal_event'] = (glucose_df['delta_30min'] > 30).astype(int)
glucose_df['carb_est'] = 0.0
glucose_df.loc[glucose_df['meal_event'] == 1, 'carb_est'] = (glucose_df['delta_30min'] * 1.2)

# Detect Insulin Events (Rapid Drop)
glucose_df['insulin_event'] = (glucose_df['delta_30min'] < -30).astype(int)
glucose_df['insulin_est_units'] = 0.0
glucose_df.loc[glucose_df['insulin_event'] == 1, 'insulin_est_units'] = (np.abs(glucose_df['delta_30min']) / 40)


def consolidate_daily_events(df, carb_col='carb_est', insulin_col='insulin_est_units',
                             timestamp_col='Timestamp', max_events_per_day=10, merge_window_min=90,
                             carb_cap = 75):
    """
    Consolidate small insulin/carb events into <= max_events_per_day realistic events per day.
    Events within merge_window_min minutes are merged together.
    """

    df = df.copy()

    # Extract Date Only for grouping
    df['Date'] = df[timestamp_col].dt.date

    consolidated = []
    
    for date, group in df.groupby('Date'):
        g = group.sort_values(timestamp_col)

        # Create Event Markers Where Insulin or Carbs > 0
        events = g[(g[carb_col] > 0) | (g[insulin_col] > 0)].copy()
        if len(events) == 0:
            continue

        # Merge Close Events within merge_window_min
        current = events.iloc[0].copy()
        merged_events = []
        
        for _, row in events.iloc[1:].iterrows():
            delta_min = (
                (row[timestamp_col] - current[timestamp_col]).total_seconds() / 60
                )
            
            if delta_min <= merge_window_min:
                current[carb_col] = min(
                    max(current[carb_col], row[carb_col]),
                    carb_cap
                    )
                
                current[insulin_col] += row[insulin_col]
                
                # keep latest timestamp (optional but usually better)
                current[timestamp_col] = row[timestamp_col]
                
            else:
                merged_events.append(current)
                current = row.copy()
                
        merged_events.append(current)
        merged_df = pd.DataFrame(merged_events)
        
        # Keep only top N events by carb+insulin total
        merged_df['total_effect'] = (merged_df[carb_col] + 20 * merged_df[insulin_col])
        merged_df = (
            merged_df
            .sort_values('total_effect', ascending=False)
            .head(max_events_per_day)
            .sort_values(timestamp_col)
            )

        consolidated.append(merged_df)
        
    if not consolidated:
        return df

    consolidated_df = pd.concat(consolidated, ignore_index=True)

    # Map Consolidated Values Back to Main Dataframe
    df['Carb Value'] = 0.0
    df['Insulin Value'] = 0.0

    for _, row in consolidated_df.iterrows():
        t = row[timestamp_col]
        idx = (df[timestamp_col] - t).abs().idxmin()
        df.loc[idx, 'Carb Value'] = row[carb_col]
        df.loc[idx, 'Insulin Value'] = row[insulin_col]

    return df


# Apply Consolidation
glucose_df = consolidate_daily_events(glucose_df)


glucose_df = glucose_df.fillna(0)
# Glucose needs to be above 0 else I dead
glu = glucose_df["Glucose"] > 0
glucose_df = glucose_df[glu]

# Save cleaned data file
glucose_df.to_csv('dexcom_with_estimates_cleaned.csv', index=False)




#Q2
"""
ML MODEL (NUMERICAL ANALYSIS OF THE DATA)
I based the analysis off of this paper: https://ieeexplore.ieee.org/document/9666882
The extracted was from the same medical device I use.
"""
time = glucose_df['Timestamp']
glucose = glucose_df["Glucose"]
carbs = glucose_df["Carb Value"]
insulin = glucose_df["Insulin Value"]

# Calculate means, bc I am curious about my stats
glucose_mean = glucose.mean()
print("Glucose Mean:", glucose_mean)
carbs_select = carbs > 0
carbs_mean = carbs[carbs_select].mean()
print("Carbs Mean:", carbs_mean)
insulin_select = insulin > 0
insulin_mean = insulin[insulin_select].mean()
print("Insulin Mean:", insulin_mean)

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

### Hammerstein Box-Jenkins Model
def hammerstein_bj(y, u, na=2, nb=2, M=2):
    """
    Parameters
    ----------
    u : input (insulin)
    y : output (glucose)
    na : AR order
    nb : input lag order
    M : nonlinearity degree
    Returns
    -------
    None.
    """
    y = np.array(y).flatten()
    u = np.array(u).flatten()
    
    X = []
    Y = []
    
    for t in range(max(na, nb), len(y)):
        row = []
         
        for i in range(1, na+1):
            row.append(float(y[t-i]))
            
        for j in range(nb+1):
            for k in range(M+1):
                row.append(float(u[t-j]**k))
                
        X.append(row)
        Y.append(float(y[t]))
        
    return np.array(X), np.array(Y)

# Create features and targets
x, y = hammerstein_bj(glucose, insulin)

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Imputation (Estimator + Transformer)
imputer = SimpleImputer(strategy="constant", fill_value=0)
x_train_imputed = imputer.fit_transform(x_train)
x_test_imputed = imputer.transform(x_test)
imputer = SimpleImputer(strategy="constant", fill_value=0)

# Scaling (Estimator + Transformer)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_imputed)
x_test_scaled = scaler.transform(x_test_imputed)

# Model training (Predictor)
model = LinearRegression()
model.fit(x_train_scaled, y_train)

# Prediction
pred = model.predict(x_test_scaled)
print("Predicted score:", pred)

# RMSE evaluation (regular function)
def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)**0.5

print("RMSE:", rmse(y_test, pred))



#Q3
"""
PLOTS
"""
import matplotlib.pyplot as plt
# Predicted vs Actual :)
plt.plot(y_test, pred, 'o', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)],
        color='red', linewidth=2, label="Prediction")
plt.title("Predicted vs Actual Glucose", fontsize=16)
plt.xlabel("Actual Glucose (mg/dL", fontsize=12)
plt.ylabel("Predicted Glucose (mg/dL)", fontsize=12)
plt.legend()
plt.show()

# Since the split is random, plotting with time could be interesting...
_, x_test_idx, _, y_test_idx = train_test_split(np.arange(len(x)), y, test_size=0.2, random_state=1)
time_test = time.iloc[x_test_idx]

plt.plot(time_test, y_test, 'o', label="Actual")
plt.plot(time_test, pred, 'o', label="Predicted")
plt.title("Glucose: Actual vs Predicted Over Time", fontsize=16)
plt.xlabel("Time", fontsize=12)
plt.ylabel("Glucose (mg/dL)", fontsize=12)
plt.legend()
plt.xticks(rotation=45)
plt.show()

# Glucose + Events
plt.plot(time, glucose, color='blue', linewidth=2, label="Glucose")
plt.scatter(time, carbs, color='orange', label='Carbs (g)')
plt.scatter(time, insulin, color='green', label='Insulin (units)')
plt.title("Glucose, Carb Intake, and Insulin Over Time", fontsize=16)
plt.xlabel("Time", fontsize=12)
plt.ylabel("Value", fontsize=12)
plt.legend()
plt.xticks(rotation=45)
plt.show()



#Q4
"""
WRITE RESULTS TO A TEST FILE
"""
results = pd.DataFrame({
    "Metric": ["RMSE"],
    "Value": [rmse(y_test, pred)]
    })

results.to_csv("model_results.csv", index=False)

