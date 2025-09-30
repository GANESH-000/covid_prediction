# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from prophet import Prophet
from sklearn.metrics import r2_score

# Plotting style
plt.style.use("ggplot")

# Load datasets
df0 = pd.read_csv("CONVENIENT_global_confirmed_cases.csv")
df1 = pd.read_csv("CONVENIENT_global_deaths.csv")
continent = pd.read_csv("continents2.csv")


# Basic Information
print(df0.info())
print(df1.info())
print(df0.describe())
print(df1.describe())

# Preprocess World Cases
world = pd.DataFrame({"Country": df0.iloc[:, 1:].columns})
cases = [pd.to_numeric(df0[country][1:]).sum() for country in world["Country"]]
world["Cases"] = cases

# Clean Country Names
country_list = []
for country in world["Country"]:
    temp = ""
    for ch in country:
        if ch in [".", "("]:
            break
        temp += ch
    country_list.append(temp.strip())
world["Country"] = country_list

# Group by Country
world = world.groupby("Country")["Cases"].sum().reset_index()

# Prepare for Choropleth Map
continent["name"] = continent["name"].str.upper()
world["Cases Range"] = pd.cut(world["Cases"], [-150000, 50000, 200000, 800000, 1500000, 15000000],
                              labels=["U50K", "50Kto200K", "200Kto800K", "800Kto1.5M", "1.5M+"])

alpha = []
for i in world["Country"].str.upper().values:
    if i == "BRUNEI":
        i = "BRUNEI DARUSSALAM"
    elif i == "US":
        i = "UNITED STATES"
    alpha_code = continent[continent["name"] == i]["alpha-3"].values
    alpha.append(alpha_code[0] if len(alpha_code) > 0 else np.nan)
world["Alpha3"] = alpha

# Plot Choropleth Map
fig = px.choropleth(world.dropna(), 
                    locations="Alpha3",
                    color="Cases Range",
                    projection="mercator",
                    color_discrete_sequence=["white", "khaki", "yellow", "orange", "red"])
fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
fig.show()

# Aggregate Global Daily Data
count_cases = [sum(pd.to_numeric(df0.iloc[i, 1:].values)) for i in range(1, len(df0))]
count_deaths = [sum(pd.to_numeric(df1.iloc[i, 1:].values)) for i in range(1, len(df1))]

df = pd.DataFrame()
df["Date"] = df0["Country/Region"][1:]
df["Cases"] = count_cases
df["Deaths"] = count_deaths
df = df.set_index("Date")

# Plot Daily Cases
df.Cases.plot(title="Daily Covid19 Cases in World", marker=".", figsize=(10, 5), label="Daily Cases")
df.Cases.rolling(window=5).mean().plot(figsize=(10, 5), label="MA5")
plt.ylabel("Cases")
plt.legend()
plt.show()

# Plot Daily Deaths
df.Deaths.plot(title="Daily Covid19 Deaths in World", marker=".", figsize=(10, 5), label="Daily Deaths")
df.Deaths.rolling(window=5).mean().plot(figsize=(10, 5), label="MA5")
plt.ylabel("Deaths")
plt.legend()
plt.show()

# Fbprophet Model Class
class Fbprophet(object):
    def fit(self, data):
        self.data = data
        self.model = Prophet(weekly_seasonality=True, daily_seasonality=False, yearly_seasonality=False)
        self.model.fit(self.data)

    def forecast(self, periods, freq):
        self.future = self.model.make_future_dataframe(periods=periods, freq=freq)
        self.df_forecast = self.model.predict(self.future)

    def plot(self, xlabel="Years", ylabel="Values"):
        self.model.plot(self.df_forecast, xlabel=xlabel, ylabel=ylabel, figsize=(9, 4))
        self.model.plot_components(self.df_forecast, figsize=(9, 6))

    def R2(self):
        return r2_score(self.data.y, self.df_forecast.yhat[:len(self.data)])

# Prepare data for Prophet
df_fb = pd.DataFrame({"ds": pd.to_datetime(df.index), "y": df["Cases"].values})

# Train and Forecast
model = Fbprophet()
model.fit(df_fb)
model.forecast(30, "D")
print("R2 Score:", model.R2())

# Plot Forecast
forecast = model.df_forecast[["ds", "yhat_lower", "yhat_upper", "yhat"]].tail(30).reset_index().set_index("ds").drop("index", axis=1)
forecast["yhat"].plot(marker=".", figsize=(10, 5))
plt.fill_between(x=forecast.index, y1=forecast["yhat_lower"], y2=forecast["yhat_upper"], color="gray")
plt.legend(["Forecast", "Bound"], loc="upper left")
plt.title("Forecasting of Next 30 Days Cases")
plt.show()
