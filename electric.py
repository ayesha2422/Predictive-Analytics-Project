import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def train_energy_model():
    np.random.seed(42)
    data_size = 500

    solar = np.random.uniform(10, 80, data_size)
    wind = np.random.uniform(5, 60, data_size)
    demand = np.random.uniform(50, 120, data_size)

    optimal_energy = 0.6 * solar + 0.4 * wind

    df = pd.DataFrame({
        'Solar': solar,
        'Wind': wind,
        'Demand': demand,
        'OptimalEnergy': optimal_energy
    })

    X = df[['Solar', 'Wind', 'Demand']]
    y = df['OptimalEnergy']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    return model, mse
