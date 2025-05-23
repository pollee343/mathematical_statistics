import numpy as np

setattr(np, 'int', int)
import scipy.sparse as sp

setattr(sp.csr_matrix, 'A', property(lambda self: self.toarray()))

import pandas as pd
from pygam import LinearGAM, s
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt


def fit_gamm(df: pd.DataFrame,
             resid_col: str = 'resid',
             features: list = ['temperature', 'precip_log', 'wind_speed'],
             splines: list = [10, 8, 8]):
    """
    Фиттит GAM для остатков с кубическими сплайнами, затем SARIMAX для AR(2).
    Возвращает объект GAM, SARIMAX и DataFrame с предсказаниями.
    """
    X = np.vstack([
        df['temperature'],
        df['precip_log'],
        df['wind_speed']
    ]).T
    y = df[resid_col].values

    gam = LinearGAM(
        s(0, n_splines=splines[0]) +
        s(1, n_splines=splines[1]) +
        s(2, n_splines=splines[2])
    ).fit(X, y)

    gam_resid = y - gam.predict(X)

    sar = SARIMAX(gam_resid, order=(2, 0, 0),
                  enforce_stationarity=False,
                  enforce_invertibility=False).fit(disp=False)

    df_result = df.copy()
    df_result['gamm_pred'] = gam.predict(X)
    df_result['ar_resid_pred'] = sar.predict()
    df_result['gammm_pred'] = df_result['gamm_pred'] + df_result['ar_resid_pred']

    return gam, sar, df_result


def plot_partial_effects(gam: LinearGAM, feature_names: list = ['temperature', 'log_precipitation', 'wind_speed']):
    """
    Строит частичные зависимости для каждого сплайнового терма GAM.
    """
    for i, name in enumerate(feature_names):
        XX = gam.generate_X_grid(term=i)
        plt.figure()
        plt.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))
        plt.title(f'Partial Dependence: {name}')
        plt.xlabel(name)
        plt.ylabel('Effect on resid')
        plt.show()
        plt.close()


def evaluate_model(df_result: pd.DataFrame, true_col: str = 'resid'):
    """
    Оценивает MAE, RMSE и строит ACF остатка после полной модели.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from statsmodels.graphics.tsaplots import plot_acf

    y_true = df_result[true_col]
    y_pred = df_result['gammm_pred']

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f'Model Quality:\n  MAE: {mae:.3f}\n  RMSE: {rmse:.3f}')

    final_resid = y_true - y_pred
    plt.figure()
    plot_acf(final_resid.dropna(), lags=48)
    plt.title('ACF of Final Residuals')
    plt.show()
    plt.close()


def summarize_results(gam: LinearGAM, sar, df_result: pd.DataFrame, features: list):
    """
    Выводит подробный summary: параметры GAM, AR(2) и оценку влияния факторов.
    """
    import warnings
    warnings.filterwarnings('ignore',
                            message="KNOWN BUG: p-values computed in this summary are likely much smaller than they should be.")
    print("=== GAM Summary ===")
    print(gam.summary())
    warnings.resetwarnings()

    print("\n=== AR(2) Parameters ===")
    print(sar.summary().tables[1])

    print("=== Estimated Marginal Effects ===")
    grid_temp = np.linspace(df_result['temperature'].min(), df_result['temperature'].max(), 5)
    print("Temperature effect at points:")
    for val in grid_temp:
        eff = gam.partial_dependence(term=0, X=np.array(
            [[val, df_result['precip_log'].mean(), df_result['wind_speed'].mean()]]))
        print(f"  {val:.2f} -> {eff[0]:.3f}")

    grid_prec = np.linspace(df_result['precip_log'].min(), df_result['precip_log'].max(), 5)
    print("Precipitation effect at points:")
    for val in grid_prec:
        eff = gam.partial_dependence(term=1, X=np.array(
            [[df_result['temperature'].mean(), val, df_result['wind_speed'].mean()]]))
        print(f"  {val:.2f} -> {eff[0]:.3f}")

    grid_wind = np.linspace(df_result['wind_speed'].min(), df_result['wind_speed'].max(), 5)
    print("Wind effect at points:")
    for val in grid_wind:
        eff = gam.partial_dependence(term=2, X=np.array(
            [[df_result['temperature'].mean(), df_result['precip_log'].mean(), val]]))
        print(f"  {val:.2f} -> {eff[0]:.3f}")
