import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.formula.api import ols
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

warnings.filterwarnings('ignore', category=RuntimeWarning)


def evaluate_data(df: pd.DataFrame, features: list, resid_col: str = 'resid'):
    """
    Проводит предварительную оценку данных:
    1. Строит box-plot для каждого признака (выявление выбросов).
    2. Строит гистограммы распределений с KDE.
    3. Строит матрицу корреляций между признаками.
    4. Рассчитывает VIF (Variance Inflation Factor) для каждого признака.
    5. Для каждого признака строит scatter plot против остатков и накладывает LOESS-кривую.
    6. Выполняет базовую линейную регрессию resid ~ features и строит ACF/PACF её остатков.

    Параметры:
    ---------
    df : pd.DataFrame
        DataFrame, содержащий колонки из списка features и колонку resid_col.
    features : list
        Список имён погодных признаков (например ['temperature','precipitation','wind_speed']).
    resid_col : str
        Название колонки с остатками из MSTL-декомпозиции (по умолчанию 'resid').
    """
    for feat in features:
        plt.figure()
        sns.boxplot(x=df[feat].dropna())
        plt.title(f'Box-plot для {feat}')
        plt.xlabel(feat)
        plt.show()
        plt.close()

    for feat in features:
        plt.figure()
        sns.histplot(df[feat].dropna(), bins=30, kde=True)
        plt.title(f'Гистограмма и KDE для {feat}')
        plt.xlabel(feat)
        plt.ylabel('Density')
        plt.show()
        plt.close()

    corr = df[features].corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Матрица корреляций')
    plt.show()
    plt.close()

    X = df[features].dropna()
    vif_values = [variance_inflation_factor(X.values, i) for i in range(len(features))]
    vif_df = pd.DataFrame({'feature': features, 'VIF': vif_values})
    print('\nVariance Inflation Factors:')
    print(vif_df)

    for feat in features:
        valid = df[[feat, resid_col]].dropna()
        if valid.shape[0] < 3:
            continue
        plt.figure()
        plt.scatter(valid[feat], valid[resid_col], alpha=0.3)
        loess_sm = lowess(valid[resid_col], valid[feat], frac=0.3)
        plt.plot(loess_sm[:, 0], loess_sm[:, 1], color='red')
        plt.title(f'{feat} vs {resid_col} (Scatter + LOESS)')
        plt.xlabel(feat)
        plt.ylabel(resid_col)
        plt.show()
        plt.close()

    formula = f"{resid_col} ~ " + ' + '.join(features)
    model = ols(formula, data=df).fit()
    resid_new = model.resid

    plt.figure()
    plot_acf(resid_new.dropna(), lags=48)
    plt.title('ACF остатков базовой регрессии')
    plt.show()
    plt.close()

    plt.figure()
    plot_pacf(resid_new.dropna(), lags=48)
    plt.title('PACF остатков базовой регрессии')
    plt.show()
    plt.close()


if __name__ == '__main__':
    df = pd.read_csv('tables/decomposed.csv', parse_dates=['date_and_time'], index_col='date_and_time')
    features = ['temperature', 'precipitation', 'wind_speed']
    evaluate_data(df, features)
