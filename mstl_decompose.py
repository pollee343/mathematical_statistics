import pandas as pd
from statsmodels.tsa.seasonal import MSTL


def mstl_decompose(
        df: pd.DataFrame,
        target_col: str,
        periods: list = [24, 24 * 7]
) -> pd.DataFrame:
    """
    Выполняет множественную STL-декомпозицию (MSTL) временного ряда.

    Параметры:
    ---------
    df : pd.DataFrame
        Исходный DataFrame, индекс должен быть типа DatetimeIndex с равномерным шагом (например, почасовым).
    target_col : str
        Название столбца с целевым рядом (число поездок).
    periods : list, optional
        Список периодов сезонности. По умолчанию [24, 168] (сутки и неделя).

    Возвращает:
    ---------
    pd.DataFrame
        DataFrame, содержащий исходный ряд, тренд, сезонные компоненты и остатки:
        - trend
        - seasonal_<period> для каждого из periods
        - resid
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Индекс DataFrame должен быть DatetimeIndex.")

    mstl = MSTL(df[target_col], periods=periods)
    res = mstl.fit()

    output = pd.DataFrame(index=df.index)
    output[target_col] = df[target_col]
    output['trend'] = res.trend

    seasonal = res.seasonal
    if hasattr(seasonal, 'shape') and seasonal.ndim == 2:
        for idx, p in enumerate(periods):
            output[f'seasonal_{p}'] = seasonal.iloc[:, idx]
    else:
        output[f'seasonal_{periods[0]}'] = seasonal

    output['resid'] = res.resid

    return output
