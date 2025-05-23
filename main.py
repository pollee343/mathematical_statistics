import numpy as np
import pandas as pd
from mstl_decompose import mstl_decompose
from data_quality_assessment import evaluate_data
from model import fit_gamm, plot_partial_effects, evaluate_model, summarize_results


def main():
    df = pd.read_csv(
        'tables/final.csv',
        sep=';',
        decimal=',',
        usecols=['date_and_time',
                 'number_of_taxi_trips',
                 'temperature',
                 'precipitation',
                 'wind_speed']
    )
    df['date_and_time'] = pd.to_datetime(
        df['date_and_time'], format='%m/%d/%Y %I:%M:%S %p', errors='raise'
    )
    df.set_index('date_and_time', inplace=True)
    df = df.asfreq('h')

    decomposed = mstl_decompose(df, target_col='number_of_taxi_trips')

    combined = decomposed.join(
        df[['temperature', 'precipitation', 'wind_speed']]
    )

    features = ['temperature', 'precipitation', 'wind_speed']
    evaluate_data(combined, features)

    combined['precip_log'] = np.log1p(combined['precipitation'])
    gam, sar, df_model = fit_gamm(
        combined,
        resid_col='resid',
        features=['temperature', 'precip_log', 'wind_speed']
    )

    plot_partial_effects(
        gam,
        feature_names=['temperature', 'log(precipitation+1)', 'wind_speed']
    )

    evaluate_model(df_model, true_col='resid')

    df_model.to_csv('tables/model_results.csv')

    summarize_results(gam, sar, df_model, ['temperature', 'precip_log', 'wind_speed'])


if __name__ == '__main__':
    main()
