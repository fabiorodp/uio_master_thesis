# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

import numpy as np


def bollingerBands(data, freq=5, std_value=2.0, column_base='close'):
    """
    Function to calculate the upper, lower and middle Bollinger Bands (BBs).
    Parameters:
    ===================
    :param data: pd.DataFrame: Time series DataFrame containing OLHCV values.
    :param freq: int: Frequency of the rolling.
    :param std_value: float: Value of the Standard Deviation.
    :param column_base: srt: Name of the column that will be the base of
                             the computations.
    Return:
    ===================
    data: pd.DataFrame: Time series DataFrame containing OLHCV values plus
                        the upper, lower and middle Bollinger Bands (BBs).
    """
    data[f'{freq} SMA'] = data[column_base].rolling(window=freq).mean()
    data['STD'] = data[column_base].rolling(window=freq).std()
    data[f'{freq} BB Upper'] = data[f'{freq} SMA'] + (data['STD'] * std_value)
    data[f'{freq} BB Lower'] = data[f'{freq} SMA'] - (data['STD'] * std_value)
    data.drop('STD', axis=1, inplace=True)

    return data


def ema(data, freq=5, column_base='close'):
    """
    Function to calculate the Exponential Moving Average (EMA).
    Parameters:
    ===================
    :param data: pd.DataFrame: Time series DataFrame containing OLHCV values.
    :param freq: int: Frequency of the rolling.
    :param column_base: srt: Name of the column that will be the base of
                         the computations.
    Return:
    ===================
    data: pd.DataFrame: Time series DataFrame containing OLHCV values plus
                        the Exponential Moving Average (EMA).
    Notes:
    ===================
    According to definition, the first value for the EMA is a SMA.
    """
    sma = data[column_base].rolling(window=freq).mean()
    ema = data[column_base].copy()
    ema.iloc[:freq] = sma.iloc[:freq]
    data[f'{freq} EMA'] = ema.ewm(span=freq, adjust=False,
                                  ignore_na=False).mean()

    return data


def sma(data, freq=5, column_base='close'):
    """
    Function to calculate the Simple Moving Average (SMA).
    Parameters:
    ===================
    :param data: pd.DataFrame: Time series DataFrame containing OLHCV values.
    :param freq: int: Frequency of the rolling.
    :param column_base: srt: Name of the column that will be the base of
                     the computations.
    Return:
    ===================
    data: pd.DataFrame: Time series DataFrame containing OLHCV values plus
                        the Simple Moving Average (SMA).
    """
    data[f'{freq} SMA'] = data[column_base].rolling(window=freq).mean()

    return data


def lwma(data, freq=5, column_base='close'):
    """
    Function to calculate the Linearly Weighted Moving Average (LWMA).
    Parameters:
    ===================
    :param data: pd.DataFrame: Time series DataFrame containing OLHCV values.
    :param freq: int: Frequency of the rolling.
    :param column_base: srt: Name of the column that will be the base of
                     the computations.
    Return:
    ===================
    data: pd.DataFrame: Time series DataFrame containing OLHCV values plus
                        the Linearly Weighted Moving Average (LWMA).
    Notes:
    ===================
    Using the .apply() method we pass our own function (a lambda function)
    to compute the dot product of weights and prices in our rolling window
    (prices in the window will be multiplied by the corresponding weight,
    then summed), then dividing it by the sum of the weights.
    """
    weights = np.arange(1, freq + 1)

    data[f'{freq} LWMA'] = \
        data[column_base].rolling(freq).apply(
            lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)

    return data
