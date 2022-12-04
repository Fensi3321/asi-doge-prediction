"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.3
"""

import pandas as pd
import datetime
import tensorflow as tf
import wandb

wandb.init(project='asi-project')

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

def max_date(data: pd.DataFrame) -> pd.DataFrame:
    return {'max_date': data['Date'].max()}


def avg_high(data: pd.DataFrame) -> pd.DataFrame:
    return {'avg_high': data['High'].mean()}


def prepare_data(data: pd.DataFrame) -> pd.DataFrame:

    # Get day of week from date
    data['Date'] = data['Date'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d')).dt.dayofweek

    # Calcualte avg price across the day
    avg_day = (data['Open'] + data['Close'] + data['Low'] + data['High']) / 4

    data['Mean'] = avg_day

    # drop price related columns as we already have unified Mean column
    data = data.drop(['Open', 'Close', 'High', 'Low', 'Adj Close'], axis=1)

    return data


import optuna
from optuna.integration import KerasPruningCallback
def train_model(df: pd.DataFrame):

    def create_model(trial: optuna.Trial):
        return tf.keras.models.Sequential([
        tf.keras.layers.Dense(trial.suggest_int('dense_1', 64, 256), activation=trial.suggest_categorical('dense_act_1', choices=['relu', 'sigmoid'])),
        tf.keras.layers.Dense(trial.suggest_int('dense_2', 32, 128), activation=trial.suggest_categorical('dense_act_2', choices=['relu', 'sigmoid'])),
        tf.keras.layers.Dense(trial.suggest_int('dense_3', 8, 16), activation=trial.suggest_categorical('dense_act_3', choices=['relu', 'sigmoid'])),
        tf.keras.layers.Dense(1),
    ])

    def objective(trial: optuna.Trial):
        def rmse(y_true, y_pred):
            return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true)))
        
        model = create_model(trial)
    
        model.compile(
        loss=rmse,
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[rmse]
        )

        model.fit(
        x_train, 
        y_train, 
        epochs=10,
        callbacks=[wandb.keras.WandbCallback(), KerasPruningCallback(trial, monitor='val_rmse')]
        )

        loss, rmse = model.evaluate(x_test, y_test)

        return rmse


    transformer = make_column_transformer(
        (MinMaxScaler(), ['Volume']),
        (OneHotEncoder(handle_unknown='ignore'), ['Date'])
    )

    x = df.drop('Mean', axis=1)
    y = df['Mean']

    x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
    )

    transformer.fit(x_train)

    x_train = transformer.transform(x_train)
    x_test = transformer.transform(x_test)

    x_train = x_train.toarray()
    x_test = x_test.toarray()


    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)

    # model = create_model()

    # model.compile(
    # loss=rmse,
    # optimizer=tf.keras.optimizers.Adam(),
    # metrics=[rmse, 'accuracy']
    # )

    # model.fit(
    #     x_train, 
    #     y_train, 
    #     epochs=10,
    #     callbacks=[wandb.keras.WandbCallback()]
    # )


from sklearn import linear_model
def train_model2(data: pd.DataFrame) -> pd.DataFrame:
    X = data.drop('Mean', axis=1)
    Y = data['Mean']
    reg = linear_model.BayesianRidge()
    reg.fit(X, Y)
