import os.path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import log_loss
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error

from nets import ElmanClassification, ElmanRegression


def load_dataset():
    if os.path.exists(f'{regression_folder}/processed_data.csv'):
        res_df = pd.read_csv(f'{regression_folder}/processed_data.csv', index_col='time')
    else:
        df = pd.read_csv(f'{regression_folder}/data.csv')

        df = df.iloc[11:, :2]
        df.columns = ['time', 'tempr']
        df['time'] = pd.to_datetime(df['time']).dt.date
        df['tempr'] = df['tempr'].astype(float)
        res_df = df.groupby('time')['tempr'].mean()
        res_df.to_csv(f'{regression_folder}/processed_data.csv')

    res_df.sort_values(by='time', inplace=True)

    return res_df


def get_learning_rate_graphs(net_params, train_df, test_df):
    learning_rates = np.arange(0.001, 0.3, 0.01).tolist()
    learning_rates += [0.5, 1, 1.5, 2, 10]
    moments = [0, 0.1]

    folder_to_save_data = 'classification_data/results'

    for moment in moments:
        all_losses = []

        for lr in learning_rates:
            net = ElmanClassification(*net_params)

            print('learning_rate: ', lr)
            for i in range(90):
                for j in range(train_df.shape[0]):
                    a, b = net.forward(train_df[atrs].iloc[j])
                    net.backward(train_df['target_one_hot'].iloc[j], lrate=lr, momentum=moment)

            predicts = []
            losses = []

            for j in range(test_df.shape[0]):
                a, b = net.forward(test_df[atrs].iloc[j])
                losses.append(log_loss(y_true=test_df['target_one_hot'].iloc[j], y_pred=b))
                predicts.append(np.argmax(a))

            all_losses.append(np.mean(losses))
            print('loss:', np.mean(losses))
            # print('accuracy:', np.mean(np.array(predicts == test_df['target'])))

        df_lr_losses = pd.DataFrame(zip(learning_rates, all_losses), columns=['learning rate', 'losses'])
        df_lr_losses.to_excel(f'{folder_to_save_data}/learning_rate_loss_moment_{round(moment, 2)}.xlsx', index=False)


def get_training_data_length_graphs(net_params, train_df, test_df):
    train_data_lengths = [30, 60, 90, 120]

    learning_rate = 0.1
    momentum = 0.1

    all_losses = []

    folder_to_save_data = 'classification_data/results'

    for length in train_data_lengths:
        net = ElmanClassification(*net_params)

        train_df = train_df.sample(frac=1, random_state=2).reset_index(drop=True)

        print(length)

        for _ in range(35):
            for j in range(length):
                a, b = net.forward(train_df[atrs].iloc[j])
                net.backward(train_df['target_one_hot'].iloc[j], lrate=learning_rate, momentum=momentum)

        losses = []

        for j in range(test_df.shape[0]):
            a, b = net.forward(test_df[atrs].iloc[j])
            losses.append(log_loss(y_true=test_df['target_one_hot'].iloc[j], y_pred=b))

        all_losses.append(np.mean(losses))

        print('loss:', np.mean(losses))

    df_lr_losses = pd.DataFrame(zip(train_data_lengths, all_losses), columns=['traing data length', 'losses'])
    df_lr_losses.to_excel(f'{folder_to_save_data}/training_data_length_loss.xlsx', index=False)


def get_number_neurons_length_graphs(net_params, train_df, test_df):
    num_atrs, _, outputs_num = net_params

    learning_rate = 0.1
    momentum = 0.1

    all_losses = []

    folder_to_save_data = 'classification_data/results'

    hidden_layer_length_list = [2, 3, 5, 7, 10]

    for hidden_layer_length in hidden_layer_length_list:
        net = ElmanClassification(num_atrs, hidden_layer_length, outputs_num)

        print(hidden_layer_length)

        for _ in range(90):
            for j in range(train_df.shape[0]):
                a, b = net.forward(train_df[atrs].iloc[j])
                net.backward(train_df['target_one_hot'].iloc[j], lrate=learning_rate, momentum=momentum)

        losses = []

        for j in range(test_df.shape[0]):
            a, b = net.forward(test_df[atrs].iloc[j])
            losses.append(log_loss(y_true=test_df['target_one_hot'].iloc[j], y_pred=b))

        all_losses.append(np.mean(losses))

        print('loss:', np.mean(losses))

    df_lr_losses = pd.DataFrame(zip(hidden_layer_length_list, all_losses), columns=['hidden layer length', 'losses'])
    df_lr_losses.to_excel(f'{folder_to_save_data}/hidden_layer_length_loss.xlsx', index=False)


if __name__ == '__main__':
    params = yaml.load(open('params.yaml'), yaml.FullLoader)

    # установка сида для того, чтобы результаты получались теми же самыми при повторном запуске
    random_state = params['random_state']

    np.random.seed(1)

    regression_folder = 'regression_data'

    df = load_dataset()

    max_value = df.tempr.max()
    df['tempr'] /= max_value

    window_size = 7

    # генерим индексы, которые получаются при скользящем окне
    start_ind = np.arange(0, window_size, 1)

    train_ind_data = [(start_ind + i, start_ind[-1] + i + 1) for i in range(300)]
    test_ind_data = [(start_ind + i, start_ind[-1] + i + 1) for i in range(300, df.shape[0] - window_size)]

    print(len(test_ind_data))
    print(test_ind_data[-1])

    test_targets = []

    net = ElmanRegression(window_size, (window_size + 1) // 2, 1)
    test_predictions = []

    for test_row in test_ind_data:
        test_ind_row, target_ind = test_row[0], test_row[1]
        a = net.forward(df.iloc[test_ind_row, 0])

        test_predictions.append(a[0])
        test_targets.append(df.iloc[target_ind].iloc[0])

    print('Стартовые значения mape:', mean_absolute_percentage_error(test_targets, test_predictions))
    print('Стартовые значения mae:', mean_absolute_error(test_targets, test_predictions))

    for j in range(1000):
        for i, train_row in enumerate(train_ind_data):
            train_ind_row, target_ind = train_row[0], train_row[1]
            a = net.forward(df.iloc[train_ind_row, 0])
            net.backward(df.iloc[target_ind, 0], lrate=0.08)

    test_predictions = []

    train_predictions = []
    train_targets = []
    for i, train_row in enumerate(train_ind_data):
        train_ind_row, target_ind = train_row[0], train_row[1]
        a = net.forward(df.iloc[train_ind_row, 0])
        train_predictions.append(a[0] * max_value)
        train_targets.append(df.iloc[target_ind, 0] * max_value)
        net.backward(df.iloc[target_ind, 0], lrate=0.08)

    print('Предсказания и факты на тестовой выборке:')
    for test_row in test_ind_data:
        test_ind_row, target_ind = test_row[0], test_row[1]
        a = net.forward(df.iloc[test_ind_row, 0])
        print(a[0] * max_value, df.iloc[target_ind, 0] * max_value)
        test_predictions.append(a[0])

    print('mse_train: ', mean_squared_error(np.array(train_targets) * max_value,
                                           np.array(train_predictions) * max_value))

    print('mape_test: ', mean_absolute_percentage_error(np.array(test_targets) * max_value,
                                                   np.array(test_predictions) * max_value))

    print('mae_tеst: ', mean_absolute_error(np.array(test_targets) * max_value,
                                       np.array(test_predictions) * max_value))

    print('mse_test: ', mean_squared_error(np.array(test_targets) * max_value,
                                       np.array(test_predictions) * max_value))
