import os.path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import log_loss
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error

from nets import ElmanClassification, ElmanRegression

import yaml


def load_dataset(regression_folder):
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


def get_learning_rate_graphs(net_params, df, train_inds, test_inds, norm_value):
    learning_rates = np.arange(0.001, 0.3, 0.01).tolist()
    learning_rates += [0.5, 1, 1.5, 2, 10]

    folder_to_save_data = 'regression_data/results'

    all_losses = []

    for lr in learning_rates:
        np.random.seed(1)
        net = ElmanRegression(*net_params)

        print('learning_rate: ', lr)
        for i in range(500):
            for train_row in train_inds:
                train_ind_row, target_ind = train_row
                net.forward(df.iloc[train_ind_row, 0])
                net.backward(df.iloc[target_ind, 0], lrate=lr)

        predicts = []
        targets = []

        for test_row in test_inds:
            test_ind_row, target_ind = test_row
            a = net.forward(df.iloc[test_ind_row, 0])
            predicts.append(a[0] * norm_value)
            targets.append(df.iloc[target_ind, 0] * norm_value)

        # print(predicts, targets)
        all_losses.append(mean_squared_error(y_true=targets, y_pred=predicts)**(1/2))
        print('loss:', all_losses[-1])

    df_lr_losses = pd.DataFrame(zip(learning_rates, all_losses), columns=['learning rate', 'средняя квадратичная ошибка'])
    df_lr_losses.to_excel(f'{folder_to_save_data}/learning_rate_loss.xlsx', index=False)


def get_training_data_length_graphs(net_params, df, norm_value):
    train_data_lengths = [150, 200, 250, 300, 330]

    all_losses = []

    folder_to_save_data = 'regression_data/results'

    for length in train_data_lengths:
        print("length:", length)
        np.random.seed(1)
        net = ElmanRegression(*net_params)

        train_ind_data = [(start_ind + i, start_ind[-1] + i + 1) for i in range(length)]
        test_ind_data = [(start_ind + i, start_ind[-1] + i + 1) for i in
                         range(length, df.shape[0] - window_size)]

        for i in range(500):
            for train_row in train_ind_data:
                train_ind_row, target_ind = train_row
                net.forward(df.iloc[train_ind_row, 0])
                net.backward(df.iloc[target_ind, 0])

        predicts = []
        targets = []

        for test_row in test_ind_data:
            test_ind_row, target_ind = test_row
            a = net.forward(df.iloc[test_ind_row, 0])
            predicts.append(a[0] * norm_value)
            targets.append(df.iloc[target_ind, 0] * norm_value)

        all_losses.append(mean_squared_error(y_true=targets, y_pred=predicts)**(1/2))

        print('loss:', all_losses[-1])

    df_lr_losses = pd.DataFrame(zip(train_data_lengths, all_losses), columns=['traing data length', 'losses'])
    df_lr_losses.to_excel(f'{folder_to_save_data}/training_data_length_loss.xlsx', index=False)


def get_number_neurons_length_graphs(net_params, df, train_ind_data, test_ind_data, norm_value):
    num_atrs, _, outputs_num = net_params

    all_losses = []

    folder_to_save_data = 'regression_data/results'

    hidden_layer_length_list = [2, 3, 5, 7, 10]

    for hidden_layer_length in hidden_layer_length_list:
        np.random.seed(1)
        net = ElmanRegression(num_atrs, hidden_layer_length, outputs_num)

        print(hidden_layer_length)

        for i in range(500):
            for train_row in train_ind_data:
                train_ind_row, target_ind = train_row
                net.forward(df.iloc[train_ind_row, 0])
                net.backward(df.iloc[target_ind, 0])

        predicts = []
        targets = []

        for test_row in test_ind_data:
            test_ind_row, target_ind = test_row
            a = net.forward(df.iloc[test_ind_row, 0])
            predicts.append(a[0] * norm_value)
            targets.append(df.iloc[target_ind, 0] * norm_value)

        all_losses.append(mean_squared_error(y_true=targets, y_pred=predicts)**(1/2))
        print('loss:', all_losses[-1])

    df_lr_losses = pd.DataFrame(zip(hidden_layer_length_list, all_losses), columns=['hidden layer length', 'losses'])
    df_lr_losses.to_excel(f'{folder_to_save_data}/hidden_layer_length_loss.xlsx', index=False)


def get_window_size_graph(net_params, df, norm_value):
    sizes = [3, 5, 7, 10, 15, 30]

    all_losses = []

    folder_to_save_data = 'regression_data/results'

    for window_size in sizes:
        print("widnow_size:", window_size)
        np.random.seed(1)
        _, hidden, outputs = net_params

        net = ElmanRegression(window_size, hidden, outputs)

        start_ind = np.arange(0, window_size, 1)

        train_ind_data = [(start_ind + i, start_ind[-1] + i + 1) for i in range(300)]
        test_ind_data = [(start_ind + i, start_ind[-1] + i + 1) for i in
                         range(300, df.shape[0] - window_size)]

        for i in range(500):
            for train_row in train_ind_data:
                train_ind_row, target_ind = train_row
                net.forward(df.iloc[train_ind_row, 0])
                net.backward(df.iloc[target_ind, 0])

        predicts = []
        targets = []

        for test_row in test_ind_data:
            test_ind_row, target_ind = test_row
            a = net.forward(df.iloc[test_ind_row, 0])
            predicts.append(a[0] * norm_value)
            targets.append(df.iloc[target_ind, 0] * norm_value)

        all_losses.append(mean_squared_error(y_true=targets, y_pred=predicts) ** (1 / 2))

        print('loss:', all_losses[-1])

    df_lr_losses = pd.DataFrame(zip(sizes, all_losses), columns=['window size', 'средняя квадратичная ошибка'])
    df_lr_losses.to_excel(f'{folder_to_save_data}/window_size_loss.xlsx', index=False)


def print_metrics(true_ar, predict_ar, dataset_type):
    print(f'mse_{dataset_type}: ', mean_squared_error(true_ar, predict_ar))

    print(f'mape_{dataset_type}: ', mean_absolute_percentage_error(true_ar, predict_ar))

    print(f'mae_{dataset_type}: ', mean_absolute_error(true_ar, predict_ar))


def train_net_print_metrics(df, last_index_train, epochs_number,
                            window_size, print_values=False, make_normalization=False):
    # генерим индексы, которые получаются при скользящем окне
    start_ind = np.arange(0, window_size, 1)

    if make_normalization:
        norm_value = df.iloc[:, 0].max()
        df.iloc[:, 0] /= norm_value

    train_ind_data = [(start_ind + i, start_ind[-1] + i + 1) for i in range(last_index_train)]
    test_ind_data = [(start_ind + i, start_ind[-1] + i + 1) for i in range(last_index_train, df.shape[0] - window_size)]

    net = ElmanRegression(window_size, (window_size + 1) // 2, 1)

    train_predictions = []
    train_targets = []

    print('train/test split:', len(train_ind_data), '/', len(test_ind_data))

    for test_row in train_ind_data:
        test_ind_row, target_ind = test_row[0], test_row[1]
        a = net.forward(df.iloc[test_ind_row, 0])

        train_predictions.append(a[0])
        train_targets.append(df.iloc[target_ind].iloc[0])

    print('Стартовые значения метрик:')

    if make_normalization:
        train_targets = np.array(train_targets)
        train_predictions = np.array(train_predictions)

        train_targets *= norm_value
        train_predictions *= norm_value

    print_metrics(train_targets, train_predictions, dataset_type='train')

    for j in range(epochs_number):
        for i, train_row in enumerate(train_ind_data):
            train_ind_row, target_ind = train_row[0], train_row[1]
            net.forward(df.iloc[train_ind_row, 0])
            net.backward(df.iloc[target_ind, 0], lrate=0.08)

    train_predictions = []
    train_targets = []

    for i, train_row in enumerate(train_ind_data):
        train_ind_row, target_ind = train_row[0], train_row[1]
        a = net.forward(df.iloc[train_ind_row, 0])
        train_predictions.append(a[0])
        train_targets.append(df.iloc[target_ind, 0])

    test_predictions = []
    test_targets = []

    if print_values:
        print('\nПредсказания и факты на тестовой выборке:')

    for i, test_row in enumerate(test_ind_data):
        test_ind_row, target_ind = test_row[0], test_row[1]
        a = net.forward(df.iloc[test_ind_row, 0])

        if print_values:
            if make_normalization:
                print(a[0]*norm_value, df.iloc[target_ind, 0]*norm_value)
            else:
                print(a[0]*norm_value, df.iloc[target_ind, 0]*norm_value)

        test_targets.append(df.iloc[target_ind, 0])
        test_predictions.append(a[0])

    if make_normalization:
        train_targets = np.array(train_targets)
        train_predictions = np.array(train_predictions)

        test_targets = np.array(test_targets)
        test_predictions = np.array(test_predictions)

        train_targets *= norm_value
        train_predictions *= norm_value

        test_targets *= norm_value
        test_predictions *= norm_value

    print('\nМетрики на трейне:')
    print_metrics(train_targets, train_predictions, dataset_type='train')

    print('\nМетрики на тесте:')
    print_metrics(test_targets, test_predictions, dataset_type='test')


def gen_garmonic_dataframe(type='sin', show_graph=False):
    if type == 'sin':
        func = np.sin
    elif type == 'cos':
        func = np.cos
    else:
        raise Exception('Напиши гармонику!')

    X_train = np.arange(0, 270, 1)
    y_train = func(X_train)

    X_test = np.arange(270, 307, 1)
    y_test = func(X_test)

    if show_graph:
        fig, ax = plt.subplots(1, 1, figsize=(15, 4))
        ax.plot(X_train, y_train, lw=3, label='train data')
        ax.plot(X_test, y_test, lw=3, label='test data')
        ax.legend(loc="lower left")
        plt.show()

    df = pd.DataFrame(np.concatenate([y_train, y_test]), columns=['values'])

    return df


def first_part_of_lab():
    params = yaml.load(open('params.yaml'), yaml.FullLoader)

    # установка сида для того, чтобы результаты получались теми же самыми при повторном запуске
    random_state = params['random_state']
    np.random.seed(1)
    #
    print('-' * 20, 'Результаты для синуса:', '-' * 20)
    sin_df = gen_garmonic_dataframe(type='sin', show_graph=False)

    train_net_print_metrics(sin_df, last_index_train=270, epochs_number=80, window_size=7)

    print('-' * 20, 'Результаты для косинуса:', '-' * 20)
    cos_df = gen_garmonic_dataframe(type='cos')

    train_net_print_metrics(cos_df, last_index_train=270, epochs_number=60, window_size=7)

    print('-' * 20, 'Результаты для реального датасета:', '-' * 20)

    df = load_dataset(regression_folder='regression_data')

    train_net_print_metrics(df, last_index_train=340, epochs_number=1200, window_size=7,
                            print_values=True, make_normalization=True)


if __name__ == '__main__':
    window_size = 7

    df = load_dataset(regression_folder='regression_data')

    start_ind = np.arange(0, window_size, 1)

    norm_value = df.iloc[:, 0].max()
    df.iloc[:, 0] /= norm_value

    last_index_train = 300

    train_ind_data = [(start_ind + i, start_ind[-1] + i + 1) for i in range(last_index_train)]
    test_ind_data = [(start_ind + i, start_ind[-1] + i + 1) for i in range(last_index_train, df.shape[0] - window_size)]

    net_params = (window_size, (window_size + 1) // 2, 1)
    print(df.shape[0])
    # get_learning_rate_graphs(net_params, df, train_ind_data, test_ind_data, norm_value)
    # get_training_data_length_graphs(net_params, df, norm_value)
    # get_number_neurons_length_graphs(net_params, df, train_ind_data, test_ind_data, norm_value)
    get_window_size_graph(net_params, df, norm_value)
    print(test_ind_data)