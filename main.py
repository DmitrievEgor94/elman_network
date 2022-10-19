import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import yaml
import os
import numpy as np

from nets import Elman


def load_dataset(file_name):
    if not os.path.exists(file_name):
        iris = datasets.load_iris(as_frame=True)

        data = iris.data
        data['target'] = iris.target

        data.to_excel(file_name, index=False)
    else:
        data = pd.read_excel(file_name)

    return data

def train_loop(model):
    pass


if __name__ == '__main__':
    params = yaml.load(open('params.yaml'), yaml.FullLoader)

    # установка сида для того, чтобы результаты получались теми же самыми при повторном запуске
    random_state = params['random_state']
    np.random.seed(1)

    df = load_dataset('iris_dataset.xlsx')

    # Кодировка таргета one_hot: 0 - [1, 0, 0], 1 - [0, 1, 0], 2 - [0, 0, 1]
    label_encoder = LabelBinarizer()
    one_hot_target = label_encoder.fit_transform(df['target'])

    df.loc[:, 'target_one_hot'] = pd.Series(list(one_hot_target))
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=random_state)
    print(train_df.shape, test_df.shape)

    atrs = [col for col in df.columns if col not in ['target', 'target_one_hot']]

    num_atrs = len(atrs)
    outputs_num = train_df.target.unique().shape[0]

    net = Elman(num_atrs, (num_atrs + outputs_num)//2, outputs_num)

    test_ind = 1

    for i in range(10):
        for j in range(train_df.shape[0]):
            a, b = net.forward(train_df[atrs].iloc[j])
            net.backward(train_df['target_one_hot'].iloc[j])

    a, b = net.forward(test_df[atrs].iloc[1])
    print(np.argmax(a), b)
    print(test_df['target_one_hot'].iloc[1])



