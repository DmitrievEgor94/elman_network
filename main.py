import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
import yaml
import os


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


if __name__ == '__main__':
    params = yaml.load(open('params.yaml'), yaml.FullLoader)

    # установка сида для того, чтобы результаты получались теми же самыми при повторном запуске
    random_state = params['random_state']

    df = load_dataset('iris_dataset.xlsx')

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=random_state)

    atrs = [col for col in df.columns if 'target' != col]



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
