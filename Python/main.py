if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPRegressor


    def normalization(array):
        array = 2. * (array - np.min(array)) / np.ptp(array) - 1
        return array


    rawdata = pd.read_excel('data_aggregated_without_zeros.xlsx', engine='openpyxl').iloc[:, :5]  # load the excel file
    nplist = rawdata.T.to_numpy()
    data = nplist.T
    data[:, 1:4] = np.log(data[:, 1:4])  # since the range of the last 2 columns is too large, take the log of them
    data[:, 0] = normalization(data[:, 0])  # do the normalization since the range become too small
    data[:, 1] = normalization(data[:, 1])
    data[:, 2] = normalization(data[:, 2])
    data[:, 3] = normalization(data[:, 3])

    X_train, X_test, y_train, y_test = train_test_split(data[:, :4], data[:, 4], test_size=40, random_state=42)

    reg = MLPRegressor(random_state=1, hidden_layer_sizes=(256, 256), max_iter=10000)
    reg.fit(X_train, y_train)
    print(np.mean((reg.predict(X_test) - y_test) ** 2))
