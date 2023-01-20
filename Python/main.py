import predicted_model as pm
import data_load as dl
import torch
from torch import nn
from torch import optim

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # to use gpu
    train_dataset, test_dataset = dl.data_load('data_aggregated_without_zeros.xlsx',
                                               5).data_prepare()  # data split into train set and test set
    model = pm.predicted_model(input_size=4, hidden_size1=256, hidden_size2=1024, hidden_size3=32).to(device)  # set the model size
    criterion = nn.MSELoss()  # loss function set as MSE
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)  # with weight_decay to care about regularization and avoid overfitting
    for epoch in range(100):  # training the whole dataset 100 times
        for batch_idx, (batch_x, batch_y) in enumerate(train_dataset):
            batch_x = batch_x.to(device=device)
            batch_y = batch_y.to(device=device)
            scores = model(batch_x)  # feed the model
            loss = criterion(scores, batch_y)
            if loss < 18000.0:
                print(loss)
            optimizer.zero_grad()
            loss.backward()  # backpropagation
            optimizer.step()

    # test on the test set and calculate the MSE error
    error = 0.0
    for test_x, test_y in test_dataset:
        test_x = test_x.to(device=device)
        predict = model(test_x)
        error += (test_y - predict) ** 2
    error /= 40
    print("the MSE on the test set is %10.4f" % error)
