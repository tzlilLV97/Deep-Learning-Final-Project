import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report


def data_preparation(df_first):
    df = df_first.iloc[:, [2, 5, 6, 7, 8, 9, 10, 11, 14, 17, 20, 23, 26, 27, 28, 29, 30, 36, 39, 42, 45, 48,  51, 52, 53, 54, 55]]
    labels = df_first.iloc[:, 4].to_numpy() - 1
    df = df.to_numpy()
    return df, labels

class PyTorchMLP(nn.Module):
    def __init__(self, num_hidden=5):
        super(PyTorchMLP, self).__init__()
        self.input_size = 27#input_size
        self.hidden_size = 27#hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()
    # self.flatten = nn.Flatten()
    #     self.linear_relu_stack = nn.Sequential(
    #         nn.Linear(27, num_hidden),
    #         nn.ReLU(),
    #         nn.Linear(num_hidden, num_hidden),
    #         nn.ReLU(),
    #         nn.Linear(num_hidden, 1),
    #         nn.ReLU()
    #     )
        # self.layer1 = nn.Linear(27, num_hidden)
        # self.layer2 = nn.Linear(num_hidden, 1)
     #   self.num_hidden = num_hidden
    def forward(self, inp):
        hidden = self.fc1(inp)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        return output
        inp = self.flatten(inp)
        return self.linear_relu_stack(inp)
        inp1 = self.layer1(inp)
        return self.layer2(inp1)
        # TODO: complete this function
        # Note that we will be using the nn.CrossEntropyLoss(), which computes the softmax operation internally, as loss criterion



def get_batch(data,labels, start, end):
    x = data[start:end]
    s = labels[start:end]
    return x, s


def estimate_accuracy_torch(model, data, labels, batch_size=5000, max_N=100000):
    """
    Estimate the accuracy of the model on the data. To reduce
    computation time, use at most `max_N` elements of `data` to
    produce the estimate.
    """

    correct = 0
    N = 0
    for i in range(0, len(data), batch_size):
        # get a batch of data
        xt, st = get_batch(data, labels, i, i + batch_size)
        # forward pass prediction
        y = model(torch.Tensor(xt))
        y = y.detach().numpy()  # convert the PyTorch tensor => numpy array
        y = np.where(y > 0.5, 1, 0)
        pred = np.squeeze(y)
        correct += np.sum(pred==st)
        N += len(st)
        if N > max_N:
            break
    return correct / N

def plot_learning_curve(iters, losses, iters_sub, train_accs, val_accs):
    """
    Plot the learning curve.
    """
    plt.title("Learning Curve: Loss per Iteration")
    plt.plot(iters, losses, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()

    plt.title("Learning Curve: Accuracy per Iteration")
    plt.plot(iters_sub, train_accs, label="Train")
    plt.plot(iters_sub, val_accs, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()

def make_prediction_torch(model, game_data):

    b = game_data
    toBePredicted = torch.Tensor(b)
    print(toBePredicted)
    output = model(toBePredicted)
    print(output)
    #  Write your code here

def pytorch_gradient_descent(model, train_data,
                                 validation_data,
                                 batch_size=100,
                                 learning_rate=0.001,
                                 weight_decay=0,
                                 max_iters=1000):
    #criterion =nn.BCELoss()# nn.CrossEntropyLoss()
    #criterion = nn.NLLLoss()
    criterion = nn.MSELoss()
    #criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=weight_decay)
    optimizer = optim.Adam(model.parameters(),
                          lr=learning_rate,
                           weight_decay=weight_decay)

    iters, losses = [], []
    iters_sub, train_accs, val_accs = [], [], []

    n = 0  # the number of iterations
    while True:
        for i in range(0, train_data[0].shape[0], batch_size):
            if (i + batch_size) > train_data[0].shape[0]:
                break
            # get the input and targets of a minibatch
            xt, st = get_batch(train_data[0],train_data[1], i, i + batch_size)
            # convert from numpy arrays to PyTorch tensors
            xt = torch.Tensor(xt)
            st = st
            st = torch.Tensor(st).float().unsqueeze(1)

            zs = model(xt) # compute prediction logit
            ## get the pytorch tensor size of zs
            ## use torch criterion to compute the loss from zs to st
         #   print(st)
            loss = criterion(st, zs)  # compute the total loss
           # print("DEBUG")
          #  print(loss)
            loss.backward()  # compute updates for each parameter
            optimizer.step()  # make the updates for each parameter
            optimizer.zero_grad()  # a clean up step for PyTorch
            # save the current training information
            iters.append(n)
            losses.append(float(loss) / batch_size)  # compute *average* loss

            if n % 100 == 0:
            #    print(classification_report(zs.detach().numpy(),st.detach().numpy()))
            #    continue
                iters_sub.append(n)
                train_cost = float(loss.detach().numpy())
                train_acc = estimate_accuracy_torch(model, train_data[0],train_data[1])
                train_accs.append(train_acc)
                val_acc = estimate_accuracy_torch(model, validation_data[0], validation_data[1])
                val_accs.append(val_acc)
                print("Iter %d. [Val Acc %.0f%%] [Train Acc %.0f%%, Loss %f]" % (
                    n, val_acc * 100, train_acc * 100, train_cost))

            # increment the iteration number
            n += 1

            if n > max_iters:
                return iters, losses, iters_sub, train_accs, val_accs
def main():
    df = pd.read_csv("data/games.csv")
    data, labels = data_preparation(df)
    n = len(data)
    len_train = int(0.8 * n)
    len_test = (n-int(len_train))//2
    len_val = len_test
    train, valid, test = (data[:len_train],labels[:len_train]), (data[len_train:len_train+len_test],labels[len_train:len_train+len_test]), (data[len_train+len_test:],labels[len_train+len_test:])
    pytorch_mlp = PyTorchMLP()
    print(len(train[0][0]))
    learning_curve_info = pytorch_gradient_descent(pytorch_mlp, train,valid ,batch_size=4000,
                                     learning_rate=0.000101,
                                     weight_decay=0.000000001,
                                     max_iters=27000)

    plot_learning_curve(*learning_curve_info)
    print(test[0][0])
    print("Length of train : ", len_train)
    print("Length of test : ", len_test)
    print("Length of validation : ",len_val)
    print("Total Length : ",len(data))


if __name__ == '__main__':
    main()