import torch
import torch.nn as nn
import torch.nn.functional as F

class Action_Predicter_Dense(nn.Module):
    def __init__(self):
        super(Action_Predicter_Dense, self).__init__()

        self.d1 = nn.Linear(64, 128)
        self.d2 = nn.Linear(128, 64)
        self.d3 = nn.Linear(64, 16)
        self.d4 = nn.Linear(16, 2)

    def forward(self, x):
        #print('Dense: ', x.shape)
        x = self.d1(x)
        #print('d1')
        x = F.relu(x)
        #print('relu')
        x = self.d2(x)
        #print('d1')
        x = F.relu(x)
        #print('relu')
        x = self.d3(x)
        #print('d1')
        x = F.relu(x)
        #print('relu')
        x = self.d4(x)
        #print('d1')
        out = F.tanh(x)

        return out

class Action_Predicter_LSTM(nn.Module):
    def __init__(self):
        super(Action_Predicter_LSTM, self).__init__()

        self.zsize = 16

        self.lstm1 = nn.LSTM(self.zsize, self.zsize, 3, dropout = 0.5)
        self.lstm2 = nn.LSTM(self.zsize, self.zsize, 1, dropout = 0.5)
        self.lstm3 = nn.LSTM(self.zsize, self.zsize, 1, dropout = 0.5)
        self.d1 = nn.Linear(64, self.zsize)
        self.d2 = nn.Linear(self.zsize * 2+(16-self.zsize), self.zsize)
        self.d3 = nn.Linear(self.zsize * 3+(16-self.zsize), self.zsize)
        self.d4 = nn.Linear(self.zsize * 4+(16-self.zsize), self.zsize)
        self.d5 = nn.Linear(self.zsize, 16)
        self.d6 = nn.Linear(16, 2)


    def forward(self, x):

        x = F.relu(self.d1(x))
        print('x.shape: ', x.shape)
        x=x.unsqueeze(0)
        print('x.shape: ', x.shape)

        outputs, (hidden, cell) = self.lstm1(x)

        #print(x.shape)
        print(hidden[-2:-1,:].shape)

        #x = torch.cat((x,hidden[-2:-1,:]),1)
        #print(x.shape)

        #outputs, (hidden, cell) = self.lstm2(F.relu(self.d2(x)))

        #print(hidden.shape)

        #x = torch.cat((x,hidden),1)

        #outputs, (hidden, cell) = self.lstm2(F.relu(self.d3(x)))

        #x = torch.cat((x,hidden),1)

        #x = self.d4(x)
        #x = F.relu(x)
        #x = self.d5(x)
        #x = F.relu(x)
        x = self.d6(hidden[-2:-1,:])

        out = torch.tanh(x)

        return out
