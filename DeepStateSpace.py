import paddle
import paddle.nn as nn

class DeepSS(nn.Layer):
    def __init__(self, input_size, hidden_size0 = 64, hidden_size1 = 64, hidden_size2 = 133, drop_out = 0.3):
        super(DeepSS, self).__init__()
        self.input_size = input_size
        self.drop_out = drop_out
        self.hidden_size0 = hidden_size0
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        
        # task1: fit by deepstatespace
        self.lstm = nn.LSTM(self.input_size, self.hidden_size0, 2)
        self.fit_net = nn.Sequential(
            nn.Linear(self.hidden_size0, self.hidden_size1),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_out),
            nn.Linear(self.hidden_size1, self.hidden_size1),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_out),
            nn.Linear(self.hidden_size1, hidden_size2),
            nn.Sigmoid()
        )

        # task2: predict by params
        self.pred_net = nn.Sequential(
            nn.Linear(hidden_size1*4, self.hidden_size1),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_out),
            nn.Linear(self.hidden_size1, self.hidden_size1),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_out),
            nn.Linear(self.hidden_size1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        params, (h, c) = self.lstm(x)
        #state = paddle.concat([params[:, 0, :], params[:, 64, :], params[:, 128, :], params[:, -1, :]], axis=-1)
        params = self.fit_net(params.reshape([-1, 64]))
        state = paddle.concat([h[0], h[1], c[0], c[1]], axis=-1)
        y_pred = self.pred_net(state)
        return params, y_pred