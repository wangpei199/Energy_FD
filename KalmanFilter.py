import numpy as np
import scipy
import paddle

class Kalman_filter():
    def kalman_step(self, F, H, b, Q, R, L, P, z=None):
        sampling = z is None

        L = paddle.matmul(F, L)
        P = paddle.matmul(paddle.matmul(F, P), paddle.transpose(F, [0, 1, 3, 2])) + Q
        z_pred = paddle.matmul(H, L) + b
        S = paddle.matmul(paddle.matmul(H, P), paddle.transpose(H, [0, 1, 3, 2])) + R
        if sampling:
            z = paddle.distribution.Normal(z_pred, S).sample([1])[0]
        else:
            log_prob = paddle.distribution.Normal(z_pred, S).log_prob(z)
        try:    #attention: S 
            S = np.linalg.inv(S.numpy())
            S = paddle.to_tensor(S)
            K = paddle.matmul(paddle.matmul(P, paddle.transpose(H, [0, 1, 3, 2])), S)
            y = z - z_pred
            L = L + paddle.matmul(K, y)
            P = P - paddle.matmul(paddle.matmul(K, H), P)
    
            if sampling:
                return L, P, z
            return L, P, log_prob
        except:
            return L, P, None

    def kalman_filtering(self, F, H, b, Q, R, L, P, z=None):
        time_steps = F.shape[1]
        if z is None:
            samples = []
            for t in range(time_steps):
                Ft = F[:, t:t+1, :, :]
                Ht = H[:, t:t+1, :, :]
                bt = b[:, t:t+1, :, :]
                Qt = Q[:, t:t+1, :, :]
                Rt = R[:, t:t+1, :, :]
                L, P, zt = self.kalman_step(Ft, Ht, bt, Qt, Rt, L, P)
                samples.append(zt)
            return samples
        else:
            log_probs = []
            for t in range(time_steps):
                Ft = F[:, t:t+1, :, :]
                Ht = H[:, t:t+1, :, :]
                bt = b[:, t:t+1, :, :]
                Qt = Q[:, t:t+1, :, :]
                Rt = R[:, t:t+1, :, :]
                zt = z[:, t:t+1, :, :]
                L, P, log_prob = self.kalman_step(Ft, Ht, bt, Qt, Rt, L, P, zt)
                log_probs.append(log_prob)
            loss = -paddle.sum(paddle.to_tensor(log_probs))
            return L, P, loss