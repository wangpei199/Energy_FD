import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit

def data_div(ind_pkl_files, ood_pkl_files, n=0.5):
    X_p, X_ood = load_data(ind_pkl_files, ood_pkl_files)
    X_p.extend(X_ood)
    scaler = MinMaxScaler()
    X_p = scaler.fit_transform(X_p)
    y = KMeans(n_clusters=2, random_state=9).fit_predict(X_p)
    
    X_t = []
    for i in range(2):
        split = StratifiedShuffleSplit(n_splits=1, test_size=n)
        a, b = split.split(y[i])
        X_t[0].extend(y[i][a])
        X_t[1].extend(y[i][b])
        
    return X_t

def load_data(pkl_list):
    X = []
    for  each_pkl in pkl_list:
        pic = open(each_pkl,'rb')
        item= pickle.load(pic)
        x = item[0][-1, 0:7]
        y = int(item[1]['label'][0])
        x.append(y)
        X.append(x)
    X = np.vstack(X)
    return X