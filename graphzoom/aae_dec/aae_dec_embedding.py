import numpy as np
import scipy.io as sio
import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib as mpl
mpl.use('TkAgg')
import warnings
warnings.filterwarnings("ignore")

# 作者 dreamcold（康玉健）
# 时间 2020-10-25


# 我们实验的网络结构
data_dict = {
    'cora':{
        'n_classes':7,
        'z_dim':128,
        'X_dim':2708,
        'N':1024
    },
    'blog':{
        'n_classes':6,
        'z_dim':128,
        'X_dim':5196,
        'N':1024
    },
    'new_citeseer':{
        'n_classes':6,
        'z_dim':128,
        'X_dim':3258,
        'N':1024
    },
    'blogcatalog':{
        'n_classes':39,
        'z_dim':128,
        'X_dim':10312,
        'N':4096
    },
    '20NG':{
        'n_classes':3,
        'z_dim':128,
        'X_dim':1727,
        'N':1024
    },
    'flickr':{
        'n_classes':195,
        'z_dim':128,
        'X_dim':80513,
        'N':8192
    }
}


# 本实验的网络结构用于辅助获得我们的嵌入式表示


# cora
n_classes = 7
z_dim = 128
X_dim = 2708
N = 1024

'''
# blogcatalog
n_classes = 39
z_dim = 128
X_dim = 10312
N = 4096
'''
'''
# blog
n_classes = 6
z_dim = 128
X_dim = 5196
N = 1024
'''
'''
# citeseer
n_classes = 6
z_dim = 128
X_dim = 3258
N = 1024
'''
'''
# 20NG
n_classes = 3
z_dim = 128
X_dim = 1727
N = 1024
'''
'''
# wine
n_classes = 3
z_dim = 32
X_dim = 178
N = 128
'''

class Q_net(nn.Module):
    def __init__(self):
        super(Q_net, self).__init__()
        self.lin1 = nn.Linear(X_dim, N)
        self.lin2 = nn.Linear(N, N)
        # Gaussian code (z)
        self.lin3gauss = nn.Linear(N, z_dim)

    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.2, training=self.training)
        x = F.relu(x)
        xgauss = self.lin3gauss(x)

        return xgauss


# 检测本文提出的算法
def read_ours_output(file_name, dataset_name):
    Q = Q_net()
    # 这里要读取我们之前训练好的模型
    Q.load_state_dict(torch.load('./outputs/ours/best/' + file_name + '.pkl',map_location='cpu'))

    # 导入原始的网络
    network = sio.loadmat('./dataset_ly/'+dataset_name + '_network.mat')["graph_sparse"]
    if not isinstance(network, np.ndarray):
        network = network.toarray()

    unlabeled_trainset = torch.from_numpy(network).float()
    reprsn = Q(unlabeled_trainset)
    reprsn = reprsn.detach().numpy()
    # 获得我们的表示
    return reprsn, network
