import argparse
import torch
import pickle
import numpy as np
from sklearn.manifold import TSNE
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Parameter
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import warnings

warnings.filterwarnings("ignore")

cuda = False

seed = 10

# v9版本针对win7系统调整
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

train_batch_size = 128
valid_batch_size = 128
beta = 10
cita = 10

data_name = '20NG_network'

epochs = 20

data_dict = {
    'cora_network': {
        'n_classes': 7,
        'z_dim': 128,
        'X_dim': 2708,
        'N': 1024
    },
    'blog_network': {
        'n_classes': 6,
        'z_dim': 128,
        'X_dim': 5196,
        'N': 1024
    },
    'new_citeseer_network': {
        'n_classes': 6,
        'z_dim': 128,
        'X_dim': 3258,
        'N': 1024
    },
    'blogcatalog_network': {
        'n_classes': 39,
        'z_dim': 128,
        'X_dim': 10312,
        'N': 4096
    },
    '20NG_network': {
        'n_classes': 3,
        'z_dim': 128,
        'X_dim': 1727,
        'N': 1024
    },
    'flickr_network': {
        'n_classes': 195,
        'z_dim': 128,
        'X_dim': 80513,
        'N': 8192
    }
}

try:
    data_dict[data_name]
    print(data_name)
except:
    print('找不到该数据集')
    print(data_name)
    exit(0)

n_classes = data_dict[data_name]['n_classes']
z_dim = data_dict[data_name]['z_dim']
X_dim = data_dict[data_name]['X_dim']
N = data_dict[data_name]['N']


# writer = SummaryWriter(comment='fun')

##################################
# Load data and create Data loaders
##################################


def load_data(data_path='../data/'):
    print('loading data!')
    trainset_labeled = pickle.load(open(data_path + "train_labeled.p", "rb"))
    # trainset_unlabeled = pickle.load(open(data_path + "train_unlabeled.p", "rb"))
    trainset_unlabeled = pickle.load(open(data_path + data_name + ".p", "rb"))
    # Set -1 as labels for unlabeled data
    trainset_unlabeled.train_labels = torch.from_numpy(np.array([-1] * 47000))
    validset = pickle.load(open(data_path + "validation.p", "rb"))

    train_labeled_loader = torch.utils.data.DataLoader(trainset_labeled,
                                                       batch_size=train_batch_size,
                                                       shuffle=True, **kwargs)

    train_unlabeled_loader = torch.utils.data.DataLoader(trainset_unlabeled,
                                                         batch_size=train_batch_size,
                                                         shuffle=True, **kwargs)

    valid_loader = torch.utils.data.DataLoader(
        validset, batch_size=valid_batch_size, shuffle=True)

    return train_labeled_loader, train_unlabeled_loader, valid_loader


#######################
# k均值聚类 for DEC
#######################


class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters=10, hidden=10, cluster_centers=None, alpha=1.0):
        super(ClusteringLayer, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.hidden = hidden
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.n_clusters,
                self.hidden,
                dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self, x):
        norm_squared = torch.sum((x.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        # soft assignment using t-distribution
        t_dist = (numerator.t() / torch.sum(numerator, 1)).t()
        return t_dist


class DEC(nn.Module):
    def __init__(self, n_clusters=10, autoencoder=None, hidden=10, cluster_centers=None, alpha=1.0):
        super(DEC, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.hidden = hidden
        self.cluster_centers = cluster_centers
        self.autoencoder = autoencoder
        self.clusteringlayer = ClusteringLayer(
            self.n_clusters, self.hidden, self.cluster_centers, self.alpha)

    def target_distribution(self, q_):
        weight = (q_ ** 2) / torch.sum(q_, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

    def forward(self, x):
        x = self.autoencoder(x)
        return self.clusteringlayer(x)

    def visualize(self, epoch, x):
        fig = plt.figure()
        ax = plt.subplot(111)
        x = self.autoencoder(x).detach()
        x = x.cpu().numpy()[:2000]
        x_embedded = TSNE(n_components=2).fit_transform(x)
        plt.scatter(x_embedded[:, 0], x_embedded[:, 1])
        fig.savefig('plots/mnist_{}.png'.format(epoch))
        plt.close(fig)


##################################
# Define Networks
##################################
# Encoder
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


# Decoder
class P_net(nn.Module):
    def __init__(self):
        super(P_net, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, X_dim)

    def forward(self, x):
        x = self.lin1(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin3(x)
        return F.sigmoid(x)


####################
# Utility functions
####################
def save_model(model, filename):
    print('Best model so far, saving it...')
    torch.save(model.state_dict(), filename)


def report_loss(epoch, recon_loss):
    '''
    Print loss
    '''
    print('Epoch-{}; recon_loss: {:.4}'.format(epoch, recon_loss.item()))


def report_kl_loss(epoch, kl_loss):
    '''
    Print loss
    '''
    print('Epoch-{}; kl_loss: {:.4}'.format(epoch, kl_loss.item()))


def create_latent(Q, loader):
    '''
    Creates the latent representation for the samples in loader
    return:
        z_values: numpy array with the latent representations
        labels: the labels corresponding to the latent representations
    '''
    Q.eval()
    labels = []

    for batch_idx, (X, target) in enumerate(loader):

        X = X * 0.3081 + 0.1307
        X.resize_(loader.batch_size, X_dim)
        X, target = Variable(X), Variable(target)
        labels.extend(target.data.tolist())
        if cuda:
            X, target = X.cuda(), target.cuda()
        # Reconstruction phase
        z_sample = Q(X)
        if batch_idx > 0:
            z_values = np.concatenate(
                (z_values, np.array(z_sample.data.tolist())))
        else:
            z_values = np.array(z_sample.data.tolist())
    labels = np.array(labels)

    return z_values, labels


def make_loss(X, z_sample, X_sample, batch_idx):
    def get_1st_loss(X, z_sample):
        pairwise_dist = torch.cdist(z_sample, z_sample)
        Y = X[:, batch_idx:batch_idx + train_batch_size]
        L1_loss = Y * pairwise_dist
        re = torch.sum(L1_loss)
        # print(re)
        return re

    # SDNE function
    def get_2nd_loss(X, X_sample):
        B = X * (beta - 1) + 1
        return torch.sum(torch.pow((X_sample - X) * B, 2))

    loss = get_1st_loss(X, z_sample) + cita * get_2nd_loss(X, X_sample)

    return loss


####################
# Train procedure
####################
def pretrain(P, Q, P_decoder, Q_encoder, data_loader):
    '''
    Train procedure for one epoch.
    '''
    TINY = 1e-15
    # Set the networks in train mode (apply dropout when needed)
    Q.train()
    P.train()

    # Loop through the labeled and unlabeled dataset getting one batch of samples from each
    # The batch size has to be a divisor of the size of the dataset or it will return
    # invalid samples
    for batch_idx, (X, target) in enumerate(data_loader):

        # Load batch and normalize samples to be between 0 and 1
        # X = X * 0.3081 + 0.1307
        X.resize_(train_batch_size, X_dim)
        X, target = Variable(X), Variable(target)
        if cuda:
            X, target = X.cuda(), target.cuda()
        if data_name == 'wine_network':
            X = torch.tensor(X, dtype=torch.float32)
        # Init gradients
        P.zero_grad()
        Q.zero_grad()

        #######################
        # Reconstruction phase
        #######################

        z_sample = Q(X)
        X_sample = P(z_sample)

        recon_loss = make_loss(X, z_sample, X_sample, batch_idx)
        recon_loss.backward()
        P_decoder.step()
        Q_encoder.step()

        P.zero_grad()
        Q.zero_grad()

    return recon_loss


def train(dec, dec_optimizer, data_loader):
    dec.train()

    for batch_idx, (X, target) in enumerate(data_loader):

        X.resize_(train_batch_size, X_dim)
        if data_name == 'wine_network':
            X = torch.tensor(X, dtype=torch.float32)
        X, target = Variable(X), Variable(target)

        output = dec(X)
        target_p = dec.target_distribution(output).detach()
        loss_function = nn.KLDivLoss(size_average=False)
        kl_loss = loss_function(output.log(), target_p) / output.shape[0]
        # dec_optimizer.zero_grad()
        kl_loss.backward()
        dec_optimizer.step()
        dec_optimizer.zero_grad()
    return kl_loss


def generate_model(train_labeled_loader, train_unlabeled_loader, valid_loader):
    torch.manual_seed(10)

    if cuda:
        Q = Q_net().cuda()
        P = P_net().cuda()
    else:
        Q = Q_net()
        P = P_net()

    # Set learning rates
    gen_lr = 0.0001
    reg_lr = 0.00005

    # Set optimizators
    P_decoder = optim.Adam(P.parameters(), lr=gen_lr)
    Q_encoder = optim.Adam(Q.parameters(), lr=gen_lr)

    for epoch in range(epochs):
        recon_loss = pretrain(P, Q, P_decoder, Q_encoder, train_unlabeled_loader)

        if epoch % 10 == 0:
            report_loss(epoch, recon_loss)
    print('pretrain done! train dec...')

    save_model(Q, '../data/outputQ/v9noDEC_' + data_name + str(beta) + '_' + str(cita) + '.pkl')

    ######################
    #  k-means
    ######################

    dec = DEC(n_clusters=n_classes, autoencoder=Q, hidden=10, cluster_centers=None, alpha=1.0)
    dec_optimizer = optim.SGD(params=dec.parameters(), lr=0.1, momentum=0.9)

    features = []
    for batch_idx, (X_2, target_2) in enumerate(train_unlabeled_loader):
        if data_name == 'wine_network':
            X_2 = torch.tensor(X_2, dtype=torch.float32)
        features.append(dec.autoencoder(X_2).detach().cpu())
    features = torch.cat(features)
    # ============K-means=======================================
    # features = dec.autoencoder(X).detach()
    kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(features)
    cluster_centers = kmeans.cluster_centers_
    cluster_centers = torch.tensor(cluster_centers, dtype=torch.float)
    dec.clusteringlayer.cluster_centers = torch.nn.Parameter(cluster_centers)
    # =========================================================

    for epoch in range(10):
        kl_loss = train(dec, dec_optimizer, train_unlabeled_loader)
        if epoch % 10 == 0:
            report_kl_loss(epoch, kl_loss)
    # writer.close()
    return Q, P


if __name__ == '__main__':

    data_name_list = ['blog_network', 'new_citeseer_network', 'blogcatalog_network', 'flickr_network']
    for name in data_name_list:
        data_name = name
        for c in range(0, 31):
            cita = c
            beta = 10
            train_labeled_loader, train_unlabeled_loader, valid_loader = load_data()
            Q, P = generate_model(train_labeled_loader, train_unlabeled_loader, valid_loader)
            save_model(Q, '../data/outputQ/v9_' + data_name + str(beta) + '_' + str(cita) + '.pkl')

        for b in range(0, 31):
            cita = 10
            beta = b
            train_labeled_loader, train_unlabeled_loader, valid_loader = load_data()
            Q, P = generate_model(train_labeled_loader, train_unlabeled_loader, valid_loader)
            save_model(Q, '../data/outputQ/v9_' + data_name + str(beta) + '_' + str(cita) + '.pkl')
