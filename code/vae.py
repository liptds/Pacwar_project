import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

from utility import *

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        features = 15
        self.features = features
        # encoder
        hidden = 30
        self.enc1 = nn.Linear(in_features=50*4, out_features=hidden)
        self.enc2 = nn.Linear(in_features=hidden, out_features=features * 2)

        # decoder
        self.dec1 = nn.Linear(in_features=features, out_features=hidden)
        self.dec2 = nn.Linear(in_features=hidden, out_features=50*4)

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling as if coming from the input space
        return sample

    def forward(self, x):
        # encoding
        x = F.relu(self.enc1(x))
        x = self.enc2(x).view(-1, 2, self.features)
        # get `mu` and `log_var`
        mu = x[:, 0, :]  # the first feature values as mean
        log_var = x[:, 1, :]  # the other feature values as variance
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)

        # decoding
        x = F.relu(self.dec1(z))
        reconstruction = torch.sigmoid(self.dec2(x))

        return reconstruction, mu, log_var

    def genrating(self, mu, log_var, n):
        # generate n new data points
        mu = mu.repeat(n,1)
        log_var = log_var.repeat(n,1)
        z = self.reparameterize(mu, log_var)

        # decoding
        x = F.relu(self.dec1(z))
        reconstruction = torch.sigmoid(self.dec2(x))
        return reconstruction


def KLD_loss(mu, logvar):
    """
    This function will add the reconstruction loss (BCELoss) and the
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    """

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return  KLD

def fit(model,optimizer,  data, device ):
    # data [data_size, 50*4]
    criterion = nn.BCELoss(reduction='sum')

    model.train()
    running_loss = 0.0
    index =torch.randperm(data.shape[0])
    data = data[index, :] # shuffle

    data_size = data.shape[0]
    batch_size = 20

    for i in range( 0, int(data_size/batch_size) + 1):
        if( i*batch_size > data_size):
            break
        x = data[i*batch_size:(i+1)*batch_size, :].to(device)

        optimizer.zero_grad()
        reconstruction, mu, logvar = model(x)
        bce_loss = criterion(reconstruction, x)

        loss = bce_loss + KLD_loss( mu, logvar)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss = running_loss/data_size
    return train_loss, mu, logvar
def one_hot_encoding(genes):
    """
    get one-hot encoding
    :param gene:
    :return: shape = [n_gene, 50*4]
    """
    encode = F.one_hot(genes, num_classes = 4 )
    encode = torch.reshape(encode, (-1,50*4))
    return encode
def one_hot_decoding(oh_enc):
    n_gene = oh_enc.shape[0]
    oh_enc = oh_enc.detach().to('cpu')
    oh_enc = torch.reshape(oh_enc, (n_gene,50,4))
    genes = torch.argmax(oh_enc, dim=2) # get genes
    list_of_gene = []
    for i in range(0,n_gene):
        gene = genes[i].numpy()
        gene = [int(x) for x in gene]
        list_of_gene.append(gene)
    return list_of_gene

def evaluation(genes, oppoents):
    def evaluate_with_enemy(gene, opponents):
        sum_s = 0
        num_of_win = 0
        num_of_loss = 0
        for g in opponents:
            a, b = get_score(gene, g)
            sum_s += a
            if (a >= b):
                num_of_win += 1
            else:
                num_of_loss += 1
        return sum_s/num_of_opp, num_of_win, num_of_loss

    num_of_opp = len(oppoents)
    sum_s = 0
    sum_win_ratio = 0

    for gene in genes:
        s, win, loss = evaluate_with_enemy(gene,oppoents)
        sum_s += s
        sum_win_ratio+= win/(win+loss)
        print(f"score = {s},  win ratio = {win/(win+loss)} " )
    print( f"average score = {sum_s/len(genes)} , average win ratio = {sum_win_ratio/len(genes)}")


def read_encode_genes(filename, num_of_gene = 10):
    genes = read_gene_list(filename)
    scores = score_genes(genes)
    scores = sorted(scores)[0:num_of_gene]
    genes = [ genes[x[1]] for x in scores ]
    # pick top gene

    genes = np.asarray(genes)
    genes = torch.tensor( genes , dtype=torch.int64)

    encode = one_hot_encoding(genes).to(torch.float)
    return encode


def score_genes(genes):
    scores = []
    for i in range(0,len(genes)):
        score = get_score_vs_opponents(genes[i], genes)
        scores.append((score,i))
    return scores


def vae_generator(ckpt_path, n, scale = 1):
    # generate n gene from vae
    # scale var to get larger variance
    ckpt = torch.load(ckpt_path)
    model = ckpt['model']

    mu = ckpt['mu']
    logvar = ckpt['logvar']

    mean_mu = torch.mean(mu, dim=0)
    std_mu = torch.std(mu, dim=0)

    mean_logvar = torch.mean(logvar, dim=0)
    std_logvar = torch.std(logvar, dim=0)

    model.eval()

    new_gene = torch.zeros((n, 200))
    for i in range(0, n):
        eps = torch.randn_like(mean_mu)
        mu = eps * std_mu + mean_mu

        eps = torch.randn_like(mean_logvar)
        logvar = eps * std_logvar + mean_logvar + torch.ones_like(mean_logvar) * np.log(scale)

        new_gene[i, :] = model.genrating(mu, logvar, 1)

    new_gene = one_hot_decoding(new_gene)
    print("diversity of vae generated gene")
    mean_std, max_std, std =  evaluate_diversity(new_gene)
    print(f"mean std = {mean_std}, max std = {max_std}")
    return new_gene

if __name__ == '__main__':
    epochs = 600
    lr = 1e-2
    device = 'cuda'
    path = "vae.ckpt"
    train_mode = True
    generation_mode= True
    test_mode = False
    if(train_mode):
        encode_genes = read_encode_genes("gene_pools.txt", num_of_gene = 25) # pick top genes to feed vae
        encode_genes.to(device)

        model = VAE().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        train_loss = []
        for epoch in range(epochs):
            print(f"Epoch {epoch+1} of {epochs}")
            train_epoch_loss,mu,logvar = fit(model,optimizer, encode_genes, device )
            train_loss.append(train_epoch_loss)
            print(f"Train Loss: {train_epoch_loss:.4f}")

        ckpt = {'model': model, 'mu':mu, 'logvar':logvar}
        torch.save(ckpt, path)


    if(generation_mode):

        opp_genes = read_gene_list("test_pools.txt")


        ckpt = torch.load(path)
        model = ckpt['model']

        mu = ckpt['mu']
        logvar = ckpt['logvar']

        mean_mu = torch.mean(mu, dim=0)
        std_mu = torch.std(mu, dim=0)

        mean_logvar = torch.mean(logvar, dim=0)
        std_logvar = torch.std(logvar, dim=0)

        mu = torch.mean( ckpt['mu'], dim=0)
        logvar = torch.mean( ckpt['logvar'], dim=0)

        model.eval()
        n = 30

        new_gene = torch.zeros( (n,200))
        for i in range(0,n):
            eps = torch.randn_like(mean_mu)
            mu = eps*std_mu +mean_mu

            eps = torch.randn_like(mean_logvar)
            logvar = eps * std_logvar + mean_logvar

            new_gene[i,:] = model.genrating(mu, logvar, 1)

        new_gene = one_hot_decoding(new_gene)
        evaluation(new_gene, opp_genes)

        mean_std, max_std, std = evaluate_diversity(new_gene)
        print("mean, max std", mean_std, max_std)

        for gene in new_gene:
            print( "".join([str(x) for x in gene]))


    if(test_mode):

        n = 100
        test_gene = "01300000111000200131131323323323222333313310113310"
        test_gene = [int(x) for x in test_gene]
        opp_genes = read_gene_list("test_pools.txt")

        ckpt = torch.load(path)
        model = ckpt['model']
        mu = torch.mean(ckpt['mu'], dim=0)
        logvar = torch.mean(ckpt['logvar'], dim=0)

        model.eval()

        new_gene = model.genrating(mu, logvar, n)
        new_gene = one_hot_decoding(new_gene)
        # evaluation(new_gene, opp_genes)
        evaluation([test_gene], new_gene)

