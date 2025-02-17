"""
This code is based on the implementation by NoviceStone
https://github.com/NoviceStone/VAE/tree/master
"""

import torch
import torch.nn as nn
from torch.distributions import Normal

def tanh_gaussian_indep(mean, in_var):
    """
    Propagates Gaussian moments through Tanh activation
    only propagating variance, ignoring covariance
    """
    EPSILON = 1e-6
    gamma   = torch.tensor([0.715951561820333,1.039358092507381,0.948607106485449,0.715951555650637,
                            0.715951343627955,1.039354251532628,0.484068718800797,0.715951452277779,
                            1.039355768740833,1.446433070133343], dtype=torch.float32, device=mean.device)

    gamma   = gamma.unsqueeze(0)
    
    in_var  = torch.clamp(in_var, min=EPSILON)
    
    in_var_hat      = in_var.unsqueeze(-1) + (1 / (2 * torch.pow(gamma, 2))).unsqueeze(1)
    # sqrt_in_var_hat = torch.sqrt(in_var_hat)
    mean            = mean.unsqueeze(-1)
    # mu_hat          = mean / sqrt_in_var_hat

    standard_normal = Normal(loc=0, scale=in_var_hat )
    
    B  = standard_normal.log_prob(mean).exp()
    C  = standard_normal.cdf(mean)
    
    A0 = 2 * C - 1
    A1 = 2 * B
    A2 = -B * mean / in_var_hat
    A3 = (1 / 3) * B * (torch.pow(mean, 2) - in_var_hat) / torch.pow(in_var_hat, 2)
    
    A0 = torch.mean(A0, dim=-1) # posterior mean
    A1 = torch.mean(A1, dim=-1) 
    A2 = torch.mean(A2, dim=-1)
    A3 = torch.mean(A3, dim=-1)
    
    V1 = in_var
    V2 = 2 * torch.pow(in_var, 2)
    V3 = 15 * torch.pow(in_var, 3)
    
    post_var = torch.pow(A1, 2) * V1 + torch.pow(A2, 2) * V2 + torch.pow(A3, 2) * V3 # posterior var

    del in_var, in_var_hat, mean, B, C, A1, A2, A3, V1, V2, V3
    
    return A0, post_var


def sigmoid_gaussian_indep(mean, in_var):
    """
    Propagates Gaussian moments through Sigmoid activation
    only propagating variance, ignoring covariance
    """
    EPSILON = 1e-6

    gamma   = torch.tensor([0.519483084417772,0.357944855434941,0.723301195883257,0.474918009444542,
                            0.519483382721337,0.357945091498072,0.519481351904163,0.357945544542554,
                            0.242049896394596,0.357944917042638], dtype=torch.float32, device=mean.device)
    
    gamma   = gamma.unsqueeze(0)

    in_var  = torch.clamp(in_var, min=EPSILON)

    in_var_hat      = in_var.unsqueeze(-1) + (1 / (2 * torch.pow(gamma, 2))).unsqueeze(1)
    # sqrt_in_var_hat = torch.sqrt(in_var_hat)
    mean            = mean.unsqueeze(-1)
    # mu_hat          = mean / sqrt_in_var_hat

    standard_normal = Normal(loc=0, scale=in_var_hat)
    
    B  = standard_normal.log_prob(mean).exp()
    C  = standard_normal.cdf(mean)
    
    A0 = C
    A1 = B
    A2 = (-1 / 2) * B * mean / in_var_hat
    A3 = (1 / 6) * B * (torch.pow(mean, 2) - in_var_hat) / torch.pow(in_var_hat, 2)
    
    A0 = torch.mean(A0, dim=-1) # posterior mean
    A1 = torch.mean(A1, dim=-1) 
    A2 = torch.mean(A2, dim=-1)
    A3 = torch.mean(A3, dim=-1)
    
    V1 = in_var
    V2 = 2 * torch.pow(in_var, 2)
    V3 = 15 * torch.pow(in_var, 3)
    
    post_var = torch.pow(A1, 2) * V1 + torch.pow(A2, 2) * V2 + torch.pow(A3, 2) * V3 # posterior var

    del in_var, in_var_hat, mean, B, C, A1, A2, A3, V1, V2, V3
    
    return A0, post_var

def slices_to_diagonal_matrices(A):
    batch_size, hidden_size = A.shape
    # Create a zero tensor of shape (batch_size, hidden_size, hidden_size)
    result = torch.zeros(batch_size, hidden_size, hidden_size, device=A.device)
    # Fill the diagonal of each matrix with the corresponding slice
    # batch_diagonal fills the diagonal for each matrix in the batch
    result.diagonal(dim1=1, dim2=2)[:] = A
    return result

def tanh_gaussian(mean, in_cov):
    """
    Propagates Gaussian moments through Tanh activation
    """
    EPSILON = 1e-6
    gamma   = torch.tensor([0.715951561820333,1.039358092507381,0.948607106485449,0.715951555650637,
                            0.715951343627955,1.039354251532628,0.484068718800797,0.715951452277779,
                            1.039355768740833,1.446433070133343], dtype=torch.float32, device=mean.device)

    gamma   = gamma.unsqueeze(0)
    
    in_var  = in_cov.diagonal(dim1=-2, dim2=-1)
    in_var  = torch.clamp(in_var, min=EPSILON)
    # Create a new covariance matrix with the clamped diagonal
    new_in_cov = in_cov - torch.diag_embed(in_cov.diagonal(dim1=-2, dim2=-1)) + torch.diag_embed(in_var)
    
    in_var_hat      = in_var.unsqueeze(-1) + (1 / (2 * torch.pow(gamma, 2))).unsqueeze(1)
    sqrt_in_var_hat = torch.sqrt(in_var_hat)
    mean            = mean.unsqueeze(-1)
    mu_hat          = mean / sqrt_in_var_hat

    standard_normal = Normal(loc=0, scale=1)
    
    B  = standard_normal.log_prob(mu_hat).exp() / sqrt_in_var_hat
    C  = standard_normal.cdf(mu_hat)
    
    A0 = 2 * C - 1
    A1 = 2 * B
    A2 = -B * mean / in_var_hat
    A3 = (1 / 3) * B * (torch.pow(mean, 2) - in_var_hat) / torch.pow(in_var_hat, 2)
    
    A0 = torch.mean(A0, dim=-1) # posterior mean
    A1 = torch.mean(A1, dim=-1) 
    A2 = torch.mean(A2, dim=-1)
    A3 = torch.mean(A3, dim=-1)
    
    V1 = new_in_cov
    V2 = 2 * torch.pow(new_in_cov, 2)
    V3 = 6 * torch.pow(new_in_cov, 3) + 9 * in_var.unsqueeze(-1) * new_in_cov * in_var.unsqueeze(1)

    # Using einsum to avoid batch dimension involvement in matrix multiplications
    term1 = torch.einsum("bi,bij,bj->bij", A1, V1, A1)
    term2 = torch.einsum("bi,bij,bj->bij", A2, V2, A2)
    term3 = torch.einsum("bi,bij,bj->bij", A3, V3, A3)
    
    post_var = term1 + term2 + term3 # posterior log_var
    del in_var, new_in_cov, in_var_hat, sqrt_in_var_hat, mean, mu_hat, B, C, A1, A2, A3, V1, V2, V3, term1, term2, term3
    return A0, post_var



def sigmoid_gaussian(mean, in_cov):
    """
    Propagates Gaussian moments through Sigmoid activation
    """

    EPSILON = 1e-6

    gamma   = torch.tensor([0.519483084417772,0.357944855434941,0.723301195883257,0.474918009444542,
                            0.519483382721337,0.357945091498072,0.519481351904163,0.357945544542554,
                            0.242049896394596,0.357944917042638], dtype=torch.float32, device=mean.device)
    gamma   = gamma.unsqueeze(0)

    in_var  = in_cov.diagonal(dim1=-2, dim2=-1)
    in_var  = torch.clamp(in_var, min=EPSILON)
    # Create a new covariance matrix with the clamped diagonal
    new_in_cov = in_cov - torch.diag_embed(in_cov.diagonal(dim1=-2, dim2=-1)) + torch.diag_embed(in_var)

    in_var_hat      = in_var.unsqueeze(-1) + (1 / (2 * torch.pow(gamma, 2))).unsqueeze(1)
    sqrt_in_var_hat = torch.sqrt(in_var_hat)
    mean            = mean.unsqueeze(-1)
    mu_hat          = mean / sqrt_in_var_hat

    standard_normal = Normal(loc=0, scale=1)
    
    B  = standard_normal.log_prob(mu_hat).exp() / sqrt_in_var_hat
    C  = standard_normal.cdf(mu_hat)
    
    A0 = C
    A1 = B
    A2 = (-1 / 2) * B * mean / in_var_hat
    A3 = (1 / 6) * B * (torch.pow(mean, 2) - in_var_hat) / torch.pow(in_var_hat, 2)
    
    A0 = torch.mean(A0, dim=-1) # posterior mean
    A1 = torch.mean(A1, dim=-1) 
    A2 = torch.mean(A2, dim=-1)
    A3 = torch.mean(A3, dim=-1)
    
    V1 = new_in_cov
    V2 = 2 * torch.pow(new_in_cov, 2)
    V3 = 6 * torch.pow(new_in_cov, 3) + 9 * in_var.unsqueeze(-1) * new_in_cov * in_var.unsqueeze(1)

    # Using einsum to avoid batch dimension involvement in matrix multiplications
    term1 = torch.einsum("bi,bij,bj->bij", A1, V1, A1)
    term2 = torch.einsum("bi,bij,bj->bij", A2, V2, A2)
    term3 = torch.einsum("bi,bij,bj->bij", A3, V3, A3)
    
    post_var = term1 + term2 + term3 # posterior log_var
    del in_var, new_in_cov, in_var_hat, sqrt_in_var_hat, mean, mu_hat, B, C, A1, A2, A3, V1, V2, V3, term1, term2, term3
    return A0, post_var
    
    



class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, data_type="binary"):
        super(VAE, self).__init__()
        # Encoder: layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc21 = nn.Linear(hidden_size, latent_size)
        self.fc22 = nn.Linear(hidden_size, latent_size)
        # Decoder: layers
        self.fc3 = nn.Linear(latent_size, hidden_size)
        self.fc41 = nn.Linear(hidden_size, input_size)
        self.fc42 = nn.Linear(hidden_size, input_size)
        # data_type: can be "binary" or "real"
        self.data_type = data_type

    def encode(self, x):
        h1 = torch.tanh(self.fc1(x))
        mean, log_var = self.fc21(h1), self.fc22(h1)
        return mean, log_var

    @staticmethod
    def reparameterize(mean, log_var):
        mu, sigma = mean, torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        return z

    def decode(self, z):
        h3 = torch.tanh(self.fc3(z))
        if self.data_type == "real":
            mean, log_var = torch.sigmoid(self.fc41(h3)), self.fc42(h3)
            return mean, log_var
        else:
            logits = self.fc41(h3)
            probs = torch.sigmoid(logits)
            return probs

    def forward(self, x):
        z_mean, z_logvar = self.encode(x)
        z = self.reparameterize(z_mean, z_logvar)
        return z_mean, z_logvar, self.decode(z)

class VAE_EP(VAE):
    def __init__(self, input_size, hidden_size, latent_size, data_type="binary"):
        super(VAE_EP, self).__init__(input_size, hidden_size, latent_size, data_type)

    def decode_EP(self, z_mean, z_var):
        """
        Rewrite decoder w/ expectation propagation
        We take the centers of the propagated Gaussians as final output 
        """
        # linear fc3
        z_mean = self.fc3(z_mean)
        w_fc3  = self.fc3.weight
        z_var  = z_var @ torch.pow(w_fc3, 2).T
        
        # tanh layer
        z_mean_h3, z_var_h3 = tanh_gaussian_indep(z_mean, z_var)

        if self.data_type == "real":
            # linear fc41
            z_mean = self.fc41(z_mean_h3)
            w_fc41 = self.fc41.weight
            z_var  = z_var_h3 @ torch.pow(w_fc41, 2).T

            # sigmoid layer
            mean, _ = sigmoid_gaussian_indep(z_mean, z_var)

            # linear fc42
            log_var = torch.log(self.fc42(z_mean_h3))

            return mean, log_var
        else:
            # linear fc41
            z_mean = self.fc41(z_mean_h3)
            w_fc41 = self.fc41.weight
            z_var  = z_var_h3 @ torch.pow(w_fc41, 2).T

            # sigmoid layer
            probs, _ = sigmoid_gaussian_indep(z_mean, z_var)

            return probs

    def forward(self, x, verbose=False):
        z_mean, z_logvar = self.encode(x)
        z_var = torch.exp(z_logvar)
        if verbose:
            print(f"z_mean: N({torch.mean(z_mean, 0)}, {torch.var(z_mean, 0)})")
            print(f"z_var: N({torch.mean(z_var, 0)}, {torch.var(z_var, 0)})")
        return z_mean, z_logvar, self.decode_EP(z_mean, z_var)


class VAE_EP_fullcov(VAE):
    def __init__(self, input_size, hidden_size, latent_size, data_type="binary"):
        super(VAE_EP_fullcov, self).__init__(input_size, hidden_size, latent_size, data_type)

    def decode_EP(self, z_mean, z_var):
        """
        Rewrite decoder w/ expectation propagation
        We take the centers of the propagated Gaussians as final output 
        """
        # linear fc3
        z_cov  = torch.diag_embed(z_var)
        z_mean = self.fc3(z_mean)
        w_fc3  = self.fc3.weight
        z_cov  = torch.einsum("oi,bij,kj->bok", w_fc3, z_cov, w_fc3)  # Shape: (batch_size, output_size, output_size)

        # tanh layer
        z_mean_h3, z_cov_h3 = tanh_gaussian(z_mean, z_cov)

        if self.data_type == "real":
            # linear fc41
            z_mean = self.fc41(z_mean_h3)
            w_fc41 = self.fc41.weight
            z_cov  = torch.einsum("oi,bij,kj->bok", w_fc41, z_cov_h3, w_fc41)

            # sigmoid layer
            mean, _ = sigmoid_gaussian(z_mean, z_cov)

            # linear fc42
            log_var = torch.log(self.fc42(z_mean_h3))

            return mean, log_var
        else:
            # linear fc41
            z_mean = self.fc41(z_mean_h3)
            w_fc41 = self.fc41.weight
            z_cov  = torch.einsum("oi,bij,kj->bok", w_fc41, z_cov_h3, w_fc41)

            # sigmoid layer
            probs, _ = sigmoid_gaussian(z_mean, z_cov)

            return probs

    def forward(self, x, verbose=False):
        z_mean, z_logvar = self.encode(x)
        z_var = torch.exp(z_logvar)
        if verbose:
            print(f"z_mean: N({torch.mean(z_mean, 0)}, {torch.var(z_mean, 0)})")
            print(f"z_var: N({torch.mean(z_var, 0)}, {torch.var(z_var, 0)})")
        return z_mean, z_logvar, self.decode_EP(z_mean, z_var)