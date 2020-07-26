import math
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from bbblayer.BBBdistributions import Normal, distribution_selector
from torch.nn.modules.utils import _pair


class BBBLinearFactorial(nn.Module):
    """
    Describes a Linear fully connected Bayesian layer with
    a distribution over each of the weights and biases
    in the layer.
    """

    def __init__(self, in_features, out_features, p_logvar_init=0.05, q_logvar_init=0.05):
        # p_logvar_init can be either
        # (list/tuples): prior model is a Gaussian distribution
        # q_logvar_init: float, the approximate posterior is currently always a factorized gaussian

        super(BBBLinearFactorial, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.p_logvar_init = p_logvar_init
        self.q_logvar_init = q_logvar_init

        # Approximate posterior weights...
        self.fc_qw_mean = Parameter(torch.Tensor(out_features, in_features))
        self.fc_qw_logvar = Parameter(torch.Tensor(out_features, in_features))
        self.weight = Normal(mu=self.fc_qw_mean, logvar=self.fc_qw_logvar)

        self.register_buffer('eps_weight', torch.Tensor(out_features, in_features))

        # initialise
        self.log_alpha = Parameter(torch.Tensor(1, 1))

        # prior model
        self.pw = Normal(mu=0.0, logvar=p_logvar_init)

        # initialize all paramaters
        self.reset_parameters()

    def reset_parameters(self):
        # initialize (trainable) approximate posterior parameters
        # stdv = 10. / math.sqrt(self.in_features)

        stdv = 1.0 / math.sqrt(self.fc_qw_mean.size(1))

        self.fc_qw_mean.data.uniform_(-stdv, stdv)
        self.fc_qw_logvar.data.fill_(self.q_logvar_init)
        self.log_alpha.data.uniform_(-stdv, stdv)

        self.eps_weight.data.zero_()

    def forward(self, input):
        raise NotImplementedError()

    def fcprobforward(self, input):
        """
        Probabilistic forwarding method.
        :param input: data tensor
        :return: output, kl-divergence
        """

        # sig_weight = torch.exp(self.fc_qw_std)
        # weight = self.fc_qw_mean + sig_weight * self.eps_weight.normal_()

        weight = self.weight.sample()

        fc_qw_mean = F.linear(input=input, weight=weight)
        fc_qw_std = torch.sqrt(1e-8 + F.linear(input=input.pow(2), weight=torch.exp(self.log_alpha) * weight.pow(2)))

        if cuda:
            fc_qw_mean.cuda()
            fc_qw_std.cuda()

        # sample from output
        if cuda:
            # output = fc_qw_mean + fc_qw_si * (torch.randn(fc_qw_mean.size())).cuda()
            output = fc_qw_mean + fc_qw_std * torch.cuda.FloatTensor(fc_qw_mean.size()).normal_()
        else:
            output = fc_qw_mean + fc_qw_std * (torch.randn(fc_qw_mean.size()))

        if cuda:
            output.cuda()

        # self.fc_qw_mean = Parameter(torch.Tensor(fc_qw_mean.cpu()))
        # self.fc_qw_std = Parameter(torch.Tensor(fc_qw_std.cpu()))

        fc_qw = Normal(mu=fc_qw_mean, logvar=fc_qw_std)

        w_sample = fc_qw.sample()

        # KL divergence
        qw_logpdf = fc_qw.logpdf(w_sample)

        kl = torch.sum(qw_logpdf - self.pw.logpdf(w_sample))

        return output, kl

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class BBB3FC(nn.Module):
    """
    Simple Neural Network having 3 FC layers with Bayesian layers.
    """
    def __init__(self):
        super(BBB3FC, self).__init__()

        self.fc1 = BBBLinearFactorial(2, 2)
        self.soft1 = nn.Sigmoid()

        self.fc2 = BBBLinearFactorial(2, 2)
        self.soft2 = nn.Sigmoid()

        self.fc3 = BBBLinearFactorial(2, 1)

        layers = [ self.fc1, self.soft1,
                   self.fc2, self.soft2, 
                   self.fc3]

        self.layers = nn.ModuleList(layers)

    def probforward(self, x):
        'Forward pass with Bayesian weights'
        kl = 0
        for layer in self.layers:
            if hasattr(layer, 'fcprobforward') and callable(layer.fcprobforward):
                x, _kl, = layer.fcprobforward(x)
                kl += _kl
            else:
                x = layer(x)
        logits = x
        return logits, kl