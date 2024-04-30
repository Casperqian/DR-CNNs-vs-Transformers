import torch
from torch import nn
from typing import List
import torch.nn.functional as F

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.float().softmax(1) * x.float().log_softmax(1)).sum(1)


class BatchEnsemble(nn.Module):
    def __init__(self, indim, outdim, ensemble_size, init_mode):
        super().__init__()
        self.ensemble_size = ensemble_size

        self.in_features = indim
        self.out_features = outdim

        # register parameters
        self.register_parameter(
            "weight", nn.Parameter(
                torch.Tensor(self.out_features, self.in_features)
            )
        )
        self.register_parameter(
            "bias", nn.Parameter(
                torch.Tensor(self.out_features)
            )
        )

        self.register_parameter(
            "alpha_be", nn.Parameter(
                torch.Tensor(self.ensemble_size, self.in_features)
            )
        )
        self.register_parameter(
            "gamma_be", nn.Parameter(
                torch.Tensor(self.ensemble_size, self.out_features)
            )
        )

        use_ensemble_bias = True
        if use_ensemble_bias and self.bias is not None:
            delattr(self, "bias")
            self.register_parameter("bias", None)
            self.register_parameter(
                "ensemble_bias", nn.Parameter(
                    torch.Tensor(self.ensemble_size, self.out_features)
                )
            )
        else:
            self.register_parameter("ensemble_bias", None)

        # initialize parameters
        self.init_mode = init_mode
        self.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D1 = x.size()

        r_x = x.unsqueeze(0).expand(self.ensemble_size, B, D1)  #
        r_x = r_x.view(self.ensemble_size, -1, D1)

        r_x = r_x * self.alpha_be.view(self.ensemble_size, 1, D1)
        r_x = r_x.view(-1, D1)

        w_r_x = nn.functional.linear(r_x, self.weight, self.bias)

        _, D2 = w_r_x.size()
        s_w_r_x = w_r_x.view(self.ensemble_size, -1, D2)
        s_w_r_x = s_w_r_x * self.gamma_be.view(self.ensemble_size, 1, D2)
        if self.ensemble_bias is not None:
            s_w_r_x = s_w_r_x + self.ensemble_bias.view(self.ensemble_size, 1, D2)
        s_w_r_x = s_w_r_x.view(-1, D2)

        return s_w_r_x  # shape = (ensemble_size*B,D2)

    def reset(self):
        init_details = [0, 1]
        initialize_tensor(self.weight, self.init_mode, init_details)
        initialize_tensor(self.alpha_be, self.init_mode, init_details)
        initialize_tensor(self.gamma_be, self.init_mode, init_details)
        if self.ensemble_bias is not None:
            initialize_tensor(self.ensemble_bias, "zeros")
        if self.bias is not None:
            initialize_tensor(self.bias, "zeros")


def initialize_tensor(
        tensor: torch.Tensor,
        initializer: str,
        init_values: List[float] = [],
) -> None:
    if initializer == "zeros":
        nn.init.zeros_(tensor)

    elif initializer == "ones":
        nn.init.ones_(tensor)

    elif initializer == "uniform":
        nn.init.uniform_(tensor, init_values[0], init_values[1])

    elif initializer == "normal":
        nn.init.normal_(tensor, init_values[0], init_values[1])

    elif initializer == "random_sign":
        with torch.no_grad():
            tensor.data.copy_(
                2.0 * init_values[1] * torch.bernoulli(
                    torch.zeros_like(tensor) + init_values[0]
                ) - init_values[1]
            )
    elif initializer == 'xavier_normal':
        torch.nn.init.xavier_normal_(tensor)

    elif initializer == 'kaiming_normal':
        torch.nn.init.kaiming_normal_(tensor)
    else:
        raise NotImplementedError(
            f"Unknown initializer: {initializer}"
        )

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, args):
        super(Algorithm, self).__init__()

    def update(self, minibatches, unlabeled=None):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

class TTFA(Algorithm):
    '''Test-Time Classifier Adjustment Module for Model-Agnostic Domain Generalization & Test-Time Adaptation via Self-Training with Nearest Neighbor Information'''
    def __init__(self, featurizer, classifier, n_outputs, args, filter_K = 100, steps = 3, num_ensemble = 5, lr = 1e-3,
                 k = 8, tau = 10,  num_classes = 5):
        super(FAN, self).__init__(args)
        self.featurizer = featurizer
        self.classifier = classifier
        self.n_outputs = n_outputs

        warmup_supports = self.classifier.weight.data
        self.warmup_supports = warmup_supports
        warmup_prob = self.classifier(self.warmup_supports)
        self.warmup_labels = torch.nn.functional.one_hot(warmup_prob.argmax(1), num_classes=num_classes)
        self.warmup_ent = softmax_entropy(warmup_prob)
        self.ent = self.warmup_ent.data

        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data

        self.filter_K = filter_K
        self.steps = steps
        self.num_ensemble = num_ensemble
        self.lr = lr
        self.k = k
        self.tau = tau
        self.init_mode = 'kaiming_normal'
        self.num_classes = num_classes
        self.mlps = BatchEnsemble(self.n_outputs, self.n_outputs // 4, self.num_ensemble,
                                  self.init_mode).cuda()
        self.optimizer = torch.optim.Adam(self.mlps.parameters(), lr=self.lr, eps=1e-8)

    def forward(self, x, adapt=False):
        # z = self.featurizer(x)
        z = x

        if adapt:
            p_supports = self.classifier(z)
            yhat = torch.nn.functional.one_hot(p_supports.argmax(1), num_classes=self.num_classes)
            ent = softmax_entropy(p_supports)

            # prediction
            self.supports = self.supports.to(z.device)
            self.labels = self.labels.to(z.device)
            self.ent = self.ent.to(z.device)
            self.supports = torch.cat([self.supports, z])
            self.labels = torch.cat([self.labels, yhat])
            #             print(f'self.labels:{self.labels}')
            self.ent = torch.cat([self.ent, ent])

        supports, labels = self.select_supports()

        for _ in range(self.steps):
            p = self.forward_and_adapt(z, supports, labels)
        return p

    def compute_logits(self, z, supports, labels, mlp):
        '''
        :param z: unlabeled test examples
        :param supports: support examples
        :param labels: labels of support examples
        :param mlp: multiple projection heads
        :return: classification logits of z
        '''
        B, dim = z.size()
        N, dim_ = supports.size()

        mlp_z = mlp(z)  # (B*ensemble_size,dim//4)
        mlp_supports = mlp(supports)  # (N*ensemble_size,dim_//4)
        assert (dim == dim_)

        logits = torch.zeros(self.num_ensemble, B, self.num_classes).to(z.device)
        for ens in range(self.num_ensemble):
            temp_centroids = (labels / (labels.sum(dim=0, keepdim=True) + 1e-8)).T @ mlp_supports[
                                                                                     ens * N: (ens + 1) * N]

            # normalize
            temp_z = torch.nn.functional.normalize(mlp_z[ens * B: (ens + 1) * B], dim=1)
            temp_centroids = torch.nn.functional.normalize(temp_centroids, dim=1)
            logits[ens] = self.tau * temp_z @ temp_centroids.T  # [B,C]

        return logits

    def select_supports(self):
        '''
        we filter support examples with high prediction entropy
        :return: filtered support examples.
        '''
        ent_s = self.ent
        y_hat = self.labels.argmax(dim=1).long()
        filter_K = self.filter_K
        if filter_K == -1:
            indices = torch.LongTensor(list(range(len(ent_s))))
        else:
            indices = []
            indices1 = torch.LongTensor(list(range(len(ent_s)))).to(ent_s.device)
            for i in range(self.num_classes):
                _, indices2 = torch.sort(ent_s[y_hat == i])
                indices.append(indices1[y_hat == i][indices2][:filter_K])
            indices = torch.cat(indices)

        self.supports = self.supports[indices]
        self.labels = self.labels[indices]
        self.ent = self.ent[indices]

        return self.supports, self.labels

    @torch.enable_grad()
    def forward_and_adapt(self, z, supports, labels):
        # targets : pseudo labels, outputs: for prediction

        with torch.no_grad():
            targets, outputs = self.target_generation(z, supports, labels)

        self.optimizer.zero_grad()

        loss = None
        logits = self.compute_logits(z, supports, labels, self.mlps)

        import time

        for ens in range(self.num_ensemble):
            if loss is None:

                loss = F.kl_div(logits[ens].float().log_softmax(-1), targets[ens].float(), reduction='batchmean')


            else:
                loss += F.kl_div(logits[ens].float().log_softmax(-1), targets[ens].float(), reduction='batchmean')

        loss.backward()

        self.optimizer.step()

        return outputs  # outputs

    def reset(self):
        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data

        self.mlps.reset()
        self.optimizer = torch.optim.Adam(self.mlps.parameters(), lr=self.lr)

        torch.cuda.empty_cache()

    # from https://jejjohnson.github.io/research_journal/snippets/numpy/euclidean/
    def euclidean_distance_einsum(self, X, Y):
        # X, Y [n, dim], [m, dim] use_featurer_cache > [n,m]
        XX = torch.einsum('nd, nd->n', X, X)[:, None]  # [n, 1]
        YY = torch.einsum('md, md->m', Y, Y)  # [m]
        XY = 2 * torch.matmul(X, Y.T)  # [n,m]
        return XX + YY - XY

    def cosine_distance_einsum(self, X, Y):
        # X, Y [n, dim], [m, dim] -> [n,m]
        X = F.normalize(X, dim=1)
        Y = F.normalize(Y, dim=1)
        XX = torch.einsum('nd, nd->n', X, X)[:, None]  # [n, 1]
        YY = torch.einsum('md, md->m', Y, Y)  # [m]
        XY = 2 * torch.matmul(X, Y.T)  # [n,m]
        return XX + YY - XY

    def target_generation(self, z, supports, labels):

        # retrieve k nearest neighbors. from "https://github.com/csyanbin/TPN/blob/master/train.py"
        dist = self.cosine_distance_einsum(z, supports)
        W = torch.exp(-dist)  # [B, N]

        temp_k = self.filter_K if self.filter_K != -1 else supports.size(0) // self.num_classes
        k = min(self.k, temp_k)

        if k < W.shape[1]:
            pass
        else:
            k = W.shape[1]
        values, indices = torch.topk(W, k, sorted=False)  # [B, k]
        topk_indices = torch.zeros_like(W).scatter_(1, indices, 1)  # [B, N] 1 for topk, 0 for else
        temp_labels = self.compute_logits(supports, supports, labels, self.mlps)  # [ens, N, C]
        temp_labels_targets = F.one_hot(temp_labels.argmax(-1), num_classes=self.num_classes).float()  # [ens, N, C]
        temp_labels_outputs = torch.softmax(temp_labels, -1)  # [ens, N, C]
        topk_indices = topk_indices.unsqueeze(0).repeat(self.num_ensemble, 1, 1)  # [ens, B, N]

        # targets for pseudo labels. we use one-hot class distribution
        targets = torch.bmm(topk_indices, temp_labels_targets)  # [ens, B, C]
        targets = targets / (targets.sum(2, keepdim=True) + 1e-12)  # [ens, B, C]
        # targets = targets.mean(0)  # [B,C]

        # outputs for prediction

        outputs = torch.bmm(topk_indices, temp_labels_outputs)  # [ens, B, C]
        outputs = outputs / (outputs.sum(2, keepdim=True) + 1e-12)  # [ens, B, C]
        outputs = outputs.mean(0)  # [B,C]

        return targets, outputs
