import math
import torch
import torch.nn.functional as F
import torch.nn as nn



class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature=0.1):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="mean")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        return loss


class contrastive_loss(nn.Module):
    def __init__(self, tau=0.1, normalize=True):
        super(contrastive_loss, self).__init__()
        self.tau = tau
        self.normalize = normalize

    def forward(self, xi, xj):

        x = torch.cat((xi, xj), dim=0)

        is_cuda = x.is_cuda
        sim_mat = torch.mm(x, x.T)
        if self.normalize:
            sim_mat_denom = torch.mm(torch.norm(x, dim=1).unsqueeze(1), torch.norm(x, dim=1).unsqueeze(1).T)
            sim_mat = sim_mat / sim_mat_denom.clamp(min=1e-16)

        sim_mat = torch.exp(sim_mat / self.tau)

        # no diag because it's not diffrentiable -> sum - exp(1 / tau)
        # diag_ind = torch.eye(xi.size(0) * 2).bool()
        # diag_ind = diag_ind.cuda() if use_cuda else diag_ind

        # sim_mat = sim_mat.masked_fill_(diag_ind, 0)

        # top
        if self.normalize:
            sim_mat_denom = torch.norm(xi, dim=1) * torch.norm(xj, dim=1)
            sim_match = torch.exp(torch.sum(xi * xj, dim=-1) / sim_mat_denom / self.tau)
        else:
            sim_match = torch.exp(torch.sum(xi * xj, dim=-1) / self.tau)

        sim_match = torch.cat((sim_match, sim_match), dim=0)

        norm_sum = torch.exp(torch.ones(x.size(0)) / self.tau)
        norm_sum = norm_sum.cuda() if is_cuda else norm_sum
        loss = torch.mean(-torch.log(sim_match / (torch.sum(sim_mat, dim=-1) - norm_sum)))

        return loss


class TC(nn.Module):
    def __init__(self, backbone, timesteps, embedding_dim, num_channels):
        super(TC, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_channels = num_channels
        self.timestep = timesteps
        self.Wk = nn.ModuleList([nn.Linear(self.num_channels, self.embedding_dim) for i in range(self.timestep)])
        self.lsoftmax = nn.LogSoftmax()
        self.backbone = backbone
        # self.backbone = Widar_CNN_BiLSTM(self.embedding_dim, self.num_classes)

    def forward(self, x_1, x_2):
        ori_feas_1, out_feas_1, outputs_1 = self.backbone(x_1)
        ori_feas_2, out_feas_2, outputs_2 = self.backbone(x_2)
        z_aug1 = ori_feas_1
        seq_len = z_aug1.shape[1]
        z_aug2 = ori_feas_2
        batch = z_aug1.shape[0]
        # randomly pick time stamps
        t_samples = torch.randint(seq_len - self.timestep, size=(1,)).long()
        print(t_samples)

        nce = 0  # average over timestep and batch
        encode_samples = torch.empty((self.timestep, batch, self.embedding_dim)).float()

        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = z_aug2[:, t_samples + i, :].view(batch, self.embedding_dim)

        c_t = out_feas_1[:, :t_samples + 1, :]
        print(c_t.shape)

        pred = torch.empty((self.timestep, batch, self.embedding_dim)).float()
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t[:, -1, :])
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
            nce += torch.sum(torch.diag(self.lsoftmax(total)))
        nce /= -1. * batch * self.timestep
        return nce, outputs_1, outputs_2


class ClusterLoss(nn.Module):
    def __init__(self, class_num, temperature, device):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss + ne_loss



class CM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def cm(inputs, indexes, features, momentum=0.5):
    return CM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM_Hard(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None


def cm_hard(inputs, indexes, features, momentum=0.5):
    return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard

        self.register_buffer('features', torch.zeros(num_samples, num_features))

    def forward(self, inputs, targets):

        inputs = F.normalize(inputs, dim=1).cuda()
        if self.use_hard:
            outputs = cm_hard(inputs, targets, self.features, self.momentum)
        else:
            outputs = cm(inputs, targets, self.features, self.momentum)

        outputs /= self.temp
        loss = F.cross_entropy(outputs, targets)
        return loss


class CrossViewCentroidContrastiveLoss(nn.Module):
    """
    Calculates the contrastive loss between centroids from two different views.

    For each sample in the batch, it finds its corresponding centroid in view 1
    and view 2, and pulls them together.
    """

    def __init__(self, temperature=0.1):
        super(CrossViewCentroidContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, centroids1, centroids2, labels1, labels2):
        """
        Args:
            centroids1 (torch.Tensor): Centroids from view 1 (num_clusters1, feat_dim).
            centroids2 (torch.Tensor): Centroids from view 2 (num_clusters2, feat_dim).
            labels1 (torch.Tensor): Pseudo-labels from view 1 for the current batch (batch_size,).
            labels2 (torch.Tensor): Pseudo-labels from view 2 for the current batch (batch_size,).

        Returns:
            torch.Tensor: The calculated cross-view centroid contrastive loss.
        """
        device = centroids1.device

        valid_mask = (labels1 != -1) & (labels2 != -1)
        labels1_valid = labels1[valid_mask]
        labels2_valid = labels2[valid_mask]

        if len(labels1_valid) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        batch_centroids1 = centroids1[labels1_valid]  # (num_valid_samples, feat_dim)

        batch_centroids1_norm = F.normalize(batch_centroids1, dim=1)
        centroids2_norm = F.normalize(centroids2, dim=1)

        sim_matrix12 = torch.matmul(batch_centroids1_norm, centroids2_norm.T)
        logits12 = sim_matrix12 / self.temperature
        # labels2_valid 告诉我们对于每个 anchor，哪个 target 是正样本
        loss12 = self.cross_entropy_loss(logits12, labels2_valid)

        batch_centroids2 = centroids2[labels2_valid]
        batch_centroids2_norm = F.normalize(batch_centroids2, dim=1)
        centroids1_norm = F.normalize(centroids1, dim=1)

        sim_matrix21 = torch.matmul(batch_centroids2_norm, centroids1_norm.T)
        logits21 = sim_matrix21 / self.temperature
        loss21 = self.cross_entropy_loss(logits21, labels1_valid)

        # 返回对称损失的平均值
        return (loss12 + loss21) / 2
