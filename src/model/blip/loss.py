import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    """
    Hard Negative NCE loss for contrastive learning.
    """

    def __init__(self, **kwargs):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, tar_img_feat: torch.Tensor, query_feat: torch.Tensor, temp):
        device = tar_img_feat.device

        sim_t2q = tar_img_feat @ query_feat.T / temp
        sim_q2t = query_feat @ tar_img_feat.T / temp

        bs = sim_t2q.size(0)
        loss_t2q = F.cross_entropy(sim_t2q, torch.arange(bs, device=device))
        loss_q2t = F.cross_entropy(sim_q2t, torch.arange(bs, device=device))

        return (loss_t2q + loss_q2t) / 2


class HardNegativeNCE(nn.Module):
    """
    Hard-Negative NCE loss for contrastive learning.
    https://arxiv.org/pdf/2301.02280.pdf
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.0, **kwargs):
        """
        Args:
            alpha: rescaling factor for positiver terms
            beta: concentration parameter

        Note:
            alpha = 1 and beta = 0 corresponds to the original Info-NCE loss
        """
        super(HardNegativeNCE, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(
        self,
        video_embds: torch.Tensor,
        text_embds: torch.Tensor,
        temp,
    ):
        """
        Args:
            video_embds: (batch_size, video_embd_dim)
            text_embds: (batch_size, text_embd_dim)
        """
        batch_size = video_embds.size(0)
        # computation of the similarity matrix
        sim_matrix = video_embds @ text_embds.T  # (batch_size, batch_size)
        # scale the similarity matrix with the temperature
        sim_matrix = sim_matrix / temp
        sim_matrix = sim_matrix.float()

        nominator = torch.diagonal(sim_matrix)

        beta_sim = self.beta * sim_matrix
        w_v2t = (
            (batch_size - 1)
            * torch.exp(beta_sim)
            / (torch.exp(beta_sim).sum(dim=1) - torch.exp(torch.diagonal(beta_sim)))
        )
        w_t2v = (
            (batch_size - 1)
            * torch.exp(beta_sim)
            / (torch.exp(beta_sim).sum(dim=0) - torch.exp(torch.diagonal(beta_sim)))
        )
        # replace the diagonal terms of w_v2t and w_t2v with alpha
        w_v2t[range(batch_size), range(batch_size)] = self.alpha
        w_t2v[range(batch_size), range(batch_size)] = self.alpha

        denominator_v2t = torch.log((torch.exp(sim_matrix) * w_v2t).sum(dim=1))
        denominator_t2v = torch.log((torch.exp(sim_matrix) * w_t2v).sum(dim=0))

        hn_nce_loss = (denominator_v2t - nominator).mean() + (
            denominator_t2v - nominator
        ).mean()
        return hn_nce_loss
