import torch
import torch.nn as nn
import torch.nn.functional as F


class CompositeSelfSupervisedLoss(nn.Module):

    def __init__(self,
                 lambda_pose=0.5,
                 lambda_infonce=0.3,
                 temperature=0.05,
                 margin=1.0):
        super(CompositeSelfSupervisedLoss, self).__init__()

        self.lambda_pose = lambda_pose
        self.lambda_infonce = lambda_infonce
        self.lambda_contrastive = 1.0 - lambda_pose - lambda_infonce

        assert self.lambda_contrastive >= 0, "Lambda values should sum to <= 1"

        self.temperature = temperature
        self.margin = margin

    def infonce_loss(self, anchor_embed, positive_embed, negative_embeds):
        anchor_norm = F.normalize(anchor_embed, dim=1)
        positive_norm = F.normalize(positive_embed, dim=1)
        negative_norm = F.normalize(negative_embeds, dim=2)

        pos_sim = torch.sum(anchor_norm * positive_norm, dim=1) / self.temperature
        neg_sim = torch.bmm(negative_norm, anchor_norm.unsqueeze(2)).squeeze(2) / self.temperature

        numerator = torch.exp(pos_sim)
        denominator = numerator + torch.sum(torch.exp(neg_sim), dim=1)

        loss = -torch.mean(torch.log(numerator / denominator))

        return loss

    def contrastive_loss(self, anchor_embed, positive_embed, negative_embeds):
        pos_dist = torch.norm(anchor_embed - positive_embed, dim=1) ** 2
        pos_loss = pos_dist.mean()

        neg_dist = torch.norm(
            anchor_embed.unsqueeze(1) - negative_embeds,
            dim=2
        )

        neg_loss = torch.mean(
            torch.clamp(self.margin - neg_dist, min=0) ** 2
        )

        loss = 0.5 * (pos_loss + neg_loss)

        return loss

    def pose_loss(self, pred_keypoints, gt_keypoints):
        loss = torch.mean((pred_keypoints - gt_keypoints) ** 2)
        return loss

    def forward(self,
                anchor_embed,
                anchor_pred,
                positive_embed,
                negative_embeds,
                anchor_keypoints_gt):
        l_infonce = self.infonce_loss(anchor_embed, positive_embed, negative_embeds)
        l_contrastive = self.contrastive_loss(anchor_embed, positive_embed, negative_embeds)
        l_pose = self.pose_loss(anchor_pred, anchor_keypoints_gt)

        total_loss = (self.lambda_pose * l_pose +
                      self.lambda_infonce * l_infonce +
                      self.lambda_contrastive * l_contrastive)

        loss_dict = {
            'total_loss': total_loss.item(),
            'pose_loss': l_pose.item(),
            'infonce_loss': l_infonce.item(),
            'contrastive_loss': l_contrastive.item()
        }

        return total_loss, loss_dict