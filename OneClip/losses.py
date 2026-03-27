import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import torch.distributed.nn
from typing import Optional

from einops import rearrange, repeat

class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor):
        output = [torch.empty_like(tensor) for _ in range(utils.get_world_size())]
        torch.distributed.all_gather(output, tensor)
        ctx.rank = utils.get_rank()
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank : ctx.batch_size * (ctx.rank + 1)],
            None,
        )

allgather = AllGather.apply


class INTRACLIPLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None
        self.temperature = temperature

    def forward(self, outputs, args):
        image_embed = outputs['image_embeds']
        text_embed = outputs['text_embeds']
        logit_scale = outputs['logit_scale']
        labels = outputs['labels']

        # normalized features
        image_embed = F.normalize(image_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)

        # gather with gradient
        image_embed_all = allgather(image_embed)
        text_embed_all = allgather(text_embed)
        labels_all = allgather(labels)
        torch.distributed.barrier()

        clip_frames = args.clip_frames
        
        bsz, dim = text_embed_all.shape
        intra_neg_num = args.neg_frames
        video_embeds_all = image_embed_all.view(bsz, clip_frames+intra_neg_num, dim)

        intra_pos_video_embeds = video_embeds_all[labels_all].view(bsz, clip_frames, dim)
        intra_neg_video_embeds = video_embeds_all[~labels_all].view(bsz, intra_neg_num, dim)

        def del_element(index, x):
            return torch.cat((x[:index], x[index+1:]))
        inter_neg_text_embeds = []
        for i in range(bsz):
            temp = del_element(i, text_embed_all) 
            inter_neg_text_embeds.append(temp)
        inter_neg_text_embeds = torch.stack(inter_neg_text_embeds, dim = 0) 

        def l2_norm(x):
            return x/x.norm(dim=-1, keepdim=True)
        
        intra_pos_video_logits = logit_scale * torch.einsum("b w d, b d -> b w", l2_norm(intra_pos_video_embeds), l2_norm(text_embed_all)) 
        intra_neg_video_logits = logit_scale * torch.einsum("b w d, b d -> b w", l2_norm(intra_neg_video_embeds), l2_norm(text_embed_all)) 
        inter_neg_text_logits = logit_scale * torch.einsum("b w d, b o d -> b w o", l2_norm(intra_pos_video_embeds), l2_norm(inter_neg_text_embeds)) 

        loss = 0
        labels = torch.zeros(bsz, dtype=torch.long, device=image_embed.device)
        for i in range(clip_frames):
            logits = torch.cat([
                intra_pos_video_logits[:, i].unsqueeze(1),
                intra_neg_video_logits, 
                inter_neg_text_logits[:, i, :],
            ], dim=1) 
            loss += F.cross_entropy(logits, labels)
        loss = loss / clip_frames
        return {'loss': loss}


class INTERCLIPLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None
        self.temperature = temperature

    def forward(self, outputs, args):
        image_embed = outputs['image_embeds']
        text_embed = outputs['text_embeds']
        logit_scale = outputs['logit_scale']

        # normalized features
        image_embed = F.normalize(image_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)

        image_embed_all = allgather(image_embed)
        text_embed_all = allgather(text_embed)
        torch.distributed.barrier()

        clip_frames = args.clip_frames

        bsz, dim = text_embed_all.shape
        inter_neg_num_per_video = args.clip_frames
        video_embeds_all = image_embed_all.view(bsz, clip_frames, dim)

        def del_element(index, x):
            return torch.cat((x[:index], x[index+1:]))
        shuffle_video_embeds = video_embeds_all[:, torch.randperm(clip_frames),:][:, :inter_neg_num_per_video, :] 
        inter_neg_video_embeds = []
        for i in range(bsz):
            temp = del_element(i, shuffle_video_embeds)
            temp = rearrange(temp, "b w d -> (b w) d")
            inter_neg_video_embeds.append(temp)
        inter_neg_video_embeds = torch.stack(inter_neg_video_embeds, dim=0) 

        inter_neg_text_embeds = []
        for i in range(bsz):
            temp = del_element(i, text_embed_all) 
            inter_neg_text_embeds.append(temp)
        inter_neg_text_embeds = torch.stack(inter_neg_text_embeds, dim = 0) 

        def l2_norm(x):
            return x/x.norm(dim=-1, keepdim=True)
        
        intra_pos_video_logits = logit_scale * torch.einsum("b w d, b d -> b w", l2_norm(video_embeds_all), l2_norm(text_embed_all)) 
        inter_neg_video_logits = logit_scale * torch.einsum("b w d, b d -> b w", l2_norm(inter_neg_video_embeds), l2_norm(text_embed_all)) 
        inter_neg_text_logits = logit_scale * torch.einsum("b w d, b o d -> b w o", l2_norm(video_embeds_all), l2_norm(inter_neg_text_embeds)) 
        loss = 0
        labels = torch.zeros(bsz, dtype=torch.long, device=image_embed.device)
        for i in range(clip_frames):
            logits = torch.cat([
                intra_pos_video_logits[:, i].unsqueeze(1),
                inter_neg_video_logits, 
                inter_neg_text_logits[:, i, :], 
            ], dim=1) 
            loss += F.cross_entropy(logits, labels) 
        loss = loss / clip_frames
        return {'loss': loss}
    