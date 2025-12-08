# @title masks
import torch
import numpy as np
import matplotlib.pyplot as plt

# def multiblock(seq, min_s, max_s, M=1):
#     mask_len = torch.rand(1) * (max_s - min_s) + min_s # in (min_s, max_s) # all blocks same size
#     mask_pos = torch.rand(M) * (1 - mask_len) # in (0, 1 - mask_len)
#     mask_len, mask_pos = (mask_len * seq).int(), mask_pos * seq
#     indices = torch.arange(seq).unsqueeze(0) # [1, seq]
#     target_mask = (indices >= mask_pos.unsqueeze(-1)) & (indices < (mask_pos + mask_len).unsqueeze(-1)) # [M, seq]
#     return target_mask


# def multiblock2d(hw=(8,8), scale=(.15,.2), aspect_ratio=(.75,1.5), M=1):
#     mask_aspect = torch.rand(1) * (aspect_ratio[1] - aspect_ratio[0]) + aspect_ratio[0] # in (min_s, max_s) # all blocks same size
#     mask_scale = torch.rand(1) * (scale[1] - scale[0]) + scale[0] # in (min_s, max_s) # all blocks same size
#     h = (mask_scale/mask_aspect)**.5# h*(h*aspect) = scale
#     w = h * mask_aspect
#     h_pos, w_pos = torch.rand(M)*(1-w), torch.rand(M)*(1-h) # in (0, 1 - mask_len)
#     h_len, h_pos = (h*hw[0]).int(), h_pos*hw[0]
#     w_len, w_pos = (w*hw[1]).int(), w_pos*hw[1]
#     h_ind, w_ind = torch.arange(hw[0]).unsqueeze(0), torch.arange(hw[1]).unsqueeze(0) # [1, seq]
#     h_mask = (h_ind>=h_pos.unsqueeze(-1)) & (h_ind<(h_pos+h_len).unsqueeze(-1)) # [M, seq]
#     w_mask = (w_ind>=w_pos.unsqueeze(-1)) & (w_ind<(w_pos+w_len).unsqueeze(-1)) # [M, seq]
#     target_mask = h_mask.unsqueeze(-1) & w_mask.unsqueeze(-2) # [M, seq, seq]
#     return target_mask

# # https://arxiv.org/pdf/2210.07224
# def randpatch(seq, mask_size=8, gamma=0.9): # num patches of seq, mask patch size, masking ratio
#     # mask = torch.rand(seq//mask_size)<gamma
#     length = seq//mask_size
#     g = torch.normal(gamma, std=.1, size=(1,)).clamp(.5,.9)
#     # g = gamma
#     idx = torch.randperm(length)[:int(length*g)]
#     mask = torch.zeros(length, dtype=bool)
#     mask[idx] = True
#     mask = mask.repeat_interleave(mask_size, dim=-1)
#     return mask # [seq] , True -> mask


# import torch
# def apply_masks(x, mask): # [b,t,d], [mask_size] # https://github.com/facebookresearch/ijepa/blob/main/src/masks/utils.py
#     mask_keep = mask.unsqueeze(-1).repeat(x.size(0), 1, x.size(-1)) # [batch,T,dim]
#     return torch.gather(x, dim=1, index=mask_keep) # [batch,mask_size,dim]





# @title ijepa multiblock next
# https://github.com/facebookresearch/ijepa/blob/main/src/masks/multiblock.py
import math
from multiprocessing import Value
import torch

class MaskCollator(object):
    def __init__(self, hw=(224, 224), enc_mask_scale=(.85,1), pred_mask_scale=(.15,.2), aspect_ratio=(.75,1.25),
        nenc=1, npred=2, min_keep=4, allow_overlap=False):
        super().__init__()
        self.height, self.width = hw
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep  # minimum number of patches to keep
        self.allow_overlap = allow_overlap  # whether to allow overlap b/w enc and pred masks

    def _sample_block_size(self, scale, aspect_ratio_scale):
        _rand = torch.rand(1).item()
        # -- Sample block scale
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale) # num patches to keep
        # -- Sample block aspect-ratio
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)
        # -- Compute block height and width (given scale and aspect-ratio)
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        while h >= self.height: h -= 1 # crop mask to be smaller than img
        while w >= self.width: w -= 1
        return (h, w)

    def _sample_block_mask(self, b_size, acceptable_regions=None):
        h, w = b_size
        def constrain_mask(mask, tries=0):
            """ Helper to restrict given mask to a set of acceptable regions """
            N = max(int(len(acceptable_regions)-tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]
        # -- Loop to sample masks until we find a valid one
        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            # -- Sample block top-left corner
            top = torch.randint(0, self.height - h, (1,))
            left = torch.randint(0, self.width - w, (1,))
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top:top+h, left:left+w] = 1
            # -- Constrain mask to a set of acceptable regions
            if acceptable_regions is not None:
                constrain_mask(mask, tries)
            mask = torch.nonzero(mask.flatten())
            # -- If mask too small try again
            valid_mask = len(mask) > self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
        mask = mask.squeeze()
        # --
        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        mask_complement[top:top+h, left:left+w] = 0
        # --
        return mask, mask_complement

    def __call__(self, B):
        '''
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample enc block (size + location) using seed
        # 2. sample pred block (size) using seed
        # 3. sample several enc block locations for each image (w/o seed)
        # 4. sample several pred block locations for each image (w/o seed)
        # 5. return enc mask and pred mask
        '''
        p_size = self._sample_block_size(scale=self.pred_mask_scale, aspect_ratio_scale=self.aspect_ratio)
        e_size = self._sample_block_size(scale=self.enc_mask_scale, aspect_ratio_scale=(1., 1.))

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = self.height * self.width
        min_keep_enc = self.height * self.width
        for _ in range(B):

            masks_p, masks_C = [], []
            for _ in range(self.npred):
                mask, mask_C = self._sample_block_mask(p_size)
                masks_p.append(mask)
                masks_C.append(mask_C)
                min_keep_pred = min(min_keep_pred, len(mask))
            collated_masks_pred.append(masks_p)

            acceptable_regions = masks_C
            try:
                if self.allow_overlap:
                    acceptable_regions= None
            except Exception as e:
                print(f'Encountered exception in mask-generator {e}')

            masks_e = []
            for _ in range(self.nenc):
                mask, _ = self._sample_block_mask(e_size, acceptable_regions=acceptable_regions)
                masks_e.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))
            collated_masks_enc.append(masks_e)
        collated_masks_pred = [[cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_pred]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        # --
        collated_masks_enc = [[cm[:min_keep_enc] for cm in cm_list] for cm_list in collated_masks_enc]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)
        return collated_masks_enc, collated_masks_pred

mask_collator = MaskCollator(hw=(32,32), enc_mask_scale=(.85, 1.), pred_mask_scale=(.15, .2), aspect_ratio=(.75, 1.5),
        nenc=1, npred=4, min_keep=4,
        # allow_overlap=True)
        allow_overlap=False)

b=16
collated_masks_enc, collated_masks_pred = mask_collator(b)
ctx_index, trg_index = torch.stack(collated_masks_enc).squeeze(0), torch.stack(collated_masks_pred).transpose(0,1).flatten(1).unique(dim=1) # [num_msk, b,num_tok]->[b,num_tok] # [64, 65], [64, 32]

# mask = torch.zeros(1 ,32*32)
# mask[:, trg_index[:1]] = 1
# mask[:, ctx_index[:1]] = .5
# mask = mask.reshape(1,32,32)

import numpy as np
import matplotlib.pyplot as plt
def imshow(img):
    npimg = img.numpy()
    plt.rcParams["figure.figsize"] = (8,8)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
# imshow(mask)

mask = torch.zeros(b ,32*32)
mask[torch.arange(b).unsqueeze(-1), trg_index] = 1
mask[torch.arange(b).unsqueeze(-1), ctx_index] = .5
mask = mask.reshape(b,1,32,32)
import torchvision
imshow(torchvision.utils.make_grid(mask, nrow=8))

# mask_collator = MaskCollator(hw=(1024,1024), enc_mask_scale=(.85, 1.), pred_mask_scale=(.15, .2), aspect_ratio=(.75, 1.5), nenc=1, npred=4, min_keep=4, allow_overlap=False)
# # %timeit collated_masks_enc, collated_masks_pred = mask_collator(64) # 225 ms 1024:4.79 s
# # %timeit ctx_index, trg_index = simplexmask2d(hw=(1024,1024), ctx_scale=(.85,1), trg_scale=(.5,.6), B=b, chaos=.5) # 265 ms ;topk 203 ms 1024:4.27 s
