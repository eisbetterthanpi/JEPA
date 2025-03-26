# @title I-JEPA
import torch
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class TransformerPredictor(nn.Module):
    def __init__(self, in_dim, d_model, out_dim=None, d_head=4, d_hid=None, nlayers=1, drop=0.):
        super().__init__()
        self.embed = nn.Linear(in_dim, d_model)# if in_dim != d_model else None
        # self.pos_enc = RotEmb(d_model, top=1, base=10000)
        # self.pos_emb = nn.Parameter(torch.randn(1, 8*8, d_model)*0.02)
        self.pos_emb = nn.Parameter(RoPE2D(dim=d_model, h=8, w=8, base=10000).unsqueeze(0), requires_grad=False)

        self.transformer = nn.Sequential(*[AttentionBlock(d_model, d_head=d_head) for _ in range(nlayers)])

        self.cls = nn.Parameter(torch.randn(1,1,d_model)*0.02) # randn zeros
        out_dim = out_dim or d_model
        self.norm = nn.RMSNorm(d_model) # LayerNorm RMSNorm
        self.lin = nn.Linear(d_model, out_dim)# if out_dim != d_model else None

    def forward(self, x, context_indices, trg_indices): # [batch, seq_len, d_model], [batch, seq_len] # True will be ignored by the attention # https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
        x = self.embed(x) # [batch, seq_len, d_model] or [batch, num_context_toks, d_model]
        batch, seq, dim = x.shape
        # x = x * self.pos_enc(context_indices)
        # print("Trans pred",x.shape, self.positional_emb[:,context_indices].shape)
        x = x + self.pos_emb[:,context_indices]

        # pred_tokens = self.cls * self.pos_enc(trg_indices) # [M, num_trg_toks, d_model]
        # pred_tokens = self.cls + self.positional_emb[0,trg_indices]
        pred_tokens = self.cls + self.pos_emb[:,trg_indices]
        # print("Trans pred",pred_tokens.shape, self.cls.shape, self.positional_emb[:,trg_indices].shape)
        pred_tokens = pred_tokens.repeat(batch, 1, 1) # [batch, num_trg_toks, d_model]
        # print(pred_tokens.requires_grad)
        # print("pred fwd", x.shape, pred_tokens.shape)
        x = torch.cat([x, pred_tokens], dim=1) # [batch, seq_len+num_trg_toks, d_model]
        out = self.transformer(x)

        out = self.norm(out)
        out = out[:,seq:] # [batch, num_trg_toks, d_model]
        out = self.lin(out)
        return out # [seq_len, batch_size, ntoken]


class TransformerModel(nn.Module):
    # def __init__(self, in_dim, d_model, out_dim=None, nhead=8, d_hid=None, nlayers=1, dropout = 0.):
    def __init__(self, in_dim, d_model, out_dim=None, d_head=4, nlayers=1, drop=0.):
        super().__init__()
        patch_size=4
        self.embed = nn.Sequential(
            # # nn.Conv1d(in_dim, d_model,7,2,7//2), nn.MaxPool1d(2,2), #nn.MaxPool1d(3, 2, 3//2),
            # # nn.Conv1d(in_dim, d_model,3,2,3//2), nn.BatchNorm1d(d_model), nn.ReLU(), nn.Conv1d(d_model, d_model,3,2,3//2)
            # nn.Conv1d(in_dim, d_model,3,2,3//2), nn.BatchNorm1d(d_model), nn.ReLU(), nn.MaxPool1d(2,2),
            # nn.Conv1d(d_model, d_model,3,2,3//2), nn.BatchNorm1d(d_model), nn.ReLU(), nn.MaxPool1d(2,2),
            # nn.Conv1d(d_model, d_model,3,2,3//2),
            nn.Conv2d(in_dim, d_model, patch_size, patch_size), # patch
            # nn.Conv2d(in_dim, d_model, kernel_size=7, stride=2, padding=7//2, bias=False), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # nn.Conv2d(in_dim, d_model, 3, 2, 3//2, bias=False), nn.BatchNorm2d(d_model), nn.ReLU(),
            # nn.Conv2d(d_model, d_model, 3, 2, 3//2, bias=False)

            )
        # self.pos_enc = RotEmb(d_model, top=1, base=10000)
        # self.pos_emb = nn.Parameter(torch.randn(1, 8*8, d_model)*0.02)
        self.pos_emb = nn.Parameter(RoPE2D(dim=d_model, h=8, w=8, base=10000).unsqueeze(0), requires_grad=False)
        self.transformer = nn.Sequential(*[AttentionBlock(d_model, d_head=d_head) for _ in range(nlayers)])
        self.norm = nn.RMSNorm(d_model) # LayerNorm RMSNorm
        self.lin = nn.Linear(d_model, out_dim) if out_dim and out_dim != d_model else None

    def forward(self, x, context_indices=None): # [batch, num_context_toks, 3], [batch, num_context_toks] # True will be ignored by the attention # https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
        # x = self.embed(x.transpose(-2,-1)).transpose(-2,-1) # [batch, T, d_model]
        x = self.embed(x).flatten(2).transpose(1,2) # [b,c,h,w]->[b,h*w,c] # [batch, seq_len, d_model] or [batch, num_context_toks, d_model]
        # x = self.pos_enc(x)
        x = x + self.pos_emb
        if context_indices != None: x = x[:,context_indices]
        # print("TransformerModel",x.shape)

        x = self.transformer(x)
        out = self.norm(x)
        if self.lin: out = self.lin(out)
        return out


batch, seq_len, d_model = 4,1024,16
in_dim = 3
model = TransformerModel(in_dim, d_model, d_head=4, nlayers=10, dropout=0.).to(device)
print(sum(p.numel() for p in model.parameters() if p.requires_grad)) # 27584
x = torch.rand((batch, in_dim, 32, 32), device=device)
# x =  torch.rand((batch, seq_len, in_dim), device=device)
out = model(x)
print(out.shape)
# # # print(out)
# model = TransformerPredictor(in_dim, d_model, out_dim=None, d_head=4, d_hid=None, nlayers=1).to(device)
# out = model(out)
# print(out.shape)




# @title SeqJEPA
import torch
import torch.nn as nn
import torch.nn.functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"


class SeqJEPA(nn.Module):
    def __init__(self, in_dim=3, d_model=32, out_dim=None, nlayers=2, d_head=4):
        super().__init__()
        if out_dim is None: out_dim = d_model
        self.patch_size = 32 # 4
        self.context_encoder = TransformerModel(in_dim, d_model, out_dim=out_dim, d_head=d_head, nlayers=nlayers, dropout=0.)
        # self.predicter = TransformerPredictor(out_dim, d_model//2, out_dim, d_head=d_head, nlayers=nlayers//2, dropout=0.)
        self.predicter = TransformerPredictor(out_dim, 3*d_model//8, out_dim, d_head=4, nlayers=1, dropout=0.)
        # self.predicter = TransformerPredictor(out_dim, 3*d_model//8, 3*4**2, d_head=d_head, nlayers=nlayers//2, dropout=0.)
        import copy
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self.target_encoder.requires_grad_(False)
        self.classifier = nn.Linear(out_dim, 18) # 10 18

    def loss(self, x): # [batch, T, 3]
        # batch, seq, dim = x.shape
        b,c,h,w = x.shape
        # # print(x.shape)
        # target_mask = multiblock(seq//self.patch_size, min_s=0.15, max_s=0.2, M=4).any(0) # best.2.3M4 og.15.2M4# mask out targets to be predicted # [M, seq]
        # # target_mask = multiblock(seq//self.patch_size, min_s=0.2, max_s=0.3, M=4).any(0) # best.2.3M4 og.15.2M4# mask out targets to be predicted # [M, seq]
        # # target_mask = randpatch(seq//self.patch_size, mask_size=8, gamma=.9) # 8.9 [seq]
        # context_mask = ~multiblock(seq//self.patch_size, min_s=0.85, max_s=1., M=1)|target_mask # og .85,1.M1 # [1, seq], True->Mask

        target_mask = multiblock2d((8,8), scale=(.2,.3), aspect_ratio=(.75,1.5), M=4).any(0).unsqueeze(0) # [1,h,w], True->Mask
        # # target_mask = simplexmask(hw=(8,8), scale=(.6,.8)).unsqueeze(0)
        context_mask = ~multiblock2d((8,8), scale=(.85,1), aspect_ratio=(1,1), M=1)|target_mask # [1,h,w], True->Mask
        # context_mask = torch.zeros((1,8,8), dtype=bool)|target_mask # [1,h,w], True->Mask
        # # context_mask = ~simplexmask(hw=(8,8), scale=(.85,1)).unsqueeze(0)|target_mask # [1,h,w]
        # # print(target_mask.shape, context_mask.shape)

        # imshow(target_mask)
        # imshow(context_mask)

        # target_mask, context_mask = target_mask.flatten(), context_mask.flatten(1) # [1,8*8]
        target_mask, context_mask = target_mask.flatten(), context_mask.flatten() # [8*8]
        target_mask, context_mask = target_mask.to(device), context_mask.to(device)
        # print(target_mask.shape)
        # print(target_mask)
        # print(context_mask)

        context_indices = (~context_mask).nonzero().squeeze(-1) # int idx [num_context_toks] , idx of context not masked
        # print('seqjepa loss context_indices',context_indices)
        # print('seqjepa loss x',x.shape)
        sx = self.context_encoder(x, context_indices=context_indices) # [batch, num_context_toks, out_dim]
        # print('seqjepa loss sx',sx.shape)

        trg_indices = target_mask.nonzero().squeeze(-1) # int idx [num_trg_toks] , idx of targets that are masked
        # print(trg_indices.shape)
        sy_ = self.predicter(sx, context_indices=context_indices, trg_indices=trg_indices) # [batch*M, num_trg_toks, out_dim]
        sy_ = F.layer_norm(sy_, (sy_.size(-1),))
        with torch.no_grad():
            sy = self.target_encoder(x.detach()) # [batch, num_trg_toks, out_dim]
            sy = sy[:,trg_indices] # nan bec len(trg_ind)==0 # print('loss sy',torch.isnan(sy).any())
            sy = F.layer_norm(sy, (sy.size(-1),))
        loss = F.mse_loss(sy, sy_)
        return loss

    def forward(self, x): # [batch, T, 3]
        sx = self.context_encoder(x)
        out = sx.mean(dim=1)
        return out

    def classify(self, x): # [batch, T, 3]
        sx = self.forward(x)
        out = self.classifier(sx)
        return out

# min_s=0.15, max_s, M
# trg.15.2M4 C.85 1

seq_jepa = SeqJEPA(in_dim=3, d_model=32, out_dim=16, nlayers=4, d_head=4).to(device)#.to(torch.float)
# seq_jepa = SeqJEPA(in_dim=3, d_model=1024, out_dim=16, nlayers=12, d_head=16).to(device)#.to(torch.float)
# optim = torch.optim.AdamW(seq_jepa.parameters(), lr=1e-3) # 1e-3?
optim = torch.optim.AdamW([{'params': seq_jepa.context_encoder.parameters()},
    # {'params': seq_jepa.predicter.parameters(), 'lr': 1e-2}], lr=1e-3)#, weight_decay=0) default 1e-2
    {'params': seq_jepa.predicter.parameters(), 'lr': 3e-3}], lr=1e-3, weight_decay=1e-2) # default 1e-2, 5e-2

# https://github.com/facebookresearch/ijepa/blob/main/configs/in1k_vith14_ep300.yaml
# d_model 1024,384
# depth 12,6/12
# wd 5e-2 - 4e-1
# adamw 1e-4 - 1e-3 - 1e-6
# ema 0.996-1

print(sum(p.numel() for p in seq_jepa.parameters() if p.requires_grad)) # 27584
# print(sum(p.numel() for p in seq_jepa.parameters())) # 27584
# print(sum(p.numel() for p in seq_jepa.predicter.transformer_encoder.parameters() if p.requires_grad)) # 27584
# print(sum(p.numel() for p in seq_jepa.context_encoder.transformer_encoder.parameters() if p.requires_grad)) # 27584
# print(sum(p.numel() for p in seq_jepa.target_encoder.transformer_encoder.parameters() if p.requires_grad)) # 27584
# d_model^2 * nlayers

# x = torch.rand((2,1024,3), device=device)
x = torch.rand((2,3,32,32), device=device)
out = seq_jepa.loss(x)
print(out.shape)

class Classifier(nn.Module):
    def __init__(self, in_dim, num_classes=10):
        super().__init__()
        self.classifier = nn.Linear(in_dim, num_classes)
    def forward(self, x): return self.classifier(x)
classifier = Classifier(16).to(device)
# classifier = Classifier(16, 18).to(device)
coptim = torch.optim.SGD(classifier.parameters(), lr=1e-3)
# optim = torch.optim.AdamW([{'params': seq_jepa.parameters()}, {'params': classifier.parameters(), 'lr': 1e-3}], lr=1e-3)




# @title strain ctrain test
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
scaler = torch.GradScaler()

def strain(model, dataloader, optim, scheduler=None): # train function with automatic mixed precision
    model.train()
    for i, (x, _) in enumerate(dataloader):
        x = x.to(device)#.to(torch.bfloat16) # [b,c,h,w] -> [b,h*w,c]
        with torch.autocast(device_type=device, dtype=torch.bfloat16): # bfloat16 float16
            loss = model.loss(x)
        optim.zero_grad()
        scaler.scale(loss).backward()
        # scaler.unscale_(optim)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 5) # 0.5

        scaler.step(optim)
        scaler.update()

        with torch.no_grad():
            m=0.999 # 0.99 m = next(momentum_scheduler)
            norms=[]
            for param_q, param_k in zip(model.student.parameters(), model.teacher.parameters()):
                param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

        # if scheduler is not None: scheduler.step()
        print("strain",loss.item())
        try: wandb.log({"loss": loss.item()})
        except NameError: pass
        if i>=50: break


def ctrain(model, classifier, dataloader, coptim, scheduler=None): # train function with automatic mixed precision
    model.eval()
    classifier.train()
    for i, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device) # [batch, ]
        # x = x.flatten(2).transpose(-2,-1).to(torch.bfloat16)
        with torch.autocast(device_type=device, dtype=torch.bfloat16): # bfloat16 float16
            with torch.no_grad():
                sx = model(x).detach()
            # print(sx[0][0])
            y_ = classifier(sx)
            loss = F.cross_entropy(y_, y)

        coptim.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(coptim)
        scaler.update()
        print("classify",loss.item())
        try: wandb.log({"closs": loss.item()})
        except NameError: pass
        if i>=10: break


def test(model, classifier, dataloader):
    model.eval()
    classifier.eval()
    for i, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device) # [batch, ]
        # x = x.flatten(2).transpose(-2,-1)#.to(torch.bfloat16)
        with torch.no_grad():
            sx = model(x)
            y_ = classifier(sx)
        correct = (y==y_.argmax(dim=1)).sum().item()
        print(correct/len(y))
        try: wandb.log({"correct": correct/len(y)})
        except NameError: pass
        if i>=10: break


for i in range(1000):
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    strain(seq_jepa, train_loader, optim)
    ctrain(seq_jepa, classifier, train_loader, coptim)
    test(seq_jepa, classifier, test_loader)



