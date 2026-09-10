import torch.nn.functional as F
import torch.nn as nn
import math

class Attention(nn.Module):
    def __init__(self,d_model,n_head,resid_std):
        super(Attention,self).__init__()

        assert (d_model%n_head==0)
        self.d_model=d_model
        self.n_head=n_head
        self.d_k=d_model//n_head
        self.c_attn=nn.Linear(d_model,3*d_model)
        self.c_proj=nn.Linear(d_model,d_model)

        nn.init.trunc_normal_(self.c_attn.weight,std=0.02)
        nn.init.zeros_(self.c_attn.bias)
        nn.init.trunc_normal_(self.c_proj.weight,std=resid_std)
        nn.init.zeros_(self.c_proj.bias)

    def forward(self,x,return_attn=False):
        B,T,d_model=x.size()
        qkv=self.c_attn(x)
        q,k,v=qkv.split(self.d_model,dim=2)
        q=q.view(B,T,self.n_head,self.d_k).transpose(1,2)
        k=k.view(B,T,self.n_head,self.d_k).transpose(1,2)
        v=v.view(B,T,self.n_head,self.d_k).transpose(1,2)

        if return_attn:
            attn=F.softmax((q@k.transpose(-1,-2))/math.sqrt(self.d_k),dim=-1)
            y=attn@v
        else:
            attn=None
            y=F.scaled_dot_product_attention(q,k,v)
        out=y.transpose(1,2).contiguous().view(B,T,d_model)
        out=self.c_proj(out)
        return (out,attn) if return_attn else out

class MLP(nn.Module):
    def __init__(self,d_model,resid_std):
        super(MLP,self).__init__()

        self.fc1=nn.Linear(d_model,4*d_model)
        self.gelu=nn.GELU(approximate='tanh')
        self.fc2=nn.Linear(4*d_model,d_model)

        nn.init.trunc_normal_(self.fc1.weight,std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.trunc_normal_(self.fc2.weight,std=resid_std)
        nn.init.zeros_(self.fc2.bias)

    def forward(self,x):
        return self.fc2(self.gelu(self.fc1(x)))

class SWiGLU(nn.Module):
    def __init__(self,d_model,resid_std):
        super(SWiGLU,self).__init__()

        hidden=8*d_model//3
        self.fc1=nn.Linear(d_model,hidden)
        self.fc3=nn.Linear(d_model,hidden)
        self.silu=nn.SiLU()
        self.fc2=nn.Linear(hidden,d_model)

        nn.init.trunc_normal_(self.fc1.weight,std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.trunc_normal_(self.fc3.weight,std=0.02)
        nn.init.zeros_(self.fc3.bias)
        nn.init.trunc_normal_(self.fc2.weight,std=resid_std)
        nn.init.zeros_(self.fc2.bias)

    def forward(self,x):
        return self.fc2(self.silu(self.fc1(x))*self.fc3(x))


def _make_norm(name,d_model,affine=True):
    if name=='layer_norm':
        return nn.LayerNorm(d_model,elementwise_affine=affine)
    if name=="rms_norm":
        return nn.RMSNorm(d_model,elementwise_affine=affine)
    else:
        raise ValueError(name)

def _make_ffn(name,d_model,resid_std):
    if name=="gelu_mlp":
        return MLP(d_model,resid_std)
    if name=="swiglu":
        return SWiGLU(d_model,resid_std)
    else:
        raise ValueError(name)


class ViTBlock(nn.Module):
    def __init__(self,d_model,n_heads,resid_std,norm_type="layer_norm",ffn_type="gelu_mlp"):
        super(ViTBlock,self).__init__()
        self.ln1=_make_norm(norm_type,d_model)
        self.attn=Attention(d_model,n_heads,resid_std)
        self.ln2=_make_norm(norm_type,d_model)
        self.ffn=_make_ffn(ffn_type,d_model,resid_std)

    def forward(self,x,return_attn=False):
        if return_attn:
            a,attn=self.attn(self.ln1(x),return_attn=True)
            x=x+a
            x=x+self.ffn(self.ln2(x))
            return x,attn
        x=x+self.attn(self.ln1(x))
        x=x+self.ffn(self.ln2(x))
        return x



