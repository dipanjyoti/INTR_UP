# --------------------------------------------------------
# References:
# https://github.com/jxhe/unify-parameter-efficient-tuning
# --------------------------------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Adapter_CrossNonParam(nn.Module):  # Original name DVPTAdapter, so far works well for medical imaging
    def __init__(self, 
                config=None, 
                d_model=None,
                bottleneck=None, 
                num_prompt_tokens=200):
        super().__init__()
        self.n_embd = config.d_model if d_model is None else d_model
        self.num_prompt_tokens = num_prompt_tokens
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear = nn.GELU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)
        self.scale = self.down_size ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.gate = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        B, N, C = x.shape

        down = self.non_linear(self.down_proj(x))
        prompt_down = down[:, :self.num_prompt_tokens, :]
        token_down = down[:, self.num_prompt_tokens:, :]

        prompt_attn = torch.matmul(prompt_down, token_down.transpose(-2, -1)) * self.scale
        prompt_attn = self.softmax(prompt_attn)
        prompt_out = torch.matmul(prompt_attn, token_down)

        combined = torch.cat([prompt_out, token_down], dim=1)
        up = self.up_proj(combined)
        return self.gate * up


class Adapter_SelfParam_CrossNonParam(nn.Module): # Original name DVPTAdapterSA, so far works well for biodiversity imaging
    def __init__(self, 
                config=None, 
                d_model=None,
                bottleneck=None, 
                num_prompt_tokens=200):
        super().__init__()
        self.n_embd = config.d_model if d_model is None else d_model
        self.num_prompt_tokens = num_prompt_tokens
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear = nn.GELU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)
        self.scale = self.down_size ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.gate = nn.Parameter(torch.tensor(1.0))
        self.prompt_self_attn = nn.MultiheadAttention(embed_dim=self.down_size, num_heads=4, batch_first=True)

    def forward(self, x):
        B, N, C = x.shape

        # Down-project
        down = self.non_linear(self.down_proj(x))
        prompt_down = down[:, :self.num_prompt_tokens, :]  # (B, P, d')
        token_down = down[:, self.num_prompt_tokens:, :]   # (B, T, d')

        # Step 1: Prompt Self-Attention (P ←→ P)
        prompt_self_attended, _ = self.prompt_self_attn(prompt_down, prompt_down, prompt_down)

        # Step 2: Cross-Attention (P ← T)
        prompt_attn = torch.matmul(prompt_self_attended, token_down.transpose(-2, -1)) * self.scale
        prompt_attn = self.softmax(prompt_attn)
        prompt_out = torch.matmul(prompt_attn, token_down)  # (B, P, d')

        combined = torch.cat([prompt_out, token_down], dim=1)
        up = self.up_proj(combined)
        return self.gate * up

class Adapter_SelfNonParam_CrossNonParam(nn.Module): # Original name was DVPTAdapterSA_NP; checking if it performs better! 
    def __init__(self, 
                config=None, 
                d_model=None,
                bottleneck=None, 
                num_prompt_tokens=200):
        super().__init__()
        self.n_embd = config.d_model if d_model is None else d_model
        self.num_prompt_tokens = num_prompt_tokens
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear = nn.GELU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)
        self.scale = self.down_size ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.gate = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        B, N, C = x.shape

        # Down-project
        down = self.non_linear(self.down_proj(x))
        prompt_down = down[:, :self.num_prompt_tokens, :]  # (B, P, d')
        token_down = down[:, self.num_prompt_tokens:, :]   # (B, T, d') #take the CLS token out

        # Step 1: Prompt Self-Attention (parameter-free)
        prompt_scores = torch.matmul(prompt_down, prompt_down.transpose(-2, -1)) * self.scale  
        prompt_weights = self.softmax(prompt_scores)
        prompt_self_attended = torch.matmul(prompt_weights, prompt_down) 

        # Step 2: Cross-Attention (P ← T)
        prompt_attn = torch.matmul(prompt_self_attended, token_down.transpose(-2, -1)) * self.scale
        prompt_attn = self.softmax(prompt_attn)
        prompt_out = torch.matmul(prompt_attn, token_down)

        combined = torch.cat([prompt_out, token_down], dim=1)
        up = self.up_proj(combined)
        return self.gate * up  

class Adapter_SelfParam_CrossParam(nn.Module): # No original name! Just to check if it performs better!
    def __init__(self, 
                config=None, 
                d_model=None,
                bottleneck=None, 
                num_prompt_tokens=200):
        super().__init__()
        self.n_embd = config.d_model if d_model is None else d_model
        self.num_prompt_tokens = num_prompt_tokens
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear = nn.GELU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.gate = nn.Parameter(torch.tensor(1.0))
        self.prompt_self_attn = nn.MultiheadAttention(embed_dim=self.down_size, num_heads=4, batch_first=True)
        self.prompt_cross_attn = nn.MultiheadAttention(embed_dim=self.down_size, num_heads=4, batch_first=True)

    def forward(self, x):
        B, N, C = x.shape

        # Down-project
        down = self.non_linear(self.down_proj(x))
        prompt_down = down[:, :self.num_prompt_tokens, :]  # (B, P, d')
        token_down = down[:, self.num_prompt_tokens:, :]   # (B, T, d')

        # Step 1: Prompt Self-Attention (P ←→ P)
        prompt_self_attended, _ = self.prompt_self_attn(prompt_down, prompt_down, prompt_down)

        # Step 2: Cross-Attention (P ← T)
        prompt_out, _ = self.prompt_cross_attn(prompt_self_attended, token_down, token_down)

        combined = torch.cat([prompt_out, token_down], dim=1)
        up = self.up_proj(combined)
        return self.gate * up


class Adapter_CrossParam(nn.Module): # Original name was Adapter_crossParam; just to how it behaves for biodiversity dataset. (Not so important)
    def __init__(self, 
                config=None, 
                d_model=None,
                bottleneck=None, 
                num_prompt_tokens=200):
        super().__init__()
        self.n_embd = config.d_model if d_model is None else d_model
        self.num_prompt_tokens = num_prompt_tokens
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear = nn.GELU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.gate = nn.Parameter(torch.tensor(1.0))
        self.prompt_cross_attn = nn.MultiheadAttention(embed_dim=self.down_size, num_heads=4, batch_first=True)

    def forward(self, x):
        B, N, C = x.shape
        
        down = self.non_linear(self.down_proj(x))
        prompt_down = down[:, :self.num_prompt_tokens, :]
        token_down = down[:, self.num_prompt_tokens:, :]

        prompt_out, _ = self.prompt_cross_attn(prompt_down, token_down, token_down)

        combined = torch.cat([prompt_out, token_down], dim=1)
        up = self.up_proj(combined)
        return self.gate * up

























class AdaptFormerAdapter(nn.Module):
    def __init__(self,
                 config=None,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="bert",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output