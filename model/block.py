import torch.nn as nn
from timm.layers import DropPath
from timm.models.vision_transformer import LayerScale
from timm.layers.trace_utils import _assert
import torch
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List
from model.mlp import MlpPETL
from model.attention import AttentionPETL

from model.adapter import Adapter_CrossNonParam, Adapter_SelfParam_CrossNonParam, Adapter_SelfNonParam_CrossNonParam, Adapter_SelfParam_CrossParam, Adapter_CrossParam

class BlockPETL(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = MlpPETL,
            params=None
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AttentionPETL(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            ############# Added module #############
            params=params
            ############# Added module end #############
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.params = params


        if self.params.adapter== 'Adapter_CrossNonParam':
            self.vpt_adapter = Adapter_CrossNonParam(
                    config=self.params, 
                    d_model=dim, 
                    bottleneck=64, 
                    num_prompt_tokens=self.params.vpt_num)
        elif self.params.adapter=='Adapter_SelfParam_CrossNonParam':
            self.vpt_adapter = Adapter_SelfParam_CrossNonParam(
                    config=self.params, 
                    d_model=dim, 
                    bottleneck=64,
                    num_prompt_tokens=self.params.vpt_num)
        elif self.params.adapter== 'Adapter_SelfNonParam_CrossNonParam':
            self.vpt_adapter = Adapter_SelfNonParam_CrossNonParam(
                    config=self.params, 
                    d_model=dim, 
                    bottleneck=64,
                    num_prompt_tokens=self.params.vpt_num)
        elif self.params.adapter== 'Adapter_SelfParam_CrossParam':
            self.vpt_adapter = Adapter_SelfParam_CrossParam(
                    config=self.params, 
                    d_model=dim, 
                    bottleneck=64,
                    num_prompt_tokens=self.params.vpt_num)
        elif self.params.adapter== 'Adapter_CrossParam':
            self.vpt_adapter = Adapter_CrossParam(
                    config=self.params, 
                    d_model=dim, 
                    bottleneck=64, 
                    num_prompt_tokens=self.params.vpt_num)
        else:
            print ("Provide an appropriate adapter")
            exit()


    def forward(self, x: torch.Tensor, idx, blur_head_lst=[], target_cls=-1) -> Tuple[torch.Tensor, torch.Tensor]: #updated for AdaPrompt_CAM
        output, attn_map = self.attn(self.norm1(x), idx, blur_head_lst=blur_head_lst, target_cls=target_cls)
        x = x + self.drop_path1(self.ls1(output))

        residual = self.norm2(x)
        mlp_out = self.mlp(residual)
        mlp_out = mlp_out + self.vpt_adapter(residual)

        x = x + self.drop_path2(self.ls2(mlp_out))
        return x, attn_map

