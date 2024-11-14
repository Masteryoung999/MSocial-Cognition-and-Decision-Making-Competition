import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (key_dim // num_heads) ** -0.5

        self.to_q = nn.Linear(query_dim, key_dim)
        self.to_k = nn.Linear(key_dim, key_dim)
        self.to_v = nn.Linear(key_dim, key_dim)
        self.to_out = nn.Linear(key_dim, query_dim)

    def forward(self, query, key_value):
        """
        Args:
            query: [num_nodes, query_dim]
            key_value: [num_nodes, key_dim]
        Returns:
            out: [num_nodes, query_dim]
        """
        # Add batch dimension for attention computation
        query = query.unsqueeze(0)  # [1, num_nodes, query_dim]
        key_value = key_value.unsqueeze(0)  # [1, num_nodes, key_dim]

        q = self.to_q(query)  # [1, num_nodes, key_dim]
        k = self.to_k(key_value)  # [1, num_nodes, key_dim]
        v = self.to_v(key_value)  # [1, num_nodes, key_dim]

        # Reshape for multi-head attention
        b, n, d = q.shape
        head_dim = d // self.num_heads

        q = q.view(b, n, self.num_heads, head_dim).transpose(1, 2)  # [1, heads, num_nodes, head_dim]
        k = k.view(b, n, self.num_heads, head_dim).transpose(1, 2)  # [1, heads, num_nodes, head_dim]
        v = v.view(b, n, self.num_heads, head_dim).transpose(1, 2)  # [1, heads, num_nodes, head_dim]

        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [1, heads, num_nodes, num_nodes]
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # [1, heads, num_nodes, head_dim]
        out = out.transpose(1, 2).contiguous().view(b, n, d)  # [1, num_nodes, key_dim]
        out = self.to_out(out)  # [1, num_nodes, query_dim]

        # Remove batch dimension
        out = out.squeeze(0)  # [num_nodes, query_dim]
        return out


class FC(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.fc(x)


class BIMModule(nn.Module):
    def __init__(self, f_dim=128, v_dim=64):
        super().__init__()
        self.f_dim = f_dim  # 128
        self.v_dim = v_dim  # 64

        # Cross attention layers
        self.cross_attn_f2v = CrossAttention(query_dim=f_dim, key_dim=v_dim)
        self.cross_attn_v2f = CrossAttention(query_dim=v_dim, key_dim=f_dim)

        # Final FC layer
        self.fc_f = FC(f_dim, f_dim)

    def forward(self, f, v):
        """
        Args:
            f: Node feature tensor [num_nodes, f_dim]  (128)
            v: Visual feature tensor [num_nodes, v_dim]  (64)
        Returns:
            f_new: Updated node features [num_nodes, f_dim]  (128)
            v_new: Updated visual features [num_nodes, v_dim]  (64)
        """
        # Bidirectional cross attention
        f2v = self.cross_attn_f2v(f, v)  # [num_nodes, f_dim]
        v2f = self.cross_attn_v2f(v, f)  # [num_nodes, v_dim]

        # Update features
        f_new = self.fc_f(f + f2v)  # [num_nodes, f_dim]
        v_new = v + v2f  # [num_nodes, v_dim]

        return f_new, v_new
