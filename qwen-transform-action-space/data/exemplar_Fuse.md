| edge | task | lvl | role | ×speedup | hunks | lines | regions | realization evidence (quoted from annotation) |
|---|---|---|---|---|---|---|---|---|
| `L2_2_11_167_168` | 2_11 | L2 | primary | 1.06 | 3 | 141 | 3 | `def _pool_tanh_kernel(` |
| `L2_2_12_1_37` | 2_12 | L2 | secondary | 4.67 | 1 | 134 | 1 | `x = self.gemm(x)` |
| `L2_2_25_163_170` | 2_25 | L2 | primary | 5.32 | 2 | 111 | 2 | `reduced = triton_min_over_channels(conv_out)  # shape: (N, 1, H, W)` |
| `L2_2_27_148_156` | 2_27 | L2 | primary | 1.25 | 1 | 268 | 1 | `x_act = F.hardswish(x)  # (B, C, D, H, W)` |
| `L2_2_28_106_107` | 2_28 | L2 | primary | 2.60 | 4 | 55 | 3 | `x = x + y` |
| `L2_2_40_52_80` | 2_40 | L2 | secondary | 2.23 | 1 | 266 | 1 | `out = triton_linear(x, self.matmul.weight, self.matmul.bias)` |
| `L2_2_42_1_69` | 2_42 | L2 | primary | 26.83 | 2 | 120 | 2 | `x = self.conv_transpose(x)` |
| `L2_2_42_1_169` | 2_42 | L2 | primary | 26.99 | 2 | 135 | 2 | `x = self.conv_transpose(x)` |
| `L2_2_46_8_94` | 2_46 | L2 | secondary | 1.44 | 4 | 258 | 3 | `x = self.conv(x)` |
| `L2_2_48_29_30` | 2_48 | L2 | primary | 1.57 | 1 | 107 | 1 | `y = x * scale_b` |
| `L2_2_54_77_87` | 2_54 | L2 | primary | 1.18 | 2 | 146 | 2 | `x = self.leaky_relu(x)` |
| `L2_2_60_112_114` | 2_60 | L2 | primary | 1.13 | 7 | 81 | 4 | `def _swish_inplace_kernel(` |
| `L2_2_63_1_142` | 2_63 | L2 | secondary | 5.23 | 2 | 114 | 1 | `x = self.linear(x)` |
| `L2_2_63_1_119` | 2_63 | L2 | secondary | 4.94 | 1 | 114 | 1 | `x = self.linear(x)` |
| `L2_2_65_86_111` | 2_65 | L2 | primary | 7.65 | 1 | 122 | 1 | `x_conv = self.conv(x)` |
| `L2_2_74_98_180` | 2_74 | L2 | primary | 1.29 | 1 | 206 | 1 | `x = triton_fused_leakyrelu_mul(x, mul_expanded, neg_slope=self.neg_slope)` |
| `L2_2_77_32_79` | 2_77 | L2 | secondary | 6.16 | 1 | 220 | 1 | `x = self.conv_transpose(x)` |
| `L2_2_77_33_73` | 2_77 | L2 | secondary | 6.18 | 1 | 255 | 1 | `x = self.conv_transpose(x)` |
| `L2_2_84_123_145` | 2_84 | L2 | primary | 21.49 | 2 | 71 | 2 | `x = self.gemm(x)` |
| `L2_2_87_134_137` | 2_87 | L2 | secondary | 7.65 | 1 | 113 | 1 | `self.total_subtract = float(subtract_value_1) + float(subtract_value_2)` |
| `L2_2_90_11_92` | 2_90 | L2 | primary | 1.12 | 1 | 141 | 1 | `x = triton_fused_leaky_add_clamp(x, self.sum_tensor)` |
| `L2_2_93_154_177` | 2_93 | L2 | primary | 1.18 | 2 | 69 | 2 | `x = triton_add_min(x, self.add_value)` |
| `L3_3_16_13_15` | 3_16 | L3 | primary | 1.56 | 7 | 149 | 3 | `out = fused_bn_relu(x, self.bn)` |
| `L3_3_2_33_61` | 3_2 | L3 | primary | 7.69 | 4 | 189 | 3 | `a = tl.load(a_ptrs, mask=a_mask, other=0.0)` |
| `L3_3_43_2_19` | 3_43 | L3 | primary | 1.84 | 1 | 60 | 1 | `att_half = (q_half @ k_half.transpose(-2, -1))  # (B, nh, T, T) in fp16` |

**`Fuse`: 25 edges, 22 distinct tasks, 23 distinct region sets.** Distinct-count alone is not evidence (Section 3.5); the point of this table is that the *shape* of each realization differs — different numbers of hunks, different code regions, different changed lines — while the named decision recurs.
