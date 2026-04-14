# Tiny ViT & TransformerEncoder 詳解

## 整體 pipeline

```
Input (batch, seq_len=15, 11)
    ↓ TinyViT
    ↓ LSTM / BiLSTM
    ↓ MultiheadAttention
    ↓ last token → FF → pred (regression)
                       → spike_head (classification)
```

---

## Part 1：Tiny ViT

### 逐步拆解

#### Step 1 — Input Projection（`input_proj`）
```python
self.input_proj = nn.Linear(inFeatures, d_model)
# (batch, 15, 11) → (batch, 15, 128)
```
取代 vit_b_16 的 `conv_proj`（它把影像 patch 投射到 768 維）。這裡直接線性投射，沒有卷積、沒有假裝是影像。

#### Step 2 — CLS Token（`cls_token`）
```python
self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
cls_tokens = self.cls_token.expand(batch_size, -1, -1)
x = torch.cat((cls_tokens, x), dim=1)
# (batch, 15, 128) → (batch, 16, 128)
```
一個可學習的「聚合 token」貼在序列最前面。TransformerEncoder 跑完後，CLS token 理論上會聚合整段時序的全局資訊（ViT 原論文的設計）。

> **注意**：後續取最後一個 token（`attnOutput[:, -1, :]`）而非 CLS token（`attnOutput[:, 0, :]`）。CLS token 在這裡的主要貢獻是提供一個「全局上下文 slot」讓 attention 能參照。

#### Step 3 — Positional Embedding（`pos_embedding`）
```python
self.pos_embedding = nn.Parameter(torch.randn(1, sequenceLength + 1, d_model))
x = x + self.pos_embedding
# (batch, 16, 128) + (1, 16, 128) → (batch, 16, 128)
```
可學習的位置編碼，告訴 Transformer 每個 token 在序列中的位置（timestep 順序）。與 NLP/ViT 標準做法相同。

#### Step 4 — TransformerEncoder（`encoder`）
```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=128, nhead=4, dim_feedforward=512,
    dropout=0.1, batch_first=True, activation='gelu'
)
self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
# (batch, 16, 128) → (batch, 16, 128)
```

每一層 TransformerEncoderLayer 內部結構：

```
x → MultiHeadSelfAttention(4 heads) → Dropout → LayerNorm → Add
  → FeedForward(128 → 512 → 128, GELU) → Dropout → LayerNorm → Add
```

- **4 heads**：每個 head 處理 128/4=32 維的子空間，各自學習不同的時序相關性
- **dim_feedforward=512**：FFN 寬度是 d_model 的 4 倍（標準比例）
- **GELU**：比 ReLU 在 Transformer 中更常見，輸出平滑、梯度穩定
- **3 層疊加**：讓模型能學到多階層的時序模式

#### Step 5 — LayerNorm（`ln`）
```python
self.ln = nn.LayerNorm(d_model)
```
Transformer 輸出最後再做一次 LayerNorm（標準 post-LN 做法）。

#### 初始化策略（`_init_weights`）
```python
nn.init.xavier_uniform_(self.input_proj.weight)     # linear layer
nn.init.trunc_normal_(self.pos_embedding, std=0.02) # ViT 官方做法
nn.init.trunc_normal_(self.cls_token, std=0.02)
```
從零訓練時初始化很重要，xavier 保持梯度尺度穩定，`std=0.02` 的 trunc_normal 是 ViT 原論文的建議值。

---

### TinyViT 輸出接 LSTM/BiLSTM

```python
# TinyViT 輸出 (batch, 16, 128)，全部送進 LSTM
out, _ = self.lstm1(vit_output)   # (batch, 16, 128) → (batch, 16, 128)
out, _ = self.lstm2(dropout(out)) # (batch, 16, 128) → (batch, 16, 64)
```

ViT 用 Attention 捕捉全局相關性，LSTM 再用順序建模強化時序因果結構。最後取 `feat[:, -1, :]`（最後一個 timestep 的隱藏狀態）作為 context vector。

---

### 參數量比較

| 元件 | vit_b_16 (model6) | TinyViT (model7) |
|------|------------------|-----------------|
| Input projection | Conv2d patch embed | Linear(11→128) |
| Transformer | 12 blocks, d=768, heads=12 | 3 blocks, d=128, heads=4 |
| 總參數 | **86M** | **~805K (embb) / ~1.1M (mmtc)** |
| 比例 | 100x | **1x** |

---

### 為何 Tiny ViT 比 vit_b_16 好

| 問題 | vit_b_16 | TinyViT |
|------|----------|---------|
| 輸入假設 | 224×224 影像（196 patches） | 15×11 時序 |
| ImageNet 預訓練 | 無意義（強行 pretrained=False 後） | 不依賴預訓練 |
| 模型容量 vs 資料量 | 嚴重不匹配（86M 參數，訓練資料不夠） | 匹配（<1M 參數） |
| Regression 輸出 | 方塊波（只能猜均值） | 能追蹤波形 |
| mMTC spike | 全 positive（FP Rate=0.88） | 有選擇性（FP Rate=0.20） |

---

## Part 2：TransformerEncoder 詳述

### 結構總覽

```
TransformerEncoder (num_layers=3)
├── Layer 0: TransformerEncoderLayer
├── Layer 1: TransformerEncoderLayer
└── Layer 2: TransformerEncoderLayer

每層內部：
x → [Self-Attention] → [Add & Norm] → [FFN] → [Add & Norm] → x'
```

---

### Self-Attention 機制

#### Q, K, V 的來源

Encoder 是 **self-attention**：Q、K、V 全部來自同一個輸入 `x`。

```
x: (batch, 16, 128)   ← 16 tokens（CLS + 15 timesteps），每個 128 維

Q = x · Wq   (batch, 16, 128)
K = x · Wk   (batch, 16, 128)
V = x · Wv   (batch, 16, 128)
```

`Wq, Wk, Wv` 都是可學習的線性投射，每個 shape 為 `(128, 128)`。

#### Multi-Head 拆分（4 heads）

4 個 head 各自分到 128/4 = 32 維的子空間：

```
Q → split → [Q1(batch,16,32), Q2(batch,16,32), Q3(batch,16,32), Q4(batch,16,32)]
K → split → [K1, K2, K3, K4]
V → split → [V1, V2, V3, V4]
```

每個 head 獨立做 Scaled Dot-Product Attention：

```
Attention(Qi, Ki, Vi) = softmax(Qi · Ki^T / √32) · Vi
                                         ↑
                             scale factor 防止 dot product 過大導致 softmax 梯度消失
```

輸出 shape：`(batch, 16, 32)` × 4 heads

#### Concat + 線性投射

```
concat([head1, head2, head3, head4])  → (batch, 16, 128)
→ Linear(128, 128)                    → (batch, 16, 128)
```

---

### Attention 矩陣的物理意義

以 seq_len=15 為例，attention 矩陣 shape `(16, 16)`：

```
          CLS  t1  t2  t3 ... t15
    CLS  [                       ]
    t1   [                       ]
    t2   [                       ]
    ...  [                       ]
    t15  [                       ]
```

每一行代表「這個 token 對所有其他 token 的注意力權重」（softmax 後總和=1）。

- **t5 那一行**：t5 這個 timestep 在「做預測」時，最關注哪些其他 timestep？
- **CLS 那一行**：CLS token 聚合了整段序列的哪些資訊？
- **不同 head**：各自學到不同的依賴關係（如 head1 學近鄰相關、head2 學遠距離相關）

---

### Add & Norm（Residual + LayerNorm）

```python
# 第一個 sublayer（Self-Attention）
x = x + Dropout(Attention(x))
x = LayerNorm(x)

# 第二個 sublayer（FFN）
x = x + Dropout(FFN(x))
x = LayerNorm(x)
```

**Residual connection（`x + ...`）的作用：**
- 梯度能直接流回早期層，避免梯度消失
- 強迫 Attention 學「修正量」而非「從零重建」，訓練更穩定

**LayerNorm（不是 BatchNorm）的作用：**
- 對每個 token 的 128 維向量做正規化（而非跨 batch）
- 不依賴 batch size，適合序列模型

---

### FFN（FeedForward Network）

```
FFN(x) = Linear(512, 128) · GELU( Linear(128, 512) · x )
```

```
(batch, 16, 128)
→ Linear(128 → 512)   # 先擴張 4 倍
→ GELU                # 非線性激活
→ Dropout(0.1)
→ Linear(512 → 128)   # 壓回原維度
→ (batch, 16, 128)
```

**GELU vs ReLU：**

```
ReLU(x) = max(0, x)    ← 在 x<0 硬截斷，梯度=0
GELU(x) ≈ x · Φ(x)    ← 平滑版，x 很負時輸出趨近 0 但非硬截斷
```

GELU 在 Transformer 系列（BERT, GPT, ViT）幾乎是標配，訓練更穩定。

**FFN 的作用：**
- Attention 做的是 token 之間的「跨位置資訊混合」
- FFN 做的是每個 token「自身特徵的非線性變換」
- 兩者互補：一個負責「誰跟誰相關」，一個負責「如何處理這個特徵」

---

### 三層疊加的效果

```
Layer 0: 學習「鄰近 timestep 的直接相關性」（局部模式）
Layer 1: 在 Layer 0 輸出基礎上，學習「更複雜的中程依賴」
Layer 2: 整合前兩層，學習「跨越整段序列的高階時序模式」
```

類比 CNN：
- Layer 0 ≈ 偵測邊緣（short-term 變化）
- Layer 1 ≈ 偵測紋理（中期趨勢）
- Layer 2 ≈ 偵測物體（整段流量模式）

---

### 與 vit_b_16 的差異

| 面向 | vit_b_16 | TinyViT |
|------|----------|---------|
| d_model | 768 | **128** |
| heads | 12 | **4** |
| head dim | 64 | **32** |
| FFN width | 3072 (4×768) | **512 (4×128)** |
| layers | 12 | **3** |
| Attention 矩陣 | (197×197) | **(16×16)** |
| 參數（Encoder 部分） | ~85M | **~500K** |

vit_b_16 設計給 196 個 image patch，這裡只有 16 個 token，12 層 attention 是嚴重 overparameterize 的。

---

### Encoder 在 pipeline 中的定位

```
輸入：(batch, 16, 128) — 16 個 token（CLS + 15 timesteps），各 128 維

Self-Attention 問的問題：
  「timestep t5 的流量特徵，跟 t1~t15 的哪些時間點最相關？」

FFN 問的問題：
  「這個 token 的 128 維特徵，應該怎麼非線性變換才最有用？」

輸出：(batch, 16, 128) — 每個 token 都被「全局上下文加工過」

→ 傳給 LSTM：再用順序建模強化因果時序結構
→ LSTM 輸出取最後一個 token → Attention → FF → pred
```

Transformer Encoder 負責「全局相關性」，LSTM 負責「時序因果順序」，兩者設計上互補。