Transformer LM Resource Accounting (AdamW)
(a) Peak Memory Usage Analysis
Assume float32 (4 bytes per element).

1. Static Memory (Parameters, Gradients, Optimizer State)
Parameters ($P$): $P = V \cdot d + N \times (16d^2 + 2d) + d + V \cdot d$ (Where $V$ is vocab size, $d$ is d_model, $N$ is num_layers, and FFN with SwiGLU has $3 \cdot d \cdot d_{ff} = 3 \cdot d \cdot 4d = 12d^2$ params, plus $4d^2$ for MHA). Memory ($M_{params}$) = $4P$ bytes.
Gradients: Same size as parameters. Memory ($M_{grads}$) = $4P$ bytes.
Optimizer State (AdamW): Stores first moment ($m$) and second moment ($v$) for each parameter. Memory ($M_{opt}$) = $2 \times 4P = 8P$ bytes.
Total Static Memory = $16P$ bytes.

2. Activation Memory ($M_{acts}$)
Based on storing outputs for the backward pass (per layer):

RMSNorm(s): $2 \cdot B \cdot L \cdot d$
MHA sublayer:
QKV projections: $3 \cdot B \cdot L \cdot d$
$Q^\top K$ and Softmax: $2 \cdot B \cdot H \cdot L^2$
Weighted sum: $B \cdot L \cdot d$
Output projection: $B \cdot L \cdot d$
FFN: $W_1$ output ($4BLd$), SiLU output ($4BLd$), $W_2$ output ($BLd$).
Final Norm & Head: $2 \cdot B \cdot L \cdot d$
Cross-entropy: $B \cdot L \cdot V$
Total Activation Memory = $4 \times { N \times [BL(16d + 2HL)] + BL(2d + V) }$ bytes.

(b) GPT-2 XL Memory Case Study
Configuration: $V=50257, L=1024, d=1600, N=48, H=25, d_{ff}=6400$.

Total Parameters ($P$): $\approx 2.127 \times 10^9$ (2.13B).
Static Memory ($16P$): $\approx 34.03 \text{ GB}$.
Activation Memory per Batch ($B=1$): $\approx 15.32 \text{ GB}$.
Memory Equation: $M \approx 15.32 \cdot B + 34.03 \text{ GB}$.

Max Batch Size for 80GB: $15.32B + 34.03 \le 80 \implies 15.32B \le 45.97 \implies B \le 2.99$. Maximum Batch Size = 2.

(c) Training FLOPs per Step
Running one step of AdamW involves a forward pass and a backward pass.

Forward Pass: $\approx 2P$ FLOPs per token.
Backward Pass: $\approx 4P$ FLOPs per token (roughly $2\times$ forward).
Optimizer Update: Negligible $O(P)$ compared to compute-heavy matrix multiplies.
Total FLOPs per step $\approx 6 \cdot P \cdot B \cdot L$. Justification: Standard heuristic where backward pass consumes $2\times$ the compute of forward pass to calculate gradients for both weights and activations.

(d) Training Time Estimation
Scenario: GPT-2 XL, 400K steps, Batch Size 1024, Context 1024, Single A100 (50% MFU).

Total Tokens: $4 \times 10^5 \times 1024 \times 1024 \approx 4.19 \times 10^{11}$ tokens.
Total FLOPs: $6 \times (2.127 \times 10^9) \times (4.19 \times 10^{11}) \approx 5.35 \times 10^{21}$ FLOPs.
Effective Throughput: $0.50 \times 19.5 \text{ TFLOPS} = 9.75 \times 10^{12} \text{ FLOP/s}$.
Training Time (seconds): $\approx 5.48 \times 10^8 \text{ seconds}$.
Training Time in Days: $\approx 6,343 \text{ days}$ (约 17.4 年). Justification: Training a 2.1B model on 400B tokens is a massive undertaking for a single GPU. Without parallelization, the sheer volume of floating-point operations exceeds the yearly capacity of one A100.