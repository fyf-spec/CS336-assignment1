!["transformer architecture"](transformer_architecture.png)

## resource(FLOPs) accounting:

#### 1. for a GPT-2 XL with params below:
vocab_size: 50257
context_length: 1024
d_model: 1600
num_layers: 48
d_ff: 6400
num_heads: 25


trainable parameters: 
vocab_size*d_model(token_embedding) + num_layers*(d_model*2 + d_model*d_model*4(Q,K,V plus output_proj)+ d_model*d_ff*3(w1,w2,w3 of FFN)) + d_model(ln_final) + vocab_size*d_model(ln_head)

required memories(single-precision):
4*#parameters

FLOPs(forward):
2 * #parameters * context_length