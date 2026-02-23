import argparse
import logging
import math
import os
import time
from typing import Optional

import numpy as np
import torch
from cs336_basics.nn import TransformerLM
from cs336_basics.optim import AdamW, get_lr_cosine_schedule, clip_gradient_norm
from cs336_basics.data import get_batch, save_checkpoint, load_checkpoint

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Transformer Language Model")
    
    # Model hyperparameters
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--context_length", type=int, default=1024)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    
    # Optimizer hyperparameters
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    
    # Schedule hyperparameters
    parser.add_argument("--warmup_iters", type=int, default=2000)
    parser.add_argument("--cosine_cycle_iters", type=int, default=600000)
    parser.add_argument("--min_lr", type=float, default=6e-5)
    
    # Training parameters
    parser.add_argument("--iterations", type=int, default=600000)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--eval_iters", type=int, default=200)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--checkpoint_interval", type=int, default=1000)
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    
    # Data parameters
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data .bin file")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation data .bin file")
    
    # Execution parameters
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1337)
    
    return parser.parse_args()

@torch.no_grad()
def estimate_loss(model, data, batch_size, context_length, eval_iters, device):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x, y = get_batch(data, batch_size, context_length, device)
        logits = model(x)
        B, T, C = logits.shape
        loss = torch.nn.functional.cross_entropy(logits.view(B*T, C), y.view(B*T))
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()

def main():
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create checkpoint directory
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    
    # Load data using np.memmap
    train_data = np.memmap(args.train_data, dtype=np.uint16, mode='r')
    val_data = np.memmap(args.val_data, dtype=np.uint16, mode='r')
    
    logger.info(f"Loaded training data: {len(train_data)} tokens")
    logger.info(f"Loaded validation data: {len(val_data)} tokens")
    
    # Initialize model
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=torch.device(args.device)
    )
    model.to(args.device)
    
    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay
    )
    
    start_iter = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        start_iter = load_checkpoint(args.resume, model, optimizer)
    
    # Pre-train evaluation
    val_loss = estimate_loss(model, val_data, args.batch_size, args.context_length, args.eval_iters, args.device)
    logger.info(f"Initial Validation Loss: {val_loss:.4f}, Perplexity: {math.exp(val_loss):.4f}")
    
    # Training loop
    model.train()
    start_time = time.time()
    
    for it in range(start_iter, args.iterations):
        # Update learning rate
        lr = get_lr_cosine_schedule(
            it, 
            max_learning_rate=args.lr, 
            min_learning_rate=args.min_lr, 
            warmup_iters=args.warmup_iters, 
            cosine_cycle_iters=args.cosine_cycle_iters
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        # Sample batch
        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
        
        # Forward pass
        logits = model(x)
        B, T, C = logits.shape
        # Flatten for cross entropy
        loss = torch.nn.functional.cross_entropy(logits.view(B*T, C), y.view(B*T))
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Gradient clipping
        if args.grad_clip != 0.0:
            clip_gradient_norm(model.parameters(), args.grad_clip)
            
        # Optimizer step
        optimizer.step()
        
        # Logging
        if (it + 1) % args.log_interval == 0:
            iter_time = (time.time() - start_time) / args.log_interval
            logger.info(f"Iter {it+1}/{args.iterations}: loss {loss.item():.4f}, lr {lr:.2e}, time {iter_time*1000:.2f}ms")
            start_time = time.time()
            
        # Evaluation
        if (it + 1) % args.eval_interval == 0:
            val_loss = estimate_loss(model, val_data, args.batch_size, args.context_length, args.eval_iters, args.device)
            logger.info(f"Iter {it+1}: Validation Loss: {val_loss:.4f}, Perplexity: {math.exp(val_loss):.4f}")
            
        # Checkpointing
        if (it + 1) % args.checkpoint_interval == 0:
            ckpt_name = os.path.join(args.checkpoint_path, f"ckpt_iter_{it+1}.pt")
            save_checkpoint(model, optimizer, it + 1, ckpt_name)
            logger.info(f"Saved checkpoint to {ckpt_name}")

    # Final checkpoint
    save_checkpoint(model, optimizer, args.iterations, os.path.join(args.checkpoint_path, "ckpt_final.pt"))
    logger.info("Training complete.")

if __name__ == "__main__":
    main()
