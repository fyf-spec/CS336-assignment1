import os
import subprocess
import time

def run_experiment(lr: float, run_name: str):
    """Run a single training experiment with the given learning rate."""
    cmd = [
        "uv", "run", "-m", "Experiments.train",
        "--train_data", "data/TinyStories_train.bin",
        "--val_data", "data/TinyStories_val.bin",
        "--experiment_name", run_name,
        "--description", f"Learning rate sweep: lr={lr}",
        "--vocab_size", "10000",
        "--context_length", "256",
        "--d_model", "512",
        "--num_layers", "4",
        "--num_heads", "16",
        "--d_ff", "1344",
        "--rope_theta", "10000",
        "--lr", str(lr),
        "--min_lr", "1e-4",
        "--warmup_iters", "200",
        "--cosine_cycle_iters", "5000",
        "--iterations", "5000",
        "--batch_size", "256", # Total tokens per step: 256 * 256 = 65,536 tokens. 5000 steps * 65536 = 327M tokens. Fits H100.
        "--log_interval", "10",
        "--eval_interval", "200",
        "--eval_iters", "20",
        "--checkpoint_interval", "1000",
        # "--no_wandb" # Removed no_wandb to allow logging on the H100
    ]
    
    print(f"[{time.strftime('%H:%M:%S')}] Starting run: {run_name} with lr={lr}")
    # We use stdout=subprocess.PIPE and stderr=subprocess.STDOUT to capture logs
    # and print them so we can monitor progress
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # Save the log output to a file
    os.makedirs("runs/sweep_logs", exist_ok=True)
    with open(f"runs/sweep_logs/{run_name}.log", "w", encoding="utf-8") as f:
        f.write(result.stdout)
        
    print(f"[{time.strftime('%H:%M:%S')}] Finished {run_name}. Exit code: {result.returncode}")
    if result.returncode != 0:
        print(f"WARNING: Run {run_name} failed. Check runs/sweep_logs/{run_name}.log")

def main():
    learning_rates = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    
    for lr in learning_rates:
        run_name = f"sweep_lr_{lr}"
        run_experiment(lr, run_name)
        
if __name__ == "__main__":
    main()
