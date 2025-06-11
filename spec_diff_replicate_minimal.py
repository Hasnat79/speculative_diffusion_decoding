import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class DiffusionConfig:
    vocab_size: int = 32000  # Standard vocabulary size
    max_seq_length: int = 512
    hidden_size: int = 768
    num_diffusion_steps: int = 20
    draft_length: int = 8  # γ parameter
    min_beta: float = 0.1
    max_beta: float = 20.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class DiscreteDistribution:
    """Handles discrete probability distributions and noise scheduling"""
    def __init__(self, config: DiffusionConfig):
        self.config = config
        # Create noise schedule (β_t)
        self.betas = torch.linspace(
            config.min_beta,
            config.max_beta,
            config.num_diffusion_steps,
            device=config.device
        )
        # Calculate alphas for diffusion process
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    
    def add_noise(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Add noise to token embeddings based on diffusion step t"""
        noise = torch.randn_like(x)
        t_long = t.long()  # Convert to long tensor for indexing
        sqrt_alphas_cumprod = self.alphas_cumprod[t_long].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod[t_long]).view(-1, 1, 1)
        return sqrt_alphas_cumprod * x + sqrt_one_minus_alphas_cumprod * noise

class DiffusionDrafter(nn.Module):
    """Discrete Diffusion Model for generating draft sequences"""
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        
        # Token embedding layer
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer layers for processing
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=8,
                dim_feedforward=config.hidden_size * 4,
                batch_first=True
            ),
            num_layers=6
        )
        
        # Time embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(1, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        
        # Output projection
        self.output_proj = nn.Linear(config.hidden_size, config.vocab_size)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Embed tokens
        x = self.token_embedding(x)
        
        # Time embedding
        t_emb = self.time_embedding(t.unsqueeze(-1).float())
        
        # Add time embedding to token embeddings
        x = x + t_emb.unsqueeze(1)
        
        # Process through transformer
        x = self.transformer(x)
        
        # Project to vocabulary distribution
        logits = self.output_proj(x)
        
        return logits

class TargetModel(nn.Module):
    """Dummy target model for verification (simplified for demonstration)"""
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=8,
                dim_feedforward=config.hidden_size * 4,
                batch_first=True
            ),
            num_layers=4
        )
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.vocab_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.transformer(x)
        return self.output(x)

class SpeculativeDiffusionModel:
    """Main class implementing speculative diffusion model inference"""
    def __init__(self, config: DiffusionConfig):
        self.config = config
        self.drafter = DiffusionDrafter(config).to(config.device)
        self.target_model = TargetModel(config).to(config.device)
        self.discrete_dist = DiscreteDistribution(config)
        
    def generate_draft(self, 
                      input_ids: torch.Tensor,
                      num_tokens: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate draft sequence using diffusion model"""
        batch_size = input_ids.shape[0]
        current_seq = input_ids
        
        # Initialize noise
        x_t = torch.randn(
            (batch_size, num_tokens, self.config.hidden_size),
            device=self.config.device
        )
        
        # Denoise through diffusion steps
        for t in reversed(range(self.config.num_diffusion_steps)):
            time_tensor = torch.ones(batch_size, device=self.config.device) * t
            logits = self.drafter(current_seq, time_tensor)
            x_t = self.discrete_dist.add_noise(logits, time_tensor)
        
        # Convert to probability distribution
        probs = F.softmax(x_t, dim=-1)
        # Sample tokens
        draft_tokens = torch.argmax(probs, dim=-1)
        
        return draft_tokens, probs

    def verify_draft(self,
                    input_ids: torch.Tensor,
                    draft_tokens: torch.Tensor,
                    draft_probs: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Verify draft tokens using target model"""
        with torch.no_grad():
            # Get only the draft part of target model's output
            concat_input = torch.cat([input_ids, draft_tokens], dim=1)
            target_logits = self.target_model(concat_input)
            # Extract only the draft portion predictions
            target_logits = target_logits[:, -draft_tokens.shape[1]:, :]
            target_probs = F.softmax(target_logits, dim=-1)
            
            # Get the probabilities for the selected draft tokens
            batch_size, seq_len = draft_tokens.shape
            batch_indices = torch.arange(batch_size).unsqueeze(1).expand(batch_size, seq_len)
            seq_indices = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len)
            draft_token_probs = draft_probs[batch_indices, seq_indices, draft_tokens]
            target_token_probs = target_probs[batch_indices, seq_indices, draft_tokens]
            
            # Calculate acceptance probabilities
            accept_probs = torch.min(
                torch.ones_like(draft_token_probs),
                target_token_probs / (draft_token_probs + 1e-8)
            )
            
            # Accept/reject tokens
            random_probs = torch.rand_like(accept_probs)
            accepted_mask = (random_probs < accept_probs).int()
            
            # Count accepted tokens
            num_accepted = int(accepted_mask.sum().item())
            
            # Create final sequence - for simplicity, we'll keep the original tokens
            # but mask out the rejected ones with zeros
            accepted_tokens = draft_tokens * accepted_mask
            
            return accepted_tokens, num_accepted

def generate_dummy_data(config: DiffusionConfig) -> torch.Tensor:
    """Generate dummy input data for testing"""
    batch_size = 4
    seq_length = 16
    return torch.randint(
        0, config.vocab_size,
        (batch_size, seq_length),
        device=config.device
    )

def main():
    # Initialize configuration
    config = DiffusionConfig()
    
    # Create model
    model = SpeculativeDiffusionModel(config)
    
    # Generate dummy input data
    input_ids = generate_dummy_data(config)
    print(f"Input shape: {input_ids.shape}")
    
    # Generate draft
    draft_tokens, draft_probs = model.generate_draft(input_ids, config.draft_length)
    print(f"Draft shape: {draft_tokens.shape}")
    
    # Verify draft
    accepted_tokens, num_accepted = model.verify_draft(
        input_ids, draft_tokens, draft_probs
    )
    print(f"Accepted tokens: {num_accepted}")
    print(f"Final sequence shape: {accepted_tokens.shape}")

if __name__ == "__main__":
    main()
