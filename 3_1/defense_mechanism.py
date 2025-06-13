# defense_mechanism.py
"""
Defense Mechanism Against Model Extraction Attacks
=================================================
This module implements various defense strategies to protect against
model extraction attacks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, List
from collections import defaultdict, deque
import time

class DefenseMechanism:
    """Base defense mechanism with multiple strategies"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.base_noise_scale = config.get('base_noise_scale', 0.01)
        self.max_noise_scale = config.get('max_noise_scale', 0.1)
        self.ood_threshold = config.get('ood_threshold', 0.85)
        self.entropy_threshold = config.get('entropy_threshold', 0.7)
        self.block_threshold = config.get('block_threshold', 0.7)
        self.deception_probability = config.get('deception_probability', 0.0)
        self.pattern_threshold = config.get('pattern_threshold', 0.3)
        self.adaptive_factor = config.get('adaptive_factor', 2.0)
        
        # Statistics tracking
        self.query_count = 0
        self.perturbed_responses = 0
        self.blocked_queries = 0
        self.deceptive_responses = 0
        
        # Pattern detection
        self.recent_queries = deque(maxlen=100)
        self.query_patterns = defaultdict(int)
        
        # OOD detection (will be fitted)
        self.ood_detector = None
        self.in_distribution_stats = None
        
    def fit_ood_detector(self, train_loader):
        """Fit out-of-distribution detector on training data"""
        features = []
        
        with torch.no_grad():
            for x, _ in train_loader:
                # Simple feature extraction - flatten images
                feat = x.flatten(1).cpu().numpy()
                features.append(feat)
        
        features = np.concatenate(features, axis=0)
        
        # Store statistics for OOD detection
        self.in_distribution_stats = {
            'mean': np.mean(features, axis=0),
            'std': np.std(features, axis=0) + 1e-8,  # avoid division by zero
            'percentiles': np.percentile(features, [5, 95], axis=0)
        }
        
    def detect_ood(self, x: torch.Tensor) -> torch.Tensor:
        """Detect out-of-distribution samples"""
        if self.in_distribution_stats is None:
            return torch.zeros(x.size(0), dtype=torch.bool)
        
        x_flat = x.flatten(1).cpu().numpy()
        
        # Z-score based detection
        z_scores = np.abs((x_flat - self.in_distribution_stats['mean']) / 
                         self.in_distribution_stats['std'])
        max_z_scores = np.max(z_scores, axis=1)
        
        # Percentile based detection
        p5, p95 = self.in_distribution_stats['percentiles']
        outside_percentiles = np.mean((x_flat < p5) | (x_flat > p95), axis=1)
        
        # Combine criteria
        ood_mask = (max_z_scores > 3.0) | (outside_percentiles > 0.1)
        
        return torch.tensor(ood_mask, dtype=torch.bool)
    
    def detect_patterns(self, x: torch.Tensor) -> float:
        """Detect suspicious query patterns"""
        # Simple hash-based pattern detection
        x_hash = hash(x.flatten().cpu().numpy().tobytes())
        self.recent_queries.append(x_hash)
        self.query_patterns[x_hash] += 1
        
        # Check for repeated queries
        if len(self.recent_queries) > 10:
            unique_ratio = len(set(self.recent_queries)) / len(self.recent_queries)
            return 1.0 - unique_ratio
        
        return 0.0
    
    def calculate_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """Calculate prediction entropy"""
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        probs = torch.clamp(probs, eps, 1.0 - eps)
        return -torch.sum(probs * torch.log(probs), dim=1)
    
    def add_noise(self, probs: torch.Tensor, noise_scale: float) -> torch.Tensor:
        """Add calibrated noise to predictions"""
        if noise_scale <= 0:
            return probs
        
        # Add Gaussian noise
        noise = torch.randn_like(probs) * noise_scale
        noisy_probs = probs + noise
        
        # Renormalize to valid probabilities
        noisy_probs = torch.clamp(noisy_probs, 1e-8, 1.0)
        noisy_probs = F.softmax(noisy_probs / 0.1, dim=1)  # Temperature scaling
        
        return noisy_probs
    
    def generate_deceptive_response(self, probs: torch.Tensor) -> torch.Tensor:
        """Generate deceptive but plausible response"""
        batch_size, num_classes = probs.shape
        
        # Create uniform-ish distribution with some randomness
        deceptive = torch.ones_like(probs) / num_classes
        
        # Add some controlled randomness
        noise = torch.randn_like(probs) * 0.1
        deceptive = deceptive + noise
        deceptive = F.softmax(deceptive, dim=1)
        
        return deceptive
    
    def should_block(self, suspicion_score: float) -> bool:
        """Decide whether to block query based on suspicion"""
        return suspicion_score > self.block_threshold
    
    def should_deceive(self, suspicion_score: float) -> bool:
        """Decide whether to return deceptive response"""
        base_prob = self.deception_probability
        # Increase deception probability with suspicion
        adjusted_prob = min(base_prob * (1 + suspicion_score), 0.8)
        return np.random.random() < adjusted_prob
    
    def defend(self, x: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """Apply defense mechanisms to query response"""
        self.query_count += 1
        batch_size = x.size(0)
        
        # Initialize suspicion score
        suspicion_scores = torch.zeros(batch_size)
        
        # 1. Out-of-distribution detection
        ood_mask = self.detect_ood(x)
        suspicion_scores[ood_mask] += 0.5
        
        # 2. Entropy-based detection
        entropies = self.calculate_entropy(probs)
        low_entropy_mask = entropies < self.entropy_threshold
        suspicion_scores[low_entropy_mask] += 0.3
        
        # 3. Pattern detection (applied to batch average)
        pattern_score = self.detect_patterns(x[0])  # Use first sample as representative
        if pattern_score > self.pattern_threshold:
            suspicion_scores += pattern_score * 0.4
        
        # Apply defenses based on suspicion
        defended_probs = probs.clone()
        
        for i in range(batch_size):
            suspicion = suspicion_scores[i].item()
            
            # Check if should block (return None/error - simplified as uniform)
            if self.should_block(suspicion):
                defended_probs[i] = torch.ones(probs.size(1)) / probs.size(1)
                self.blocked_queries += 1
                continue
            
            # Check if should deceive
            if self.should_deceive(suspicion):
                defended_probs[i] = self.generate_deceptive_response(probs[i:i+1])[0]
                self.deceptive_responses += 1
                continue
            
            # Add noise based on suspicion
            base_noise = self.base_noise_scale
            adaptive_noise = min(base_noise * (1 + suspicion * self.adaptive_factor), 
                               self.max_noise_scale)
            
            if adaptive_noise > 0:
                defended_probs[i] = self.add_noise(probs[i:i+1], adaptive_noise)[0]
                self.perturbed_responses += 1
        
        return defended_probs
    
    def get_stats(self) -> Dict:
        """Get defense statistics"""
        if self.query_count == 0:
            return {'query_count': 0}
        
        return {
            'query_count': self.query_count,
            'perturbed_responses': self.perturbed_responses,
            'blocked_queries': self.blocked_queries,
            'deceptive_responses': self.deceptive_responses,
            'perturbation_rate': self.perturbed_responses / self.query_count,
            'block_rate': self.blocked_queries / self.query_count,
            'deception_rate': self.deceptive_responses / self.query_count
        }


class DefendedBlackBoxAPI:
    """Black-box API with integrated defense mechanisms"""
    
    def __init__(self, victim: nn.Module, defense_config: Dict):
        self.victim = victim
        self.defense = DefenseMechanism(defense_config)
        
    def fit_defense(self, train_loader):
        """Fit defense mechanisms on training data"""
        self.defense.fit_ood_detector(train_loader)
        
    @torch.no_grad()
    def query(self, x: torch.Tensor, *, logits=False) -> torch.Tensor:
        """Query with defense mechanisms applied"""
        # Get victim predictions
        victim_logits = self.victim(x.to(next(self.victim.parameters()).device))
        victim_probs = torch.exp(victim_logits)  # My_MNIST returns log-probs
        
        # Apply defenses
        defended_probs = self.defense.defend(x, victim_probs)
        
        # Return in requested format
        if logits:
            return torch.log(defended_probs + 1e-8)  # Convert back to log-probs
        else:
            return defended_probs
    
    def get_defense_report(self) -> Dict:
        """Get comprehensive defense report"""
        return self.defense.get_stats()