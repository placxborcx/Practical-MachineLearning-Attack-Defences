# defense_mechanism_fixed.py
"""
Fixed Defense Mechanism Against Model-Extraction Attacks
========================================================
Key fixes:
1. Lower baseline suspicion and thresholds for better sensitivity
2. Improved OOD detection using multiple features
3. Better controlled output release
4. Enhanced query pattern analysis
"""

from __future__ import annotations
import logging
import hashlib
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import entropy
from sklearn.covariance import EllipticEnvelope

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DefenseMechanism:
    """Multi-layer defense wrapper for a victim model."""

    def __init__(
        self,
        victim_model: nn.Module,
        device: torch.device,
        defense_config: Optional[Dict] = None,
    ):
        self.victim_model = victim_model
        self.device = device

        # Fixed configuration with lower thresholds
        self.config: Dict = {
            # --- Controlled release --------------------------------------
            "top_k": 3,  # Release top-3 logits with noise
            # --- Suspicion baseline & thresholds -------------------------
            "baseline_suspicion": 0.15,  # Increased from 0.05
            "ood_threshold": 0.70,       # Lowered from 0.85
            "ood_contamination": 0.10,
            "entropy_threshold": 2.0,     # Changed to absolute entropy
            "perturb_threshold": 0.25,    # Lowered from 0.40
            "block_threshold": 0.70,      # Increased from 0.60
            # --- Perturbation scales -------------------------------------
            "base_noise_scale": 0.02,     # Increased
            "max_noise_scale": 0.30,      # Increased
            "temperature_base": 1.5,      # New: base temperature
            # --- Pattern-analysis window ---------------------------------
            "sequence_length": 100,
            "similarity_threshold": 0.85,
            "query_rate_window": 10,      # New: for rate limiting
            "max_queries_per_window": 50, # New: rate limit
        }
        if defense_config:
            self.config.update(defense_config)

        # Initialize components
        self._initialize_components()

        # Enhanced stats tracking
        self.total_queries = 0
        self.blocked_queries = 0
        self.perturbed_responses = 0
        self.query_history: deque = deque(maxlen=self.config["sequence_length"])
        self.query_fingerprints: defaultdict[str, int] = defaultdict(int)
        self.query_timestamps: deque = deque(maxlen=self.config["query_rate_window"])
        
        # Track suspicion scores for debugging
        self.suspicion_history: deque = deque(maxlen=100)

        logger.info("Fixed Defense mechanism initialized")

    def _initialize_components(self) -> None:
        """Prepare feature extractor & OOD detector shell."""
        self.feature_extractor = self._create_feature_extractor()
        self.ood_detector: Optional[EllipticEnvelope] = None
        self.training_features_mean = None
        self.training_features_std = None

    def _create_feature_extractor(self) -> nn.Module:
        """Return a wrapper that exposes penultimate activations."""

        class _Extractor(nn.Module):
            def __init__(self, base: nn.Module):
                super().__init__()
                self.base = base
                self._features = None
                self._hook = None

                # Find last Linear layer
                self._last_linear = None
                for m in reversed(list(base.modules())):
                    if isinstance(m, nn.Linear):
                        self._last_linear = m
                        break

            def forward(self, x):
                def hook_fn(_, inp, __):
                    self._features = inp[0] if isinstance(inp, tuple) else inp

                if self._hook is not None:
                    self._hook.remove()
                self._hook = self._last_linear.register_forward_hook(hook_fn)

                out = self.base(x)
                self._hook.remove()
                return self._features, out

        return _Extractor(self.victim_model).to(self.device)

    @torch.no_grad()
    def fit_ood_detector(self, train_loader) -> None:
        """Fit OOD detector and collect training statistics."""
        feats = []
        self.feature_extractor.eval()
        
        logger.info("Fitting OOD detector...")
        for idx, (x, _) in enumerate(train_loader):
            if idx >= 50:  # Use more samples for better statistics
                break
            f, _ = self.feature_extractor(x.to(self.device))
            feats.append(f.cpu().numpy())
        
        feats = np.concatenate(feats, axis=0)
        
        # Store training statistics
        self.training_features_mean = feats.mean(axis=0)
        self.training_features_std = feats.std(axis=0) + 1e-8
        
        # Fit OOD detector
        self.ood_detector = EllipticEnvelope(
            contamination=self.config["ood_contamination"], 
            random_state=42
        ).fit(feats)
        
        logger.info(f"OOD detector fitted on {len(feats)} samples")

    @torch.no_grad()
    def _detect_ood(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Enhanced OOD detection with multiple signals."""
        if self.ood_detector is None:
            return np.zeros(len(x)), np.zeros(len(x), dtype=bool)

        feats, _ = self.feature_extractor(x)
        feats_np = feats.cpu().numpy()
        
        # Standard OOD scores
        ood_scores = -self.ood_detector.score_samples(feats_np)
        is_ood = self.ood_detector.predict(feats_np) == -1
        
        # Additional: distance from training distribution
        if self.training_features_mean is not None:
            normalized_feats = (feats_np - self.training_features_mean) / self.training_features_std
            distance_scores = np.linalg.norm(normalized_feats, axis=1)
            # Combine with OOD scores
            ood_scores = 0.7 * ood_scores + 0.3 * (distance_scores / (distance_scores.mean() + 1e-8))
        
        return ood_scores, is_ood

    def _analyze_query_patterns(self, x: torch.Tensor) -> float:
        #Enhanced pattern analysis with multiple signals.
        
        fps = self._generate_fingerprints(x)
        
        # 1. Duplicate detection
        dup_count = sum(self.query_fingerprints[fp] > 1 for fp in fps)
        dup_ratio = dup_count / len(fps)
        
        # 2. Query diversity (entropy of fingerprints)
        if len(self.query_fingerprints) > 10:
            fp_counts = np.array(list(self.query_fingerprints.values()))
            fp_probs = fp_counts / fp_counts.sum()
            fp_entropy = entropy(fp_probs)
            # Low entropy = repetitive queries
            diversity_score = 1.0 - (fp_entropy / np.log(len(self.query_fingerprints) + 1))
        else:
            diversity_score = 0.0
        
        # Update fingerprint counts
        for fp in fps:
            self.query_fingerprints[fp] += 1
        
        
        # 3. Query rate analysis (CPU/GPU safe)
        if torch.cuda.is_available():
            # GPU -> cuda event
            evt = torch.cuda.Event(enable_timing=True)
            evt.record()
            self.query_timestamps.append(evt)
        else:
            # CPU -> use perf_counter
            self.query_timestamps.append(time.perf_counter())

        if len(self.query_timestamps) >= self.config["max_queries_per_window"]:
            rate_score = 0.5
        else:
            rate_score = 0.0

        # Combine signals
        pattern_score = 0.4 * dup_ratio + 0.4 * diversity_score + 0.2 * rate_score
        
        return min(pattern_score, 1.0)

    @staticmethod
    def _generate_fingerprints(x: torch.Tensor) -> List[str]:
        """Generate fingerprints for query detection."""
        fps = []
        flat = x.cpu().numpy().reshape(len(x), -1)
        for row in flat:
            # Use more aggressive downsampling for better duplicate detection
            sample = np.round(row[::5], 1).tobytes()
            fps.append(hashlib.md5(sample).hexdigest()[:16])  # Shorter hash
        return fps

    def _calculate_suspicion_score(
        self, x: torch.Tensor, ood_scores: np.ndarray, pattern_score: float
    ) -> float:
        """Enhanced suspicion scoring with multiple signals."""
        # 1. OOD component (more sensitive)
        ood_component = (ood_scores > self.config["ood_threshold"]).mean() * 0.3
        
        # 2. Pattern component
        pattern_component = pattern_score * 0.3
        
        # 3. Entropy analysis (more sophisticated)
        with torch.no_grad():
            _, logits = self.feature_extractor(x)
            probs = F.softmax(logits, dim=1)
            
            # Batch entropy
            batch_entropy = -(probs * torch.log(probs + 1e-8)).sum(1).mean().item()
            
            # Check for suspiciously low entropy (overconfident predictions)
            if batch_entropy < self.config["entropy_threshold"]:
                entropy_component = 0.2
            else:
                entropy_component = 0.0
            
            # 4. Confidence uniformity (new signal)
            max_probs = probs.max(dim=1)[0]
            confidence_std = max_probs.std().item()
            if confidence_std < 0.1:  # Suspiciously uniform confidence
                uniformity_component = 0.2
            else:
                uniformity_component = 0.0
        
        score = (
            self.config["baseline_suspicion"]
            + ood_component
            + pattern_component
            + entropy_component
            + uniformity_component
        )
        
        # Store for debugging
        self.suspicion_history.append(score)
        
        return min(score, 1.0)

    def _controlled_release(self, outputs: torch.Tensor, k: int) -> torch.Tensor:
        """Improved controlled release that adds noise instead of zeroing."""
        batch_size = outputs.size(0)
        
        # Get top-k values and indices
        topk_values, topk_indices = torch.topk(outputs, k=k, dim=1)
        
        # Create noise for non-top-k values
        noise_scale = 0.1
        noise = torch.randn_like(outputs) * noise_scale
        
        # Create output tensor
        controlled = outputs.clone()
        
        # Add noise to non-top-k values
        mask = torch.ones_like(outputs, dtype=torch.bool)
        mask.scatter_(1, topk_indices, False)
        controlled[mask] += noise[mask]
        
        # Ensure top-1 remains unchanged
        top1_indices = topk_indices[:, 0].unsqueeze(1)
        controlled.scatter_(1, top1_indices, topk_values[:, 0].unsqueeze(1))
        
        return F.log_softmax(controlled, dim=1)

    def _apply_deceptive_perturbation(
        self, outputs: torch.Tensor, suspicion: float
    ) -> torch.Tensor:
        """Apply graduated perturbation based on suspicion level."""
        if suspicion < self.config["perturb_threshold"]:
            return outputs
        
        # Calculate perturbation strength
        perturb_strength = (suspicion - self.config["perturb_threshold"]) / (
            1.0 - self.config["perturb_threshold"]
        )
        
        # 1. Temperature scaling
        T = self.config["temperature_base"] + 2.0 * perturb_strength
        scaled_logits = outputs / T
        
        # 2. Add calibrated noise
        noise_scale = self.config["base_noise_scale"] + (
            self.config["max_noise_scale"] - self.config["base_noise_scale"]
        ) * perturb_strength
        noise = torch.randn_like(outputs) * noise_scale
        
        # 3. Mix with uniform distribution for high suspicion
        if suspicion > 0.5:
            uniform_logits = torch.ones_like(outputs) * (-np.log(outputs.size(1)))
            alpha = 0.3 * perturb_strength
            perturbed = (1 - alpha) * (scaled_logits + noise) + alpha * uniform_logits
        else:
            perturbed = scaled_logits + noise
        
        self.perturbed_responses += outputs.size(0)
        
        return F.log_softmax(perturbed, dim=1)

    def defend(self, x: torch.Tensor) -> torch.Tensor:
        """Main defense entry point."""
        self.total_queries += x.size(0)
        
        # Detection & scoring
        ood_scores, _ = self._detect_ood(x)
        pattern_score = self._analyze_query_patterns(x)
        suspicion = self._calculate_suspicion_score(x, ood_scores, pattern_score)
        
        # Log suspicion score for debugging
        if len(self.suspicion_history) % 10 == 0:
            avg_suspicion = (
                float(np.mean(self.suspicion_history))
                if self.suspicion_history else 0.0
            )
            logger.debug(f"Average suspicion score: {avg_suspicion:.3f}")
        
        # Get victim outputs
        with torch.no_grad():
            _, raw = self.feature_extractor(x)
        
        # Apply controlled release
        controlled = self._controlled_release(raw, self.config["top_k"])
        
        # Apply perturbation based on suspicion
        if suspicion > self.config["block_threshold"]:
            self.blocked_queries += x.size(0)
            logger.warning(f"High suspicion query (score {suspicion:.3f})")
            # Return heavily perturbed output
            return self._apply_deceptive_perturbation(controlled, 1.0)
        
        defended = self._apply_deceptive_perturbation(controlled, suspicion)
        
        return defended

    def get_defense_summary(self) -> Dict:
        """Get defense statistics."""
        avg_suspicion = np.mean(list(self.suspicion_history)) if self.suspicion_history else 0
        return {
            "total_queries": self.total_queries,
            "blocked_queries": self.blocked_queries,
            "perturbed_responses": self.perturbed_responses,
            "unique_query_patterns": len(self.query_fingerprints),
            "average_suspicion": float(avg_suspicion),
            "block_rate": self.blocked_queries / max(1, self.total_queries),
            "perturb_rate": self.perturbed_responses / max(1, self.total_queries),
        }


class DefendedBlackBoxAPI:
    """Drop-in replacement for BlackBoxAPI with defense."""

    def __init__(self, victim: nn.Module, defense_config: Optional[Dict] = None):
        self.victim = victim
        self.device = next(victim.parameters()).device
        self.defense = DefenseMechanism(victim, self.device, defense_config)
        self.api_calls = 0

    @torch.no_grad()
    def query(self, x: torch.Tensor, *, logits: bool = False) -> torch.Tensor:
        """Route query through defense."""
        self.api_calls += 1
        defended = self.defense.defend(x.to(self.device))
        return defended if logits else torch.exp(defended)

    def fit_defense(self, train_loader) -> None:
        """Fit the OOD detector."""
        self.defense.fit_ood_detector(train_loader)

    def get_defense_report(self) -> Dict:
        """Get comprehensive defense report."""
        rep = self.defense.get_defense_summary()
        rep["api_calls"] = self.api_calls
        return rep