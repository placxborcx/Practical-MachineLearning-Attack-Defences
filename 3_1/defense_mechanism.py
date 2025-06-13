# defense_mechanism.py
"""
Comprehensive Defense Mechanism Against Model Extraction Attacks
================================================================
Combines multiple defense strategies:
1. Malicious Query Detection (OOD-based and PRADA-inspired)
2. Adaptive Deceptive Perturbation
3. Query Pattern Analysis
4. Dynamic Response Strategy
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, deque
import hashlib
import time
from scipy.stats import entropy
from sklearn.covariance import EllipticEnvelope
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DefenseMechanism:
    """
    Comprehensive defense system against model extraction attacks.
    
    Key Components:
    1. OOD Detection: Identifies out-of-distribution queries
    2. PRADA-inspired: Detects adversarial patterns
    3. Adaptive Perturbation: Intelligently adds noise based on suspicion level
    4. Query Pattern Analysis: Monitors query sequences for suspicious patterns
    """
    
    def __init__(self, 
                 victim_model: nn.Module,
                 device: torch.device,
                 defense_config: Optional[Dict] = None):
        """
        Initialize defense mechanism.
        
        Args:
            victim_model: The model to protect
            device: Computing device
            defense_config: Configuration parameters
        """
        self.victim_model = victim_model
        self.device = device
        
        # Default configuration
        self.config = {
            # OOD Detection
            'ood_threshold': 0.85,
            'ood_contamination': 0.1,  # Expected fraction of outliers
            
            # PRADA-inspired parameters
            'entropy_threshold': 0.7,
            'confidence_threshold': 0.95,
            
            # Perturbation parameters
            'base_noise_scale': 0.01,
            'max_noise_scale': 0.2,
            'adaptive_factor': 2.0,
            
            # Pattern detection
            'sequence_length': 100,
            'similarity_threshold': 0.9,
            'pattern_threshold': 0.3,
            
            # Response strategy
            'suspicion_decay': 0.95,
            'block_threshold': 0.8,
            'deception_probability': 0.3
        }
        
        if defense_config:
            self.config.update(defense_config)
        
        # Initialize components
        self._initialize_components()
        
        # Tracking variables
        self.query_history = deque(maxlen=self.config['sequence_length'])
        self.suspicion_scores = defaultdict(float)
        self.query_fingerprints = defaultdict(int)
        self.total_queries = 0
        self.blocked_queries = 0
        self.perturbed_responses = 0
        
        logger.info("Defense mechanism initialized with adaptive multi-layer protection")
    
    def _initialize_components(self):
        """Initialize defense components"""
        # OOD detector (will be fitted with training data)
        self.ood_detector = None
        self.training_features = None
        
        # Feature extractor for OOD detection
        self.feature_extractor = self._create_feature_extractor()
        
        # Pattern analyzer
        self.pattern_buffer = []
        self.detected_patterns = set()
        
    def _create_feature_extractor(self) -> nn.Module:
        """Create a feature extraction network for OOD detection"""
        class FeatureExtractor(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model
                
                # Hook to extract intermediate features
                self.features = None
                self.hook = None
                
            def forward(self, x):
                # Extract features from penultimate layer
                def hook_fn(module, input, output):
                    self.features = input[0] if isinstance(input, tuple) else input
                
                # Register hook on the last linear layer
                for name, module in self.base_model.named_modules():
                    if isinstance(module, nn.Linear):
                        last_linear = module
                
                if self.hook is not None:
                    self.hook.remove()
                self.hook = last_linear.register_forward_hook(hook_fn)
                
                output = self.base_model(x)
                
                if self.hook is not None:
                    self.hook.remove()
                
                return self.features, output
        
        return FeatureExtractor(self.victim_model).to(self.device)
    
    def fit_ood_detector(self, train_loader):
        """
        Fit the OOD detector using training data.
        
        Args:
            train_loader: DataLoader containing training data
        """
        logger.info("Fitting OOD detector with training data...")
        
        features_list = []
        self.feature_extractor.eval()
        
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(train_loader):
                if batch_idx > 50:  # Use subset for efficiency
                    break
                    
                data = data.to(self.device)
                features, _ = self.feature_extractor(data)
                features_list.append(features.cpu().numpy())
        
        # Concatenate and fit OOD detector
        self.training_features = np.concatenate(features_list, axis=0)
        self.ood_detector = EllipticEnvelope(
            contamination=self.config['ood_contamination'],
            random_state=42
        )
        self.ood_detector.fit(self.training_features)
        
        logger.info(f"OOD detector fitted with {len(self.training_features)} samples")
    
    def _detect_ood(self, inputs: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect out-of-distribution inputs.
        
        Args:
            inputs: Query inputs
            
        Returns:
            ood_scores: OOD scores for each input
            is_ood: Boolean array indicating OOD inputs
        """
        if self.ood_detector is None:
            return np.zeros(len(inputs)), np.zeros(len(inputs), dtype=bool)
        
        with torch.no_grad():
            features, _ = self.feature_extractor(inputs)
            features_np = features.cpu().numpy()
        
        # Get OOD scores
        ood_scores = -self.ood_detector.score_samples(features_np)
        is_ood = self.ood_detector.predict(features_np) == -1
        
        return ood_scores, is_ood
    
    def _analyze_query_patterns(self, inputs: torch.Tensor) -> float:
        """
        Analyze query patterns for suspicious behavior.
        
        Args:
            inputs: Current query batch
            
        Returns:
            pattern_score: Suspicion score based on patterns (0-1)
        """
        pattern_score = 0.0
        
        # 1. Check for duplicate/similar queries
        current_fingerprints = self._generate_fingerprints(inputs)
        duplicate_ratio = sum(1 for fp in current_fingerprints 
                             if self.query_fingerprints[fp] > 2) / len(current_fingerprints)
        pattern_score += duplicate_ratio * 0.3
        
        # Update fingerprint counts
        for fp in current_fingerprints:
            self.query_fingerprints[fp] += 1
        
        # 2. Analyze query sequence entropy
        if len(self.query_history) >= 10:
            recent_queries = list(self.query_history)[-10:]
            sequence_entropy = self._calculate_sequence_entropy(recent_queries)
            if sequence_entropy < self.config['entropy_threshold']:
                pattern_score += 0.3
        
        # 3. Check for adversarial patterns (high variance, structured noise)
        variance = torch.var(inputs).item()
        mean_abs = torch.mean(torch.abs(inputs)).item()
        if variance > 0.8 and mean_abs > 0.4:
            pattern_score += 0.4
        
        return min(pattern_score, 1.0)
    
    def _generate_fingerprints(self, inputs: torch.Tensor) -> List[str]:
        """Generate fingerprints for input queries"""
        fingerprints = []
        for i in range(inputs.size(0)):
            # Use a subset of pixels for fingerprinting
            data_subset = inputs[i].cpu().numpy().flatten()[::10]
            data_rounded = np.round(data_subset, decimals=2)
            fp = hashlib.md5(data_rounded.tobytes()).hexdigest()
            fingerprints.append(fp)
        return fingerprints
    
    def _calculate_sequence_entropy(self, queries: List[torch.Tensor]) -> float:
        """Calculate entropy of query sequence"""
        # Simple entropy based on query diversity
        fingerprints = []
        for q in queries:
            fps = self._generate_fingerprints(q)
            fingerprints.extend(fps)
        
        # Calculate frequency distribution
        unique, counts = np.unique(fingerprints, return_counts=True)
        probs = counts / counts.sum()
        return entropy(probs)
    
    def _calculate_suspicion_score(self, 
                                  inputs: torch.Tensor,
                                  ood_scores: np.ndarray,
                                  pattern_score: float) -> float:
        """
        Calculate overall suspicion score for the query.
        
        Args:
            inputs: Query inputs
            ood_scores: OOD detection scores
            pattern_score: Pattern analysis score
            
        Returns:
            suspicion_score: Overall suspicion score (0-1)
        """
        # Combine different detection mechanisms
        ood_component = np.mean(ood_scores > self.config['ood_threshold']) * 0.4
        pattern_component = pattern_score * 0.4
        
        # PRADA-inspired: Check prediction entropy
        with torch.no_grad():
            _, outputs = self.feature_extractor(inputs)
            probs = F.softmax(outputs, dim=1)
            pred_entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            entropy_component = (pred_entropy.mean() < self.config['entropy_threshold']) * 0.2
        
        suspicion_score = ood_component + pattern_component + entropy_component.item()
        
        return min(suspicion_score, 1.0)
    
    def _apply_deceptive_perturbation(self, 
                                     outputs: torch.Tensor,
                                     suspicion_score: float) -> torch.Tensor:
        """
        Apply adaptive deceptive perturbation based on suspicion level.
        
        Args:
            outputs: Model outputs (log probabilities)
            suspicion_score: Current suspicion score
            
        Returns:
            perturbed_outputs: Strategically perturbed outputs
        """
        if suspicion_score < 0.2:
            # Low suspicion: minimal or no perturbation
            return outputs
        
        # Calculate adaptive noise scale
        noise_scale = self.config['base_noise_scale'] + \
                     (self.config['max_noise_scale'] - self.config['base_noise_scale']) * suspicion_score
        
        # Apply different perturbation strategies based on suspicion level
        if suspicion_score < 0.5:
            # Medium suspicion: Add calibrated noise
            noise = torch.randn_like(outputs) * noise_scale
            perturbed = outputs + noise
            
        elif suspicion_score < 0.8:
            # High suspicion: More aggressive perturbation
            # Reduce confidence of top predictions
            probs = F.softmax(outputs, dim=1)
            top_k = 3
            topk_indices = torch.topk(probs, top_k, dim=1).indices
            
            perturbed = outputs.clone()
            for i in range(outputs.size(0)):
                # Reduce top-k logits
                perturbed[i, topk_indices[i]] -= noise_scale * 2
                # Add noise to other classes
                mask = torch.ones_like(perturbed[i], dtype=bool)
                mask[topk_indices[i]] = False
                perturbed[i, mask] += torch.randn(mask.sum()) * noise_scale
        
        else:
            # Very high suspicion: Deceptive responses
            if np.random.random() < self.config['deception_probability']:
                # Occasionally return random predictions
                perturbed = torch.randn_like(outputs)
            else:
                # Heavily perturb outputs
                perturbed = outputs + torch.randn_like(outputs) * self.config['max_noise_scale'] * 2
        
        self.perturbed_responses += outputs.size(0)
        return perturbed
    
    def defend(self, inputs: torch.Tensor, return_metrics: bool = False) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Main defense interface - process queries through defense mechanism.
        
        Args:
            inputs: Query inputs
            return_metrics: Whether to return defense metrics
            
        Returns:
            outputs: Defended outputs (potentially perturbed)
            metrics: Optional defense metrics
        """
        self.total_queries += inputs.size(0)
        
        # 1. OOD Detection
        ood_scores, is_ood = self._detect_ood(inputs)
        
        # 2. Pattern Analysis
        pattern_score = self._analyze_query_patterns(inputs)
        
        # 3. Calculate suspicion score
        suspicion_score = self._calculate_suspicion_score(inputs, ood_scores, pattern_score)
        
        # 4. Update tracking
        self.query_history.append(inputs)
        
        # 5. Get model outputs
        with torch.no_grad():
            _, outputs = self.feature_extractor(inputs)
        
        # 6. Apply defense strategy based on suspicion level
        if suspicion_score > self.config['block_threshold']:
            # Very suspicious: consider blocking or heavy perturbation
            self.blocked_queries += inputs.size(0)
            logger.warning(f"High suspicion detected: {suspicion_score:.3f}")
            
            # Apply heavy perturbation instead of blocking
            defended_outputs = self._apply_deceptive_perturbation(outputs, suspicion_score)
        else:
            # Apply adaptive perturbation
            defended_outputs = self._apply_deceptive_perturbation(outputs, suspicion_score)
        
        # 7. Prepare metrics if requested
        metrics = None
        if return_metrics:
            metrics = {
                'suspicion_score': suspicion_score,
                'ood_ratio': np.mean(is_ood),
                'pattern_score': pattern_score,
                'total_queries': self.total_queries,
                'blocked_queries': self.blocked_queries,
                'perturbed_responses': self.perturbed_responses,
                'unique_patterns': len(self.query_fingerprints)
            }
        
        return defended_outputs, metrics
    
    def get_defense_summary(self) -> Dict:
        """Get summary of defense performance"""
        return {
            'total_queries_processed': self.total_queries,
            'blocked_queries': self.blocked_queries,
            'perturbed_responses': self.perturbed_responses,
            'unique_query_patterns': len(self.query_fingerprints),
            'block_rate': self.blocked_queries / max(self.total_queries, 1),
            'perturbation_rate': self.perturbed_responses / max(self.total_queries, 1)
        }


class DefendedBlackBoxAPI:
    """
    Protected API wrapper with defense mechanism.
    Drop-in replacement for the original BlackBoxAPI.
    """
    
    def __init__(self, victim: nn.Module, defense_config: Optional[Dict] = None):
        self.victim = victim
        self.device = next(victim.parameters()).device
        self.defense = DefenseMechanism(victim, self.device, defense_config)
        self.api_calls = 0
        
    def fit_defense(self, train_loader):
        """Fit the defense mechanism with training data"""
        self.defense.fit_ood_detector(train_loader)
        
    @torch.no_grad()
    def query(self, x: torch.Tensor, *, logits=False) -> torch.Tensor:
        """
        Protected query interface.
        
        Args:
            x: Input queries
            logits: Whether to return log probabilities
            
        Returns:
            outputs: Protected outputs (potentially perturbed)
        """
        self.api_calls += 1
        
        # Apply defense mechanism
        defended_outputs, metrics = self.defense.defend(x, return_metrics=True)
        
        # Log suspicious activity
        if metrics and metrics['suspicion_score'] > 0.5:
            logger.info(f"API Call {self.api_calls}: Suspicion={metrics['suspicion_score']:.3f}, "
                       f"OOD={metrics['ood_ratio']:.3f}, Pattern={metrics['pattern_score']:.3f}")
        
        # Return appropriate format
        if logits:
            return defended_outputs
        else:
            return torch.exp(defended_outputs)  # Convert log-probs to probs
    
    def get_defense_report(self) -> Dict:
        """Get comprehensive defense report"""
        report = self.defense.get_defense_summary()
        report['api_calls'] = self.api_calls
        return report


def evaluate_defense(victim_model: nn.Module, 
                    defended_api: DefendedBlackBoxAPI,
                    test_loader,
                    n_queries: int = 8000) -> Dict:
    """
    Evaluate defense effectiveness against extraction attack.
    
    Returns metrics comparing defended vs undefended scenarios.
    """
    from extraction_attack import SurrogateNet, build_query_set, train_surrogate, accuracy, agreement
    
    device = next(victim_model.parameters()).device
    
    # 1. Undefended baseline
    class UndefendedAPI:
        def __init__(self, model):
            self.model = model
        @torch.no_grad()
        def query(self, x, *, logits=False):
            output = self.model(x.to(device))
            return output if logits else torch.exp(output)
    
    undefended_api = UndefendedAPI(victim_model)
    
    # Build query sets
    logger.info("Building query sets...")
    undefended_qset = build_query_set(undefended_api, n_queries)
    defended_qset = build_query_set(defended_api, n_queries)
    
    # Train surrogates
    logger.info("Training surrogate models...")
    undefended_surrogate = train_surrogate(undefended_qset)
    defended_surrogate = train_surrogate(defended_qset)
    
    # Evaluate
    results = {
        'undefended': {
            'surrogate_accuracy': accuracy(undefended_surrogate, test_loader),
            'agreement': agreement(victim_model, undefended_surrogate, test_loader)
        },
        'defended': {
            'surrogate_accuracy': accuracy(defended_surrogate, test_loader),
            'agreement': agreement(victim_model, defended_surrogate, test_loader)
        },
        'defense_metrics': defended_api.get_defense_report()
    }
    
    # Calculate defense effectiveness
    acc_reduction = results['undefended']['surrogate_accuracy'] - results['defended']['surrogate_accuracy']
    agr_reduction = results['undefended']['agreement'] - results['defended']['agreement']
    
    results['effectiveness'] = {
        'accuracy_reduction': acc_reduction,
        'agreement_reduction': agr_reduction,
        'relative_acc_reduction': acc_reduction / results['undefended']['surrogate_accuracy'],
        'relative_agr_reduction': agr_reduction / results['undefended']['agreement']
    }
    
    return results


if __name__ == "__main__":
    # Example usage
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", action="store_true", help="Run defense evaluation")
    parser.add_argument("--config", type=str, help="Path to defense config JSON")
    args = parser.parse_args()
    
    if args.evaluate:
        # Load necessary components
        from extraction_attack import CFG, get_loaders, load_victim
        
        # Setup
        device = torch.device(CFG["device"])
        victim = load_victim()
        train_loader, test_loader = get_loaders(CFG["batch_size"])
        
        # Configure defense
        defense_config = None
        if args.config and Path(args.config).exists():
            import json
            with open(args.config) as f:
                defense_config = json.load(f)
        
        # Create defended API
        defended_api = DefendedBlackBoxAPI(victim, defense_config)
        defended_api.fit_defense(train_loader)
        
        # Evaluate
        logger.info("Starting defense evaluation...")
        results = evaluate_defense(victim, defended_api, test_loader)
        
        # Print results
        print("\n" + "="*60)
        print("DEFENSE EVALUATION RESULTS")
        print("="*60)
        
        print("\nUndefended Performance:")
        print(f"  Surrogate Accuracy: {results['undefended']['surrogate_accuracy']*100:.2f}%")
        print(f"  Agreement Rate: {results['undefended']['agreement']*100:.2f}%")
        
        print("\nDefended Performance:")
        print(f"  Surrogate Accuracy: {results['defended']['surrogate_accuracy']*100:.2f}%")
        print(f"  Agreement Rate: {results['defended']['agreement']*100:.2f}%")
        
        print("\nDefense Effectiveness:")
        print(f"  Accuracy Reduction: {results['effectiveness']['accuracy_reduction']*100:.2f}%")
        print(f"  Agreement Reduction: {results['effectiveness']['agreement_reduction']*100:.2f}%")
        print(f"  Relative Impact: {results['effectiveness']['relative_acc_reduction']*100:.2f}%")
        
        print("\nDefense Statistics:")
        for k, v in results['defense_metrics'].items():
            print(f"  {k}: {v}")
        
        # Save results
        import json
        with open("defense_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print("\nResults saved to defense_results.json")