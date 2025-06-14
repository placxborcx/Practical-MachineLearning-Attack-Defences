
"""
Fixed Comprehensive Defense Testing Script
==========================================
Debug and fix the defense mechanism testing
"""

import json
import time
import torch
import numpy as np
from typing import Dict, List, Tuple

# Import necessary components
from extraction_attack import (
    CFG, get_loaders, load_victim, BlackBoxAPI,
    build_query_set, train_surrogate, accuracy, agreement,
    SurrogateNet, DEVICE
)

# Import the fixed defense mechanism
try:
    from defense_mechanism_fixed import DefendedBlackBoxAPI, DefenseMechanism
except:
    from defense_mechanism import DefendedBlackBoxAPI, DefenseMechanism

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


IMG_SHAPE = (1, 28, 28)  # (C, H, W) for MNIST
C, H, W = IMG_SHAPE


def debug_defense_config_application(defense_config, config_name):
    """Debug if defense configuration is properly applied"""
    print(f"\n--- Debug Config Application: {config_name} ---")
    
    victim = load_victim()
    defended_api = DefendedBlackBoxAPI(victim, defense_config)
    
    # Check if config was applied
    actual_config = defended_api.defense.config
    print("Expected vs Actual Config:")
    for key, expected_value in defense_config.items():
        actual_value = actual_config.get(key, "NOT_FOUND")
        match = "✓" if actual_value == expected_value else "✗"
        print(f"  {key}: {expected_value} -> {actual_value} {match}")
    
    return defended_api


def test_defense_activation_directly():
    """Direct test of defense activation with manual queries"""
    print("\n" + "="*60)
    print("DIRECT DEFENSE ACTIVATION TEST")
    print("="*60)
    
    victim = load_victim()
    train_loader, test_loader = get_loaders(CFG["batch_size"])
    
    # Test with very aggressive settings to ensure activation
    aggressive_config = {
        'baseline_suspicion': 0.5,     # Very high baseline
        'perturb_threshold': 0.1,      # Very low threshold
        'block_threshold': 0.6,        # Lower block threshold
        'ood_threshold': 0.5,          # Lower OOD threshold
        'top_k': 2,                    # More restrictive
        'base_noise_scale': 0.1,       # Higher noise
        'max_noise_scale': 0.5,        # Much higher max noise
    }
    
    defended_api = DefendedBlackBoxAPI(victim, aggressive_config)
    defended_api.fit_defense(train_loader)
    
    print("\nTesting with aggressive config:")
    print(f"Config: {aggressive_config}")
    
    # Test with same data multiple times to trigger pattern detection
    test_data = next(iter(test_loader))[0][:5].to(DEVICE)
    
    print("\nQuerying same data multiple times to trigger defenses:")
    for i in range(10):
        output = defended_api.query(test_data, logits=True)
        stats = defended_api.get_defense_report()
        
        print(f"Query {i+1}: Avg Suspicion={stats.get('average_suspicion', 0):.3f}, "
              f"Perturb Rate={stats.get('perturb_rate', 0)*100:.1f}%, "
              f"Block Rate={stats.get('block_rate', 0)*100:.1f}%")
        
        # Check if output is being modified
        with torch.no_grad():
            original_output = victim(test_data)
            original_logprobs = torch.log_softmax(original_output, dim=1)
            diff = torch.abs(output - original_logprobs).mean().item()
            print(f"         Output difference from original: {diff:.6f}")


def test_single_defense_configuration_fixed(
    victim_model, 
    train_loader, 
    test_loader,
    defense_config, 
    config_name,
    n_queries=3000  # Reduced for debugging
):
    """Fixed version with better debugging"""
    
    print(f"\n{'='*60}")
    print(f"Testing: {config_name}")
    print('='*60)
    
    # Debug: Check if config is applied correctly
    defended_api = debug_defense_config_application(defense_config, config_name)
    defended_api.fit_defense(train_loader)
    
    # 1. Baseline (no defense) - Use smaller query set for debugging
    print("\n1. Baseline Attack (No Defense):")
    undefended_api = BlackBoxAPI(victim_model)
    
    t_start = time.perf_counter()
    qset_undefended = build_query_set(undefended_api, n_queries)
    surrogate_undefended = train_surrogate(qset_undefended)
    baseline_time = time.perf_counter() - t_start
    
    baseline_acc = accuracy(surrogate_undefended, test_loader)
    baseline_agr = agreement(victim_model, surrogate_undefended, test_loader)
    
    print(f"  Surrogate Accuracy: {baseline_acc*100:.2f}%")
    print(f"  Agreement Rate: {baseline_agr*100:.2f}%")
    print(f"  Attack Time: {baseline_time:.2f}s")
    
    # 2. With Defense - Add debugging
    print(f"\n2. Attack Against {config_name} Defense:")
    
    # Reset the defended API to ensure clean state
    defended_api = DefendedBlackBoxAPI(victim_model, defense_config)
    defended_api.fit_defense(train_loader)
    
    t_start = time.perf_counter()
    
    # Build query set with monitoring
    print("   Building query set with defense monitoring...")
    qset_defended = build_query_set_with_monitoring(defended_api, n_queries)
    
    surrogate_defended = train_surrogate(qset_defended)
    defended_time = time.perf_counter() - t_start
    
    defended_acc = accuracy(surrogate_defended, test_loader)
    defended_agr = agreement(victim_model, surrogate_defended, test_loader)
    
    print(f"  Surrogate Accuracy: {defended_acc*100:.2f}%")
    print(f"  Agreement Rate: {defended_agr*100:.2f}%")
    print(f"  Attack Time: {defended_time:.2f}s")
    
    # 3. Detailed Defense Report
    defense_report = defended_api.get_defense_report()
    print(f"\n3. Defense Statistics:")
    for key, value in defense_report.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # 4. Compare outputs directly
    print(f"\n4. Output Comparison Test:")
    test_batch = next(iter(test_loader))[0][:5].to(DEVICE)
    
    with torch.no_grad():
        original_output = victim_model(test_batch)
        defended_output = defended_api.query(test_batch, logits=True)
        
        original_probs = torch.softmax(original_output, dim=1)
        defended_probs = torch.exp(defended_output)  # defended_output is log_softmax
        
        kl_div = torch.nn.functional.kl_div(defended_output, original_probs, 
                                          reduction='batchmean').item()
        l1_diff = torch.abs(original_probs - defended_probs).mean().item()
        
        print(f"  KL Divergence: {kl_div:.6f}")
        print(f"  L1 Difference: {l1_diff:.6f}")
        print(f"  Max Prob Diff: {torch.abs(original_probs - defended_probs).max().item():.6f}")
    
    # 5. Impact on Normal Usage
    print(f"\n5. Normal Usage Impact:")
    normal_acc = test_normal_usage_accuracy(victim_model, defended_api, test_loader, n_samples=500)
    victim_acc = accuracy(victim_model, test_loader)
    print(f"  Victim Model Accuracy: {victim_acc*100:.2f}%")
    print(f"  Normal Usage Accuracy: {normal_acc*100:.2f}%")
    print(f"  Accuracy Drop: {(victim_acc - normal_acc)*100:.2f}%")
    
    # 6. Defense Effectiveness Analysis
    print(f"\n6. Defense Effectiveness:")
    if baseline_acc > 0:
        acc_reduction = (baseline_acc - defended_acc) / baseline_acc * 100
    else:
        acc_reduction = 0
    
    if baseline_agr > 0:
        agr_reduction = (baseline_agr - defended_agr) / baseline_agr * 100
    else:
        agr_reduction = 0
    
    print(f"  Accuracy Reduction: {acc_reduction:.2f}%")
    print(f"  Agreement Reduction: {agr_reduction:.2f}%")
    
    effectiveness_score = (acc_reduction + agr_reduction) / 2
    print(f"  Overall Effectiveness Score: {effectiveness_score:.2f}%")
    
    # Check if there's actually any difference
    if abs(baseline_acc - defended_acc) < 0.001 and abs(baseline_agr - defended_agr) < 0.001:
        print("  ⚠️  WARNING: No significant difference detected between baseline and defended!")
        print("     This suggests the defense is not activating properly.")
    
    return {
        'baseline': {
            'accuracy': float(baseline_acc),
            'agreement': float(baseline_agr),
            'time': baseline_time
        },
        'defended': {
            'accuracy': float(defended_acc),
            'agreement': float(defended_agr),
            'time': defended_time
        },
        'defense_stats': defense_report,
        'normal_usage': {
            'accuracy': float(normal_acc),
            'drop': float(victim_acc - normal_acc)
        },
        'effectiveness': {
            'acc_reduction': float(acc_reduction),
            'agr_reduction': float(agr_reduction),
            'score': float(effectiveness_score)
        },
        'output_analysis': {
            'kl_divergence': float(kl_div),
            'l1_difference': float(l1_diff)
        }
    }


def build_query_set_with_monitoring(api, n_queries):
    """Build query set while monitoring defense activation"""
    print(f"   Building {n_queries} queries...")
    
    queries = []
    labels = []
    
    # Track defense stats periodically
    check_intervals = [100, 500, 1000, 2000, n_queries]
    next_check = 0
    
    # Use mixed query strategy to trigger defenses
    batch_size = 32
    n_batches = n_queries // batch_size
    
    for batch_idx in range(n_batches):
        # Create batch of queries
        if batch_idx % 3 == 0:
            # Random queries (30% of batches)
            x = torch.randn(batch_size, C, H, W, device=DEVICE)
        elif batch_idx % 3 == 1:
            # Repeated queries to trigger pattern detection (30% of batches)
            base_query = torch.randn(1, C, H, W, device=DEVICE)
            x = base_query.repeat(batch_size, 1, 1, 1)
            # Add small variations
            x += torch.randn_like(x) * 0.1
        else:
            # Semi-structured queries (40% of batches)
            x = torch.randn(batch_size, C, H, W, device=DEVICE)
            # Make some channels similar to trigger OOD detection
            x += torch.randn_like(x) * 0.05
        
        # Query the API
        y_pred = api.query(x, logits=True)
        
        queries.append(x.cpu())
        labels.append(y_pred.argmax(dim=1).cpu())
        
        # Check defense stats at intervals
        current_queries = (batch_idx + 1) * batch_size
        if next_check < len(check_intervals) and current_queries >= check_intervals[next_check]:
            stats = api.get_defense_report()
            print(f"     Query {current_queries}: Suspicion={stats.get('average_suspicion', 0):.3f}, "
                  f"Perturb={stats.get('perturb_rate', 0)*100:.1f}%, "
                  f"Block={stats.get('block_rate', 0)*100:.1f}%")
            
            next_check += 1
    
    # Handle remaining queries
    remaining = n_queries - n_batches * batch_size
    if remaining > 0:
        x = torch.randn(remaining, C, H, W, device=DEVICE)
        y_pred = api.query(x, logits=True)
        queries.append(x.cpu())
        labels.append(y_pred.argmax(dim=1).cpu())
    
    return torch.cat(queries), torch.cat(labels)


def test_normal_usage_accuracy(victim_model, defended_api, test_loader, n_samples=1000):
    """Test accuracy on normal (non-attack) queries"""
    correct = 0
    total = 0
    
    victim_model.eval()
    with torch.no_grad():
        for data, labels in test_loader:
            if total >= n_samples:
                break
                
            batch_size = min(len(data), n_samples - total)
            data = data[:batch_size].to(DEVICE)
            labels = labels[:batch_size]
            
            # Get defended predictions
            defended_output = defended_api.query(data, logits=True)
            defended_pred = defended_output.argmax(dim=1).cpu()
            
            correct += (defended_pred == labels).sum().item()
            total += batch_size
    
    return correct / total


def run_comprehensive_defense_test_fixed():
    """Fixed comprehensive defense testing"""
    
    print("\n" + "="*80)
    print("FIXED COMPREHENSIVE DEFENSE TESTING")
    print("="*80)
    
    # First run direct activation test
    test_defense_activation_directly()
    
    # Setup
    victim = load_victim()
    train_loader, test_loader = get_loaders(CFG["batch_size"])
    
    # More distinct test configurations based on direct test results
    # We know the defense works, so let's test more realistic scenarios
    test_configs = {
        'no_defense': {
            # Essentially disable all defenses
            "baseline_suspicion": 0.0,
            "block_threshold": 1.0,
            "perturb_threshold": 1.0,
            "top_k": 100,  # No controlled release
            "base_noise_scale": 0.0,
        },
        'light_defense': {
            # Light defense that should trigger occasionally
            "baseline_suspicion": 0.15,
            "block_threshold": 0.85,
            "perturb_threshold": 0.70,
            "top_k": 5,
            "base_noise_scale": 0.01,
            "ood_threshold": 0.80,
        },
        'moderate_defense': {
            # Moderate defense - should trigger more often
            "baseline_suspicion": 0.25,
            "block_threshold": 0.70,
            "perturb_threshold": 0.50,
            "top_k": 3,
            "base_noise_scale": 0.03,
            "max_noise_scale": 0.20,
            "ood_threshold": 0.70,
            "temperature_base": 1.8,
        },
        'aggressive_defense': {
            # More aggressive - should trigger frequently
            "baseline_suspicion": 0.35,
            "block_threshold": 0.60,
            "perturb_threshold": 0.30,
            "top_k": 2,
            "base_noise_scale": 0.05,
            "max_noise_scale": 0.30,
            "ood_threshold": 0.60,
            "temperature_base": 2.5,
        }
    }
    
    # Store all results
    all_results = {}
    
    # Test each configuration
    for config_name, config in test_configs.items():
        try:
            results = test_single_defense_configuration_fixed(
                victim, 
                train_loader, 
                test_loader,
                config, 
                config_name,
                n_queries=2000  # Reduced for debugging
            )
            all_results[config_name] = results
        except Exception as e:
            print(f"Error testing {config_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*80)
    print("FIXED DEFENSE TESTING SUMMARY")
    print("="*80)
    
    print("\n{:<18} | {:>8} | {:>8} | {:>8} | {:>8} | {:>10} | {:>8}".format(
        "Configuration", "Surr.Acc", "Agreemnt", "Norm.Acc", "Effectiv", "KL-Div", "L1-Diff"
    ))
    print("-" * 90)
    
    for config_name, results in all_results.items():
        surr_acc = results['defended']['accuracy'] * 100
        agreement = results['defended']['agreement'] * 100
        normal_acc = results['normal_usage']['accuracy'] * 100
        effectiveness = results['effectiveness']['score']
        kl_div = results['output_analysis']['kl_divergence']
        l1_diff = results['output_analysis']['l1_difference']
        
        print("{:<18} | {:>8.2f}% | {:>8.2f}% | {:>8.2f}% | {:>8.2f}% | {:>10.6f} | {:>8.6f}".format(
            config_name, surr_acc, agreement, normal_acc, effectiveness, kl_div, l1_diff
        ))
    
    # Save detailed results
    output = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'victim_accuracy': float(accuracy(victim, test_loader)),
        'configurations': test_configs,
        'results': all_results
    }
    
    with open('defense_test_results_fixed.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nDetailed results saved to defense_test_results_fixed.json")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", 
                       help="Run debug test to check defense activation")
    parser.add_argument("--direct", action="store_true",
                       help="Run direct activation test only")
    
    args = parser.parse_args()
    
    if args.debug:
        test_defense_activation_directly()
    elif args.direct:
        test_defense_activation_directly()
    else:
        run_comprehensive_defense_test_fixed()