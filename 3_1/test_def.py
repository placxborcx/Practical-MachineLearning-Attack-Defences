# test_defense_effectiveness.py
"""
Comprehensive Defense Testing Script
===================================
Tests the effectiveness of defense mechanisms against extraction attacks
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


def test_single_defense_configuration(
    victim_model, 
    train_loader, 
    test_loader,
    defense_config, 
    config_name,
    n_queries=5000
):
    """Test a single defense configuration"""
    
    print(f"\n{'='*60}")
    print(f"Testing: {config_name}")
    print('='*60)
    
    # 1. Baseline (no defense)
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
    
    # 2. With Defense
    print(f"\n2. Attack Against {config_name} Defense:")
    defended_api = DefendedBlackBoxAPI(victim_model, defense_config)
    defended_api.fit_defense(train_loader)
    
    t_start = time.perf_counter()
    qset_defended = build_query_set(defended_api, n_queries)
    surrogate_defended = train_surrogate(qset_defended)
    defended_time = time.perf_counter() - t_start
    
    defended_acc = accuracy(surrogate_defended, test_loader)
    defended_agr = agreement(victim_model, surrogate_defended, test_loader)
    
    print(f"  Surrogate Accuracy: {defended_acc*100:.2f}%")
    print(f"  Agreement Rate: {defended_agr*100:.2f}%")
    print(f"  Attack Time: {defended_time:.2f}s")
    
    # 3. Defense Report
    defense_report = defended_api.get_defense_report()
    print(f"\n3. Defense Statistics:")
    print(f"  Total Queries: {defense_report['total_queries']}")
    print(f"  Blocked Queries: {defense_report['blocked_queries']} ({defense_report.get('block_rate', 0)*100:.1f}%)")
    print(f"  Perturbed Queries: {defense_report['perturbed_responses']} ({defense_report.get('perturb_rate', 0)*100:.1f}%)")
    print(f"  Average Suspicion: {defense_report.get('average_suspicion', 0):.3f}")
    
    # 4. Impact on Normal Usage
    print(f"\n4. Normal Usage Impact:")
    normal_acc = test_normal_usage_accuracy(victim_model, defended_api, test_loader)
    print(f"  Normal Usage Accuracy: {normal_acc*100:.2f}%")
    print(f"  Accuracy Drop: {(accuracy(victim_model, test_loader) - normal_acc)*100:.2f}%")
    
    # 5. Defense Effectiveness
    print(f"\n5. Defense Effectiveness:")
    acc_reduction = (baseline_acc - defended_acc) / baseline_acc * 100
    agr_reduction = (baseline_agr - defended_agr) / baseline_agr * 100
    
    print(f"  Accuracy Reduction: {acc_reduction:.2f}%")
    print(f"  Agreement Reduction: {agr_reduction:.2f}%")
    
    effectiveness_score = (acc_reduction + agr_reduction) / 2
    print(f"  Overall Effectiveness Score: {effectiveness_score:.2f}%")
    
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
            'drop': float(accuracy(victim_model, test_loader) - normal_acc)
        },
        'effectiveness': {
            'acc_reduction': float(acc_reduction),
            'agr_reduction': float(agr_reduction),
            'score': float(effectiveness_score)
        }
    }


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


def run_comprehensive_defense_test():
    """Run comprehensive defense testing"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE DEFENSE TESTING")
    print("="*80)
    
    # Setup
    victim = load_victim()
    train_loader, test_loader = get_loaders(CFG["batch_size"])
    
    # Test configurations
    test_configs = {
        'weak_defense': {
            'baseline_suspicion': 0.05,
            'perturb_threshold': 0.50,
            'top_k': 10
        },
        'moderate_defense': {
            'baseline_suspicion': 0.15,
            'perturb_threshold': 0.30,
            'top_k': 5,
            'base_noise_scale': 0.02,
            'max_noise_scale': 0.20
        },
        'strong_defense': {
            'baseline_suspicion': 0.20,
            'perturb_threshold': 0.25,
            'top_k': 3,
            'base_noise_scale': 0.03,
            'max_noise_scale': 0.30,
            'ood_threshold': 0.65,
            'temperature_base': 2.0
        },
        'aggressive_defense': {
            'baseline_suspicion': 0.25,
            'perturb_threshold': 0.20,
            'top_k': 1,
            'base_noise_scale': 0.05,
            'max_noise_scale': 0.40,
            'ood_threshold': 0.60,
            'block_threshold': 0.60,
            'temperature_base': 2.5
        }
    }
    
    # Store all results
    all_results = {}
    
    # Test each configuration
    for config_name, config in test_configs.items():
        results = test_single_defense_configuration(
            victim, 
            train_loader, 
            test_loader,
            config, 
            config_name,
            n_queries=5000
        )
        all_results[config_name] = results
    
    # Summary
    print("\n" + "="*80)
    print("DEFENSE TESTING SUMMARY")
    print("="*80)
    
    print("\n{:<20} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10}".format(
        "Configuration", "Surr. Acc", "Agreement", "Normal Acc", "Effective", "Trade-off"
    ))
    print("-" * 85)
    
    for config_name, results in all_results.items():
        surr_acc = results['defended']['accuracy'] * 100
        agreement = results['defended']['agreement'] * 100
        normal_acc = results['normal_usage']['accuracy'] * 100
        effectiveness = results['effectiveness']['score']
        
        # Trade-off score: high effectiveness with low normal usage impact
        trade_off = effectiveness - (results['normal_usage']['drop'] * 200)
        
        print("{:<20} | {:>10.2f}% | {:>10.2f}% | {:>10.2f}% | {:>10.2f}% | {:>10.2f}".format(
            config_name, surr_acc, agreement, normal_acc, effectiveness, trade_off
        ))
    
    # Save detailed results
    output = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'victim_accuracy': float(accuracy(victim, test_loader)),
        'configurations': test_configs,
        'results': all_results
    }
    
    with open('defense_test_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\nDetailed results saved to defense_test_results.json")
    
    return all_results


def debug_defense_activation():
    """Debug why defense is not activating"""
    
    print("\n" + "="*60)
    print("DEBUG: Defense Activation Test")
    print("="*60)
    
    # Setup
    victim = load_victim()
    train_loader, test_loader = get_loaders(CFG["batch_size"])
    
    # Create defended API with debug config
    debug_config = {
        'baseline_suspicion': 0.20,
        'perturb_threshold': 0.25,
        'ood_threshold': 0.70,
        'top_k': 3
    }
    
    defended_api = DefendedBlackBoxAPI(victim, debug_config)
    defended_api.fit_defense(train_loader)
    
    # Test with a small batch
    print("\nTesting defense activation...")
    for i, (data, labels) in enumerate(test_loader):
        if i >= 5:  # Test 5 batches
            break
            
        data = data[:10].to(DEVICE)  # Use 10 samples
        
        # Query and check defense stats
        _ = defended_api.query(data, logits=True)
        
        # Get current stats
        stats = defended_api.get_defense_report()
        print(f"\nBatch {i+1}:")
        print(f"  Average Suspicion: {stats.get('average_suspicion', 0):.3f}")
        print(f"  Perturb Rate: {stats.get('perturb_rate', 0)*100:.1f}%")
        print(f"  Block Rate: {stats.get('block_rate', 0)*100:.1f}%")
    
    print("\nFinal Defense Report:")
    final_stats = defended_api.get_defense_report()
    for key, value in final_stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", 
                       help="Run debug test to check defense activation")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick test with fewer queries")
    
    args = parser.parse_args()
    
    if args.debug:
        debug_defense_activation()
    else:
        run_comprehensive_defense_test()