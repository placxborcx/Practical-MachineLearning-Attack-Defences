# test_defense.py
"""
Test Defense Mechanism Against Model Extraction Attack
=====================================================
This script integrates the defense mechanism with your extraction attack
to evaluate its effectiveness.
"""

import json
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict

# Import your attack components
from extraction_attack import (
    CFG, get_loaders, load_victim, BlackBoxAPI,
    build_query_set, train_surrogate, accuracy, agreement,
    SurrogateNet
)

# Import defense mechanism
from defense_mechanism import DefendedBlackBoxAPI, DefenseMechanism

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_extraction_attack(api, n_queries: int, label: str = ""):
    """Run extraction attack and return results"""
    print(f"\n[{label}] Running extraction attack with {n_queries} queries...")
    
    t_start = time.perf_counter()
    
    # Build query set
    qset = build_query_set(api, n_queries)
    query_time = time.perf_counter() - t_start
    
    # Train surrogate
    t_train = time.perf_counter()
    surrogate = train_surrogate(qset)
    train_time = time.perf_counter() - t_train
    
    total_time = time.perf_counter() - t_start
    
    return {
        'surrogate': surrogate,
        'query_time': query_time,
        'train_time': train_time,
        'total_time': total_time
    }


def comprehensive_defense_evaluation():
    """Run comprehensive evaluation of defense mechanism"""
    
    # Setup
    device = torch.device(CFG["device"])
    victim = load_victim()
    train_loader, test_loader = get_loaders(CFG["batch_size"])
    
    # Test different defense configurations
    defense_configs = {
        'baseline': None,  # No defense
        'low': {
            'base_noise_scale': 0.005,
            'max_noise_scale': 0.05,
            'block_threshold': 0.9,
            'ood_threshold': 0.9
        },
        'medium': {
            'base_noise_scale': 0.01,
            'max_noise_scale': 0.1,
            'block_threshold': 0.7,
            'ood_threshold': 0.85
        },
        'high': {
            'base_noise_scale': 0.02,
            'max_noise_scale': 0.2,
            'block_threshold': 0.5,
            'ood_threshold': 0.8,
            'deception_probability': 0.4
        },
        'adaptive': {
            'base_noise_scale': 0.01,
            'max_noise_scale': 0.15,
            'block_threshold': 0.6,
            'ood_threshold': 0.85,
            'adaptive_factor': 3.0,
            'pattern_threshold': 0.25
        }
    }
    
    results = {}
    
    # Test each configuration
    for config_name, config in defense_configs.items():
        print(f"\n{'='*60}")
        print(f"Testing defense configuration: {config_name.upper()}")
        print('='*60)
        
        if config_name == 'baseline':
            # Undefended API
            api = BlackBoxAPI(victim)
            defense_report = None
        else:
            # Defended API
            api = DefendedBlackBoxAPI(victim, config)
            api.fit_defense(train_loader)
            
        # Run attack with different query budgets
        query_budgets = [2000, 5000, 8000]
        config_results = {}
        
        for n_queries in query_budgets:
            # Run extraction attack
            attack_result = run_extraction_attack(api, n_queries, f"{config_name}-{n_queries}")
            
            # Evaluate surrogate
            surrogate = attack_result['surrogate']
            surrogate_acc = accuracy(surrogate, test_loader)
            agr_rate = agreement(victim, surrogate, test_loader)
            
            # Get defense metrics if applicable
            defense_metrics = api.get_defense_report() if hasattr(api, 'get_defense_report') else {}
            
            config_results[f'n{n_queries}'] = {
                'surrogate_accuracy': surrogate_acc,
                'agreement_rate': agr_rate,
                'attack_time': attack_result['total_time'],
                'defense_metrics': defense_metrics
            }
            
            print(f"\n  Queries: {n_queries}")
            print(f"  Surrogate Accuracy: {surrogate_acc*100:.2f}%")
            print(f"  Agreement Rate: {agr_rate*100:.2f}%")
            
            if defense_metrics:
                print(f"  Perturbed Responses: {defense_metrics.get('perturbed_responses', 0)}")
                print(f"  Perturbation Rate: {defense_metrics.get('perturbation_rate', 0)*100:.2f}%")
        
        results[config_name] = config_results
    
    # Calculate defense effectiveness
    print("\n" + "="*60)
    print("DEFENSE EFFECTIVENESS SUMMARY")
    print("="*60)
    
    baseline_8k = results['baseline']['n8000']
    
    effectiveness = {}
    for config_name in ['low', 'medium', 'high', 'adaptive']:
        if config_name in results:
            defended_8k = results[config_name]['n8000']
            
            acc_reduction = baseline_8k['surrogate_accuracy'] - defended_8k['surrogate_accuracy']
            agr_reduction = baseline_8k['agreement_rate'] - defended_8k['agreement_rate']
            
            effectiveness[config_name] = {
                'accuracy_reduction': acc_reduction * 100,
                'agreement_reduction': agr_reduction * 100,
                'relative_impact': (agr_reduction / baseline_8k['agreement_rate']) * 100
            }
            
            print(f"\n{config_name.upper()} Defense:")
            print(f"  Accuracy Reduction: {acc_reduction*100:.2f}%")
            print(f"  Agreement Reduction: {agr_reduction*100:.2f}%")
            print(f"  Relative Impact: {effectiveness[config_name]['relative_impact']:.2f}%")
    
    # Save detailed results
    output = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'configurations': defense_configs,
        'results': results,
        'effectiveness': effectiveness,
        'victim_accuracy': accuracy(victim, test_loader)
    }
    
    with open('defense_evaluation_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\nDetailed results saved to defense_evaluation_results.json")
    
    # Visualize defense impact
    print("\n" + "="*60)
    print("DEFENSE IMPACT VISUALIZATION")
    print("="*60)
    
    configs = ['baseline', 'low', 'medium', 'high', 'adaptive']
    
    print("\nSurrogate Accuracy (%) vs Query Budget:")
    print("Config     | 2000q | 5000q | 8000q |")
    print("-"*36)
    
    for config in configs:
        if config in results:
            line = f"{config:10} |"
            for n in [2000, 5000, 8000]:
                acc = results[config][f'n{n}']['surrogate_accuracy'] * 100
                line += f" {acc:5.1f} |"
            print(line)
    
    print("\nAgreement Rate (%) vs Query Budget:")
    print("Config     | 2000q | 5000q | 8000q |")
    print("-"*36)
    
    for config in configs:
        if config in results:
            line = f"{config:10} |"
            for n in [2000, 5000, 8000]:
                agr = results[config][f'n{n}']['agreement_rate'] * 100
                line += f" {agr:5.1f} |"
            print(line)
    
    return results


def test_specific_defense_config(defense_config: Dict = None):
    """Test a specific defense configuration"""
    
    # Setup
    device = torch.device(CFG["device"])
    victim = load_victim()
    train_loader, test_loader = get_loaders(CFG["batch_size"])
    
    # Default config if none provided
    if defense_config is None:
        defense_config = {
            'base_noise_scale': 0.01,
            'max_noise_scale': 0.15,
            'ood_threshold': 0.85,
            'entropy_threshold': 0.7,
            'pattern_threshold': 0.3,
            'block_threshold': 0.7,
            'deception_probability': 0.3
        }
    
    print("\nDefense Configuration:")
    print(json.dumps(defense_config, indent=2))
    
    # Create APIs
    undefended_api = BlackBoxAPI(victim)
    defended_api = DefendedBlackBoxAPI(victim, defense_config)
    defended_api.fit_defense(train_loader)
    
    # Run attacks
    n_queries = 8000
    
    print(f"\nRunning extraction attacks with {n_queries} queries...")
    
    # Undefended
    t0 = time.perf_counter()
    undefended_result = run_extraction_attack(undefended_api, n_queries, "Undefended")
    undefended_time = time.perf_counter() - t0
    
    # Defended
    t0 = time.perf_counter()
    defended_result = run_extraction_attack(defended_api, n_queries, "Defended")
    defended_time = time.perf_counter() - t0
    
    # Evaluate
    undefended_acc = accuracy(undefended_result['surrogate'], test_loader)
    undefended_agr = agreement(victim, undefended_result['surrogate'], test_loader)
    
    defended_acc = accuracy(defended_result['surrogate'], test_loader)
    defended_agr = agreement(victim, defended_result['surrogate'], test_loader)
    
    # Get defense report
    defense_report = defended_api.get_defense_report()
    
    # Print results
    print("\n" + "="*60)
    print("DEFENSE TEST RESULTS")
    print("="*60)
    
    print("\nModel Performance:")
    print(f"  Victim Accuracy: {accuracy(victim, test_loader)*100:.2f}%")
    
    print("\nUndefended Attack:")
    print(f"  Surrogate Accuracy: {undefended_acc*100:.2f}%")
    print(f"  Agreement Rate: {undefended_agr*100:.2f}%")
    print(f"  Attack Time: {undefended_time:.2f}s")
    
    print("\nDefended Attack:")
    print(f"  Surrogate Accuracy: {defended_acc*100:.2f}%")
    print(f"  Agreement Rate: {defended_agr*100:.2f}%")
    print(f"  Attack Time: {defended_time:.2f}s")
    
    print("\nDefense Impact:")
    acc_reduction = (undefended_acc - defended_acc) * 100
    agr_reduction = (undefended_agr - defended_agr) * 100
    print(f"  Accuracy Reduction: {acc_reduction:.2f}%")
    print(f"  Agreement Reduction: {agr_reduction:.2f}%")
    print(f"  Relative Impact: {agr_reduction/(undefended_agr*100)*100:.2f}%")
    
    print("\nDefense Statistics:")
    for k, v in defense_report.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    return {
        'undefended': {
            'accuracy': undefended_acc,
            'agreement': undefended_agr,
            'time': undefended_time
        },
        'defended': {
            'accuracy': defended_acc,
            'agreement': defended_agr,
            'time': defended_time
        },
        'defense_report': defense_report,
        'impact': {
            'accuracy_reduction': acc_reduction,
            'agreement_reduction': agr_reduction
        }
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--comprehensive", action="store_true", 
                       help="Run comprehensive evaluation with multiple configs")
    parser.add_argument("--config", type=str, 
                       help="Path to defense config JSON file")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test with default config")
    
    args = parser.parse_args()
    
    if args.comprehensive:
        # Run comprehensive evaluation
        results = comprehensive_defense_evaluation()
        
    elif args.config:
        # Load and test specific config
        with open(args.config) as f:
            config = json.load(f)
        results = test_specific_defense_config(config)
        
    else:
        # Default quick test
        print("Running quick defense test with default configuration...")
        results = test_specific_defense_config()
        
        # Save quick test results
        with open('quick_defense_test.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("\nResults saved to quick_defense_test.json")