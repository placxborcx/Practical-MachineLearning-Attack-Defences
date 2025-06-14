# enhanced_defense_evaluation.py
"""
Enhanced Defense Evaluation with Normal Usage Impact Assessment
==============================================================
Evaluates both attack prevention effectiveness and impact on legitimate users
"""

import json
import time
import torch
import numpy as np
from typing import Dict, List, Tuple

# Import from extraction_attack - make sure all needed functions are imported
from extraction_attack import (
    CFG, get_loaders, load_victim, BlackBoxAPI,
    build_query_set, train_surrogate, accuracy, agreement,
    SurrogateNet, DEVICE
)

from defense_mechanism import DefendedBlackBoxAPI, DefenseMechanism

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_normal_usage_impact(victim_model, defended_api, test_loader, num_samples=1000):
    """
    Evaluate the impact of defense mechanism on normal/legitimate queries
    
    Args:
        victim_model: Original undefended model
        defended_api: Defended API
        test_loader: Test data loader
        num_samples: Number of samples to test
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("\n[Normal Usage Evaluation] Testing impact on legitimate queries...")
    
    device = next(victim_model.parameters()).device
    
    # Collect predictions from both models
    undefended_predictions = []
    defended_predictions = []
    true_labels = []
    
    sample_count = 0
    victim_model.eval()
    
    # Create undefended API for fair comparison
    undefended_api = BlackBoxAPI(victim_model)
    
    with torch.no_grad():
        for data, labels in test_loader:
            if sample_count >= num_samples:
                break
                
            batch_size = min(len(data), num_samples - sample_count)
            data = data[:batch_size].to(device)
            labels = labels[:batch_size]
            
            # Undefended predictions (using BlackBoxAPI for consistency)
            undefended_output = undefended_api.query(data, logits=True)
            undefended_pred = undefended_output.argmax(dim=1).cpu()
            
            # Defended predictions (through defended API)
            defended_output = defended_api.query(data, logits=True)
            defended_pred = defended_output.argmax(dim=1).cpu()
            
            undefended_predictions.extend(undefended_pred.numpy())
            defended_predictions.extend(defended_pred.numpy())
            true_labels.extend(labels.numpy())
            
            sample_count += batch_size
    
    # Calculate metrics
    undefended_predictions = np.array(undefended_predictions)
    defended_predictions = np.array(defended_predictions)
    true_labels = np.array(true_labels)
    
    # Accuracy
    undefended_accuracy = (undefended_predictions == true_labels).mean()
    defended_accuracy = (defended_predictions == true_labels).mean()
    
    # Agreement between defended and undefended
    prediction_agreement = (undefended_predictions == defended_predictions).mean()
    
    # Accuracy drop
    accuracy_drop = undefended_accuracy - defended_accuracy
    
    # Debug output
    print(f"  Debug - Undefended correct: {(undefended_predictions == true_labels).sum()}/{len(true_labels)}")
    print(f"  Debug - Defended correct: {(defended_predictions == true_labels).sum()}/{len(true_labels)}")
    print(f"  Debug - Predictions differ: {(undefended_predictions != defended_predictions).sum()}")
    
    # Per-class analysis
    class_impacts = {}
    for class_id in range(10):
        class_mask = true_labels == class_id
        if class_mask.sum() > 0:
            undefended_class_acc = (undefended_predictions[class_mask] == true_labels[class_mask]).mean()
            defended_class_acc = (defended_predictions[class_mask] == true_labels[class_mask]).mean()
            class_impacts[class_id] = {
                'undefended_acc': float(undefended_class_acc),
                'defended_acc': float(defended_class_acc),
                'accuracy_drop': float(undefended_class_acc - defended_class_acc)
            }
    
    return {
        'undefended_accuracy': float(undefended_accuracy),
        'defended_accuracy': float(defended_accuracy),
        'accuracy_drop': float(accuracy_drop),
        'prediction_agreement': float(prediction_agreement),
        'class_impacts': class_impacts,
        'samples_tested': sample_count
    }


def comprehensive_defense_evaluation_with_normal_usage():
    """Enhanced evaluation including normal usage impact"""
    
    # Setup
    device = torch.device(CFG["device"])
    victim = load_victim()
    train_loader, test_loader = get_loaders(CFG["batch_size"])
    
    # Defense configurations
    defense_configs = {
        'baseline': None,
        'low': {
            'base_noise_scale': 0.005,
            'max_noise_scale': 0.05,
            'block_threshold': 0.9,
            'ood_threshold': 0.9,
            'perturb_threshold': 0.4,
            'top_k': 10  # Allow all logits for low defense
        },
        'medium': {
            'base_noise_scale': 0.01,
            'max_noise_scale': 0.1,
            'block_threshold': 0.7,
            'ood_threshold': 0.85,
            'perturb_threshold': 0.3,
            'top_k': 5  # Restrict to top-5 logits
        },
        'high': {
            'base_noise_scale': 0.02,
            'max_noise_scale': 0.2,
            'block_threshold': 0.5,
            'ood_threshold': 0.8,
            'deception_probability': 0.4,
            'perturb_threshold': 0.2,
            'top_k': 1  # Only top-1 logit (maximum protection)
        },
        'adaptive': {
            'base_noise_scale': 0.01,
            'max_noise_scale': 0.15,
            'block_threshold': 0.6,
            'ood_threshold': 0.85,
            'perturb_threshold': 0.25,
            'top_k': 3  # Top-3 logits
        }
    }
    
    results = {}
    
    # Get baseline victim accuracy first
    print("\n[Baseline Evaluation]")
    victim_accuracy = accuracy(victim, test_loader)
    print(f"Original Victim Model Accuracy: {victim_accuracy*100:.2f}%")
    
    # Test each configuration
    for config_name, config in defense_configs.items():
        print(f"\n{'='*60}")
        print(f"Testing defense configuration: {config_name.upper()}")
        print('='*60)
        
        if config_name == 'baseline':
            # For baseline, just run undefended attack
            api = BlackBoxAPI(victim)
            
            print(f"\nAttack Test (No Defense):")
            n_queries = 5000
            
            # Run extraction attack
            t_start = time.perf_counter()
            qset = build_query_set(api, n_queries)
            surrogate = train_surrogate(qset)
            attack_time = time.perf_counter() - t_start
            
            surrogate_acc = accuracy(surrogate, test_loader)
            agr_rate = agreement(victim, surrogate, test_loader)
            
            print(f"  Surrogate Accuracy: {surrogate_acc*100:.2f}%")
            print(f"  Agreement Rate: {agr_rate*100:.2f}%")
            
            results[config_name] = {
                'normal_usage': {
                    'undefended_accuracy': float(victim_accuracy),
                    'defended_accuracy': float(victim_accuracy),
                    'accuracy_drop': 0.0,
                    'prediction_agreement': 1.0
                },
                'attack_prevention': {
                    'surrogate_accuracy': float(surrogate_acc),
                    'agreement_rate': float(agr_rate),
                    'attack_time': attack_time
                }
            }
            continue
            
        # Create defended API
        defended_api = DefendedBlackBoxAPI(victim, config)
        defended_api.fit_defense(train_loader)
        
        # 1. Normal usage impact evaluation
        normal_usage_results = evaluate_normal_usage_impact(
            victim, defended_api, test_loader, num_samples=2000
        )
        
        print(f"\nNormal Usage Impact:")
        print(f"  Original Accuracy: {normal_usage_results['undefended_accuracy']*100:.2f}%")
        print(f"  Defended Accuracy: {normal_usage_results['defended_accuracy']*100:.2f}%")
        print(f"  Accuracy Drop: {normal_usage_results['accuracy_drop']*100:.2f}%")
        print(f"  Prediction Agreement: {normal_usage_results['prediction_agreement']*100:.2f}%")
        
        # 2. Attack prevention evaluation (simplified)
        print(f"\nAttack Prevention Test:")
        n_queries = 5000
        
        # Run extraction attack
        t_start = time.perf_counter()
        qset = build_query_set(defended_api, n_queries)
        surrogate = train_surrogate(qset)
        attack_time = time.perf_counter() - t_start
        
        surrogate_acc = accuracy(surrogate, test_loader)
        agr_rate = agreement(victim, surrogate, test_loader)
        
        print(f"  Surrogate Accuracy: {surrogate_acc*100:.2f}%")
        print(f"  Agreement Rate: {agr_rate*100:.2f}%")
        
        # Store results
        results[config_name] = {
            'normal_usage': normal_usage_results,
            'attack_prevention': {
                'surrogate_accuracy': float(surrogate_acc),
                'agreement_rate': float(agr_rate),
                'attack_time': attack_time
            },
            'defense_report': defended_api.get_defense_report()
        }
    
    # Summary
    print("\n" + "="*60)
    print("DEFENSE EVALUATION SUMMARY")
    print("="*60)
    
    baseline_results = results.get('baseline', {})
    if baseline_results:
        baseline_acc = baseline_results['attack_prevention']['surrogate_accuracy']
        baseline_agr = baseline_results['attack_prevention']['agreement_rate']
    else:
        baseline_acc = victim_accuracy
        baseline_agr = 1.0
    
    print("\nDefense Trade-off Analysis:")
    print("Config    | Normal Acc Drop | Surrogate Acc | Agreement | Trade-off Score")
    print("-" * 75)
    
    for config_name in ['low', 'medium', 'high', 'adaptive']:
        if config_name in results:
            normal_drop = results[config_name]['normal_usage']['accuracy_drop'] * 100
            surrogate_acc = results[config_name]['attack_prevention']['surrogate_accuracy'] * 100
            agreement_rate = results[config_name]['attack_prevention']['agreement_rate'] * 100
            
            # Trade-off score: lower surrogate accuracy is better, lower normal drop is better
            # Score = (100 - surrogate_acc) - (2 * normal_drop)  # Penalize normal usage impact more
            trade_off_score = (100 - surrogate_acc) - (2 * normal_drop)
            
            print(f"{config_name:9} | {normal_drop:15.2f}% | {surrogate_acc:13.2f}% | {agreement_rate:9.2f}% | {trade_off_score:15.2f}")
    
    # Save results
    output = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'victim_accuracy': float(victim_accuracy),
        'configurations': defense_configs,
        'results': results
    }
    
    with open('enhanced_defense_evaluation.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\nDetailed results saved to enhanced_defense_evaluation.json")
    
    return results


def test_defense_transparency():
    """
    Test how transparent the defense is to normal users
    Runs multiple legitimate queries and checks consistency
    """
    print("\n" + "="*60)
    print("DEFENSE TRANSPARENCY TEST")
    print("="*60)
    
    # Setup
    device = torch.device(CFG["device"])
    victim = load_victim()
    train_loader, test_loader = get_loaders(CFG["batch_size"])
    
    # Test configuration
    test_config = {
        'base_noise_scale': 0.01,
        'max_noise_scale': 0.1,
        'block_threshold': 0.7,
        'ood_threshold': 0.85
    }
    
    # Create defended API
    defended_api = DefendedBlackBoxAPI(victim, test_config)
    defended_api.fit_defense(train_loader)
    
    print("\nTesting query consistency for legitimate users...")
    
    # Test same query multiple times
    consistency_results = []
    
    for data, labels in test_loader:
        if len(consistency_results) >= 10:  # Test 10 batches
            break
            
        data = data[:10].to(device)  # Use first 10 samples
        labels = labels[:10]
        
        # Query same data 5 times
        predictions_list = []
        for i in range(5):
            output = defended_api.query(data, logits=True)
            pred = output.argmax(dim=1).cpu()
            predictions_list.append(pred)
        
        # Check consistency
        first_pred = predictions_list[0]
        consistency = all((pred == first_pred).all() for pred in predictions_list[1:])
        consistency_results.append(consistency)
        
        if not consistency:
            print(f"  Inconsistency detected in batch {len(consistency_results)}")
    
    consistency_rate = sum(consistency_results) / len(consistency_results)
    print(f"\nQuery Consistency Rate: {consistency_rate*100:.2f}%")
    print("(100% means defense is deterministic for legitimate queries)")
    
    return consistency_rate


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", 
                       help="Run full evaluation with normal usage impact")
    parser.add_argument("--transparency", action="store_true",
                       help="Test defense transparency")
    
    args = parser.parse_args()
    
    if args.transparency:
        test_defense_transparency()
    elif args.full:
        comprehensive_defense_evaluation_with_normal_usage()
    else:
        # Run enhanced evaluation by default
        print("Running enhanced defense evaluation...")
        comprehensive_defense_evaluation_with_normal_usage()