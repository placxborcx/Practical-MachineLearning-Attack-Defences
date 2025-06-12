# -*- coding: utf-8 -*-
"""
Realistic Model Extraction Attack Implementation
Scenario: Extracting a commercial digit recognition API through query access only
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import time
import json
from a3_mnist import My_MNIST


class APISimulator:
    """
    Simulates a commercial API that provides predictions for MNIST digits
    In real scenario, this would be replaced by actual API calls
    """
    def __init__(self, model, device, query_limit=None, add_defense=False):
        self.model = model
        self.device = device
        self.query_count = 0
        self.query_limit = query_limit
        self.add_defense = add_defense
        self.query_history = []
        
    def query(self, data):
        """
        Query the API with input data
        
        Args:
            data: Input images (tensor)
            
        Returns:
            predictions: Model predictions (labels only, no confidence scores)
        """
        # Check query limit
        if self.query_limit and self.query_count >= self.query_limit:
            raise Exception(f"Query limit reached: {self.query_limit}")
        
        self.query_count += len(data)
        self.query_history.append({
            'timestamp': time.time(),
            'batch_size': len(data)
        })
        
        # Move data to device and get predictions
        data = data.to(self.device)
        self.model.eval()
        
        with torch.no_grad():
            output = self.model(data)
            
            # Add defense mechanism (optional)
            if self.add_defense:
                # Add small random noise to outputs before argmax
                noise = torch.randn_like(output) * 0.1
                output = output + noise
            
            predictions = output.argmax(dim=1)
        
        return predictions.cpu()
    
    def get_query_stats(self):
        """Get statistics about API usage"""
        return {
            'total_queries': self.query_count,
            'query_limit': self.query_limit,
            'queries_remaining': self.query_limit - self.query_count if self.query_limit else 'unlimited'
        }


class AttackerModel(nn.Module):
    """
    The attacker's surrogate model
    Using a different architecture than the target to simulate realistic scenario
    """
    def __init__(self):
        super(AttackerModel, self).__init__()
        # Simpler architecture than target model
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class ModelExtractionAttack:
    """
    Complete model extraction attack implementation
    """
    def __init__(self, api, device, budget=10000):
        """
        Initialize the attack
        
        Args:
            api: API simulator (target model interface)
            device: Computing device
            budget: Query budget (number of allowed queries)
        """
        self.api = api
        self.device = device
        self.budget = budget
        self.surrogate_model = AttackerModel().to(device)
        self.query_data = []
        self.query_labels = []
        
    def generate_synthetic_queries(self, num_queries):
        """
        Generate synthetic queries for the attack
        Strategy: Use a mix of real MNIST-like data and synthetic variations
        """
        print(f"\nGenerating {num_queries} synthetic queries...")
        
        # Load a small subset of MNIST for reference
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        reference_data = datasets.MNIST('./data', train=True, transform=transform)
        subset_indices = torch.randperm(len(reference_data))[:1000]
        reference_subset = Subset(reference_data, subset_indices)
        reference_loader = DataLoader(reference_subset, batch_size=100, shuffle=True)
        
        synthetic_data = []
        
        # Strategy 1: Use some real MNIST samples
        for i, (data, _) in enumerate(reference_loader):
            if len(synthetic_data) * 28 * 28 >= num_queries * 0.3:  # 30% real data
                break
            synthetic_data.append(data)
        
        # Strategy 2: Generate augmented versions
        augmentation_transforms = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Strategy 3: Generate random noise patterns
        remaining_queries = num_queries - sum(d.size(0) for d in synthetic_data)
        if remaining_queries > 0:
            noise_data = torch.randn(remaining_queries, 1, 28, 28) * 0.3 + 0.5
            synthetic_data.append(noise_data)
        
        return torch.cat(synthetic_data)[:num_queries]
    
    def execute_queries(self, query_data):
        """
        Execute queries to the target API
        """
        print(f"\nExecuting {len(query_data)} queries to target API...")
        
        # Batch queries to simulate realistic API usage
        batch_size = 100
        predictions = []
        
        for i in range(0, len(query_data), batch_size):
            batch = query_data[i:i+batch_size]
            try:
                batch_predictions = self.api.query(batch)
                predictions.append(batch_predictions)
                
                # Simulate API rate limiting
                time.sleep(0.01)  # Small delay between batches
                
                if (i + batch_size) % 1000 == 0:
                    print(f"  Queried {i + batch_size}/{len(query_data)} samples...")
                    
            except Exception as e:
                print(f"API Error: {e}")
                break
        
        return torch.cat(predictions) if predictions else torch.tensor([])
    
    def train_surrogate(self, epochs=30):
        """
        Train the surrogate model using collected query-response pairs
        """
        print(f"\nTraining surrogate model for {epochs} epochs...")
        
        # Create training dataset
        dataset = TensorDataset(
            torch.cat(self.query_data),
            torch.cat(self.query_labels)
        )
        train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        # Training setup
        optimizer = optim.Adam(self.surrogate_model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        
        # Training loop
        training_history = {'loss': [], 'epoch': []}
        
        for epoch in range(epochs):
            epoch_loss = 0
            self.surrogate_model.train()
            
            for batch_data, batch_labels in train_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                output = self.surrogate_model(batch_data)
                loss = F.nll_loss(output, batch_labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            training_history['loss'].append(avg_loss)
            training_history['epoch'].append(epoch + 1)
            
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
            
            scheduler.step()
        
        return training_history
    
    def run_attack(self):
        """
        Execute the complete model extraction attack
        """
        print("="*60)
        print("Starting Model Extraction Attack")
        print("="*60)
        
        # Phase 1: Query generation
        query_data = self.generate_synthetic_queries(self.budget)
        self.query_data.append(query_data)
        
        # Phase 2: Query execution
        predictions = self.execute_queries(query_data)
        self.query_labels.append(predictions)
        
        print(f"\nCollected {len(predictions)} query-response pairs")
        print(f"API Query Stats: {self.api.get_query_stats()}")
        
        # Phase 3: Surrogate model training
        training_history = self.train_surrogate()
        
        return training_history


def evaluate_extraction_attack(target_model, surrogate_model, test_loader, device):
    """
    Comprehensive evaluation of the model extraction attack
    
    Returns:
        Dictionary containing all evaluation metrics
    """
    print("\n" + "="*60)
    print("Evaluating Model Extraction Attack")
    print("="*60)
    
    results = {}
    
    # 1. Individual Model Performance
    print("\n1. Individual Model Performance:")
    print("-" * 40)
    
    # Evaluate target model
    target_correct = 0
    target_total = 0
    target_predictions = []
    true_labels = []
    
    target_model.eval()
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = target_model(data)
            predictions = outputs.argmax(dim=1)
            
            target_correct += (predictions == labels).sum().item()
            target_total += labels.size(0)
            
            target_predictions.extend(predictions.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    target_accuracy = 100.0 * target_correct / target_total
    results['target_accuracy'] = float(target_accuracy)  # Convert to Python float
    print(f"Target Model Accuracy: {target_accuracy:.2f}%")
    
    # Evaluate surrogate model
    surrogate_correct = 0
    surrogate_total = 0
    surrogate_predictions = []
    
    surrogate_model.eval()
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = surrogate_model(data)
            predictions = outputs.argmax(dim=1)
            
            surrogate_correct += (predictions == labels).sum().item()
            surrogate_total += labels.size(0)
            
            surrogate_predictions.extend(predictions.cpu().numpy())
    
    surrogate_accuracy = 100.0 * surrogate_correct / surrogate_total
    results['surrogate_accuracy'] = float(surrogate_accuracy)  # Convert to Python float
    print(f"Surrogate Model Accuracy: {surrogate_accuracy:.2f}%")
    
    # 2. Attack Fidelity Metrics
    print("\n2. Attack Fidelity Metrics:")
    print("-" * 40)
    
    # Agreement rate
    agreement = np.sum(np.array(target_predictions) == np.array(surrogate_predictions))
    agreement_rate = 100.0 * agreement / len(target_predictions)
    results['agreement_rate'] = float(agreement_rate)  # Convert to Python float
    print(f"Model Agreement Rate: {agreement_rate:.2f}%")
    
    # Accuracy gap
    accuracy_gap = abs(target_accuracy - surrogate_accuracy)
    results['accuracy_gap'] = float(accuracy_gap)  # Convert to Python float
    print(f"Accuracy Gap: {accuracy_gap:.2f}%")
    
    # 3. Per-class Performance
    print("\n3. Per-class Performance Analysis:")
    print("-" * 40)
    
    # Calculate per-class metrics
    from sklearn.metrics import classification_report
    target_report = classification_report(true_labels, target_predictions, 
                                        output_dict=True, zero_division=0)
    surrogate_report = classification_report(true_labels, surrogate_predictions, 
                                           output_dict=True, zero_division=0)
    
    print("\nPer-class Accuracy Comparison:")
    print("Class | Target | Surrogate | Difference")
    print("-" * 40)
    
    class_differences = []
    for i in range(10):
        target_acc = target_report[str(i)]['recall'] * 100
        surrogate_acc = surrogate_report[str(i)]['recall'] * 100
        diff = abs(target_acc - surrogate_acc)
        class_differences.append(float(diff))  # Convert to Python float
        print(f"  {i}   | {target_acc:6.2f}% | {surrogate_acc:9.2f}% | {diff:10.2f}%")
    
    avg_class_diff = float(np.mean(class_differences))  # Convert to Python float
    results['avg_class_difference'] = avg_class_diff
    print(f"\nAverage per-class difference: {avg_class_diff:.2f}%")
    
    # 4. Attack Success Criteria
    print("\n4. Attack Success Evaluation:")
    print("-" * 40)
    
    # Define success criteria
    success_criteria = {
        'High Accuracy': surrogate_accuracy > 90,
        'Low Accuracy Gap': accuracy_gap < 5,
        'High Agreement': agreement_rate > 85,
        'Consistent Per-class': avg_class_diff < 10
    }
    
    success_count = sum(success_criteria.values())
    
    for criterion, passed in success_criteria.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{criterion}: {status}")
    
    overall_success = success_count >= 3  # At least 3 out of 4 criteria
    results['attack_success'] = bool(overall_success)  # Convert numpy.bool_ to Python bool
    results['success_criteria_met'] = int(success_count)  # Convert to Python int
    
    print(f"\nOverall Attack Success: {'YES' if overall_success else 'NO'} ({success_count}/4 criteria met)")
    
    # 5. Visualizations
    print("\n5. Generating Evaluation Visualizations...")
    
    # Confusion matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Target model confusion matrix
    cm_target = confusion_matrix(true_labels, target_predictions)
    sns.heatmap(cm_target, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('Target Model Confusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    
    # Surrogate model confusion matrix
    cm_surrogate = confusion_matrix(true_labels, surrogate_predictions)
    sns.heatmap(cm_surrogate, annot=True, fmt='d', cmap='Reds', ax=ax2)
    ax2.set_title('Surrogate Model Confusion Matrix')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
    print("  Saved: confusion_matrices.png")
    
    # Model agreement visualization
    plt.figure(figsize=(10, 8))
    agreement_matrix = confusion_matrix(target_predictions, surrogate_predictions)
    sns.heatmap(agreement_matrix, annot=True, fmt='d', cmap='Greens')
    plt.title('Model Agreement Matrix\n(Target vs Surrogate Predictions)')
    plt.xlabel('Surrogate Predictions')
    plt.ylabel('Target Predictions')
    plt.savefig('model_agreement.png', dpi=150, bbox_inches='tight')
    print("  Saved: model_agreement.png")
    
    # Performance comparison bar chart
    plt.figure(figsize=(10, 6))
    metrics = ['Overall\nAccuracy', 'Agreement\nRate', 'Avg Class\nDifference']
    target_values = [target_accuracy, 100, 0]
    surrogate_values = [surrogate_accuracy, agreement_rate, avg_class_diff]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, target_values, width, label='Target', color='blue', alpha=0.7)
    plt.bar(x + width/2, surrogate_values, width, label='Surrogate', color='red', alpha=0.7)
    
    plt.xlabel('Metrics')
    plt.ylabel('Percentage (%)')
    plt.title('Model Extraction Attack Performance Comparison')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.savefig('performance_comparison.png', dpi=150, bbox_inches='tight')
    print("  Saved: performance_comparison.png")
    
    plt.close('all')
    
    return results


def main():
    """
    Main execution function
    """
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load target model
    print("\nLoading target model...")
    target_model = My_MNIST().to(device)
    target_model.load_state_dict(torch.load('target_model.pth', map_location=device))
    
    # Create API simulator (simulates limited access to target model)
    api = APISimulator(target_model, device, query_limit=10000, add_defense=False)
    
    # Initialize attack
    attack = ModelExtractionAttack(api, device, budget=8000)
    
    # Execute attack
    training_history = attack.run_attack()
    
    # Save surrogate model
    torch.save(attack.surrogate_model.state_dict(), 'surrogate_model.pth')
    print("\nSurrogate model saved as 'surrogate_model.pth'")
    
    # Prepare test data for evaluation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    
    # Evaluate attack
    evaluation_results = evaluate_extraction_attack(
        target_model, 
        attack.surrogate_model, 
        test_loader, 
        device
    )
    
    # Save evaluation results
    with open('attack_evaluation_results.json', 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    print("\nEvaluation results saved to 'attack_evaluation_results.json'")
    
    # Generate attack report
    print("\n" + "="*60)
    print("ATTACK SUMMARY REPORT")
    print("="*60)
    print(f"Total Queries Used: {api.query_count}")
    print(f"Query Budget: {attack.budget}")
    print(f"Query Efficiency: {api.query_count/attack.budget*100:.1f}%")
    print(f"Final Surrogate Accuracy: {evaluation_results['surrogate_accuracy']:.2f}%")
    print(f"Extraction Fidelity: {evaluation_results['agreement_rate']:.2f}%")
    print(f"Attack Success: {'YES' if evaluation_results['attack_success'] else 'NO'}")
    print("="*60)


if __name__ == "__main__":
    main()