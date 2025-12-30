"""
Main entry point for CLTV prediction pipeline.
Demonstrates full workflow from data preprocessing to model evaluation.
"""

import sys
sys.path.append('.')

from src import (
    DataPreprocessor, save_processed_data,
    create_dataloaders, save_scaler_and_features,
    create_model, Trainer,
    evaluate_model, plot_predictions, plot_training_history,
    set_seed, get_device, create_experiment_dir
)
import torch
import pandas as pd


def main():
    """Run full CLTV prediction pipeline."""
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    
    # Create experiment directory
    exp_dir = create_experiment_dir('experiments')
    
    print("\n" + "="*70)
    print("GPU-ACCELERATED CLTV PREDICTION PIPELINE")
    print("="*70)
    
    # ========== STEP 1: Data Preprocessing ==========
    print("\n[STEP 1] DATA PREPROCESSING")
    print("-" * 70)
    
    preprocessor = DataPreprocessor(
        observation_days=365,  # 1 year of historical data
        prediction_days=90     # Predict 90 days ahead
    )
    
    # Process data
    train_df, test_df = preprocessor.process('online_retail_II.csv')
    
    # Save processed data
    save_processed_data(train_df, test_df, output_dir='data')
    
    # ========== STEP 2: Create DataLoaders ==========
    print("\n[STEP 2] CREATING DATALOADERS")
    print("-" * 70)
    
    train_loader, test_loader, scaler, feature_cols = create_dataloaders(
        train_df=train_df,
        test_df=test_df,
        batch_size=256,
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available()
    )
    
    # Save scaler and features for deployment
    save_scaler_and_features(scaler, feature_cols, output_dir=exp_dir)
    
    input_dim = len(feature_cols)
    
    # ========== STEP 3: Baseline Model ==========
    print("\n[STEP 3] TRAINING BASELINE MODEL")
    print("-" * 70)
    
    baseline_model = create_model(
        model_type='baseline',
        input_dim=input_dim,
        device=device,
        hidden_dims=[256, 128, 64]
    )
    
    baseline_trainer = Trainer(
        model=baseline_model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=1e-3,
        weight_decay=1e-5,
        use_amp=True,
        use_zero_inflated_loss=False
    )
    
    baseline_history = baseline_trainer.train(
        num_epochs=50,
        save_dir=exp_dir,
        model_name='baseline'
    )
    
    # Evaluate baseline
    _, baseline_preds, baseline_targets = baseline_trainer.evaluate()
    baseline_metrics = evaluate_model(baseline_preds, baseline_targets, "Baseline NN")
    
    # ========== STEP 4: Advanced Mixture-of-Experts Model ==========
    print("\n[STEP 4] TRAINING MIXTURE-OF-EXPERTS MODEL")
    print("-" * 70)
    
    moe_model = create_model(
        model_type='moe',
        input_dim=input_dim,
        device=device,
        num_experts=4,
        expert_hidden_dim=128,
        gate_hidden_dim=64,
        use_zero_inflation=True
    )
    
    moe_trainer = Trainer(
        model=moe_model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=1e-3,
        weight_decay=1e-5,
        use_amp=True,
        use_zero_inflated_loss=True
    )
    
    moe_history = moe_trainer.train(
        num_epochs=50,
        save_dir=exp_dir,
        model_name='moe'
    )
    
    # Evaluate MoE
    _, moe_preds, moe_targets = moe_trainer.evaluate()
    moe_metrics = evaluate_model(moe_preds, moe_targets, "Mixture-of-Experts")
    
    # ========== STEP 5: Comparison and Visualization ==========
    print("\n[STEP 5] RESULTS COMPARISON")
    print("-" * 70)
    
    print("\nBaseline vs Mixture-of-Experts:")
    print(f"{'Metric':<20} {'Baseline':<15} {'MoE':<15} {'Improvement':<15}")
    print("-" * 65)
    
    for metric in ['rmse', 'mae', 'gini', 'top_10_capture', 'lift_at_10']:
        baseline_val = baseline_metrics[metric]
        moe_val = moe_metrics[metric]
        
        if metric in ['rmse', 'mae']:
            improvement = ((baseline_val - moe_val) / baseline_val * 100)
            print(f"{metric.upper():<20} {baseline_val:<15.2f} {moe_val:<15.2f} {improvement:>+.1f}%")
        else:
            improvement = ((moe_val - baseline_val) / baseline_val * 100)
            print(f"{metric:<20} {baseline_val:<15.4f} {moe_val:<15.4f} {improvement:>+.1f}%")
    
    # Plot results
    print("\nGenerating visualizations...")
    
    plot_training_history(
        baseline_history,
        save_path=f'{exp_dir}/baseline_training.png'
    )
    
    plot_training_history(
        moe_history,
        save_path=f'{exp_dir}/moe_training.png'
    )
    
    plot_predictions(
        baseline_targets,
        baseline_preds,
        model_name="Baseline NN",
        save_path=f'{exp_dir}/baseline_predictions.png'
    )
    
    plot_predictions(
        moe_targets,
        moe_preds,
        model_name="Mixture-of-Experts",
        save_path=f'{exp_dir}/moe_predictions.png'
    )
    
    # ========== STEP 6: Save Results ==========
    print("\n[STEP 6] SAVING RESULTS")
    print("-" * 70)
    
    # Save metrics
    results_df = pd.DataFrame({
        'Model': ['Baseline', 'MoE'],
        'RMSE': [baseline_metrics['rmse'], moe_metrics['rmse']],
        'MAE': [baseline_metrics['mae'], moe_metrics['mae']],
        'Gini': [baseline_metrics['gini'], moe_metrics['gini']],
        'Top10_Capture': [baseline_metrics['top_10_capture'], moe_metrics['top_10_capture']],
        'Lift@10': [baseline_metrics['lift_at_10'], moe_metrics['lift_at_10']]
    })
    
    results_df.to_csv(f'{exp_dir}/results.csv', index=False)
    print(f"Saved results to {exp_dir}/results.csv")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nAll outputs saved to: {exp_dir}")


if __name__ == '__main__':
    main()
