import os
import pandas as pd
from src.data import generate_data, get_train_val_test_splits
from src.baseline import BaselineRanker, evaluate_baseline
from src.model import engineer_features, MatchModel, evaluate_model
from src.quality_check import run_conversion_quality_check
from src.payment import PaymentGateway, InternalSystem, reconcile_payments

def main():
    print("==================================================")
    print("PlaceMux: Conversion-Quality Check Demo")
    print("==================================================\n")
    
    # 1. Load real sample data
    print("1. Generating and loading synthetic data...")
    df = generate_data(num_samples=1000, random_state=42)
    train_df, val_df, test_df = get_train_val_test_splits(df)
    
    print(f"Total records: {len(df)}")
    print("Row counts per payment status:")
    print(df['payment_status'].value_counts().to_string())
    print()

    # 2. Confirm baseline is ready and run it
    print("2. Running Baseline Ranker...")
    baseline = BaselineRanker(threshold=0.6)
    try:
        baseline.check_readiness()
        print("Baseline readiness check: PASSED")
    except RuntimeError as e:
        print(f"Baseline readiness check: FAILED - {e}")
        return
        
    baseline_metrics = evaluate_baseline(test_df, baseline, split_name="Test")
    print()

    # 3. Train ML Model (if not exists) and load from disk
    model_path = os.path.join("models", "conversion_quality_model.joblib")
    
    if not os.path.exists(model_path):
        print("Model artifact not found. Training and saving initial model...")
        X_train = engineer_features(train_df)
        y_train = train_df['is_good_match']
        X_val = engineer_features(val_df)
        y_val = val_df['is_good_match']
        
        temp_model = MatchModel()
        temp_model.train(X_train, y_train)
        
        # Simple tuning / check on validation set
        print("Validating model on Val set:")
        evaluate_model(temp_model, X_val, y_val, split_name="Val")
        
        temp_model.save(model_path)
        print(f"Model saved to {model_path}\n")
    
    print("3. Loading Trained Model Artifact from disk...")
    model = MatchModel.load(model_path)
    X_test = engineer_features(test_df)
    y_test = test_df['is_good_match']
    model_metrics = evaluate_model(model, X_test, y_test, split_name="Test")
    
    # Compare with baseline
    print(f"Delta vs Baseline -> Precision: {model_metrics['precision'] - baseline_metrics['precision']:.3f}, Recall: {model_metrics['recall'] - baseline_metrics['recall']:.3f}")
    print()

    # 4. Pick one concrete example
    print("4. Concrete Example Explainability...")
    sample_row = test_df.iloc[0:1]
    sample_X = engineer_features(sample_row)
    explanation = model.explain(sample_row, sample_X)
    print(f"Student Payment Status: {sample_row.iloc[0]['payment_status']}")
    print(f"Explanation: {explanation}")
    print()
    
    # 5. Run conversion-quality check
    print("5. Running Conversion-Quality Check...")
    # Evaluate across the entire dataset or test set. Using full dataset here for comprehensive segment analysis
    run_conversion_quality_check(df, model, baseline_metrics, threshold=0.05)
    print()

    # 6. Simulate Payment Failure Handling
    print("6. Simulating Payment Failure mid-transaction...")
    gateway = PaymentGateway(mode="test")
    system = InternalSystem()
    
    print("Attempting to process application for 'student_fail_1'...")
    success = system.process_application("student_fail_1", "job_1", gateway)
    app_status = system.applications["app_student_fail_1_job_1"]["status"]
    print(f"Payment Success: {success}")
    print(f"Application state after failure: {app_status} (Not unlocked, not double-charged)")
    print()

    # 7. Gateway reconciliation check
    print("7. Running Gateway Reconciliation Check...")
    mismatches = reconcile_payments(system.payment_records, gateway.transactions)
    if not mismatches:
        print("0 mismatches found. Internal and Gateway are perfectly synced.")
    else:
        for m in mismatches:
            print(m)
            
    # Artificially inject a mismatch to prove it works
    print("\nInjecting a fake mismatch to prove detection...")
    system.payment_records["idem_app_student_fail_1_job_1"]["status"] = "completed"
    mismatches = reconcile_payments(system.payment_records, gateway.transactions)
    for m in mismatches:
        print(m)

if __name__ == "__main__":
    main()
