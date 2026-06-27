import pytest
from src.payment import PaymentGateway, InternalSystem, reconcile_payments

def test_successful_payment():
    gateway = PaymentGateway(mode="test")
    system = InternalSystem()
    
    success = system.process_application("student1", "job1", gateway)
    assert success is True
    
    app_status = system.applications["app_student1_job1"]["status"]
    assert app_status == "unlocked"
    
    # Check no mismatches
    mismatches = reconcile_payments(system.payment_records, gateway.transactions)
    assert len(mismatches) == 0

def test_mid_transaction_failure():
    gateway = PaymentGateway(mode="test")
    system = InternalSystem()
    
    # User ID with "fail" triggers simulated mid-transaction drop
    success = system.process_application("student_fail", "job1", gateway)
    assert success is False
    
    app_status = system.applications["app_student_fail_job1"]["status"]
    assert app_status == "payment_failed" # Not unlocked, not corrupted
    
    # Check no mismatches (both should show failed/missing)
    mismatches = reconcile_payments(system.payment_records, gateway.transactions)
    assert len(mismatches) == 0

def test_idempotent_retry():
    gateway = PaymentGateway(mode="test")
    system = InternalSystem()
    
    # First attempt
    success1 = system.process_application("student2", "job2", gateway)
    assert success1 is True
    
    # Count gateway transactions
    assert len(gateway.transactions) == 1
    
    # Simulate a retry on the same application
    success2 = system.process_application("student2", "job2", gateway)
    assert success2 is True
    
    # Gateway transactions should still be 1 because of idempotency key
    assert len(gateway.transactions) == 1

def test_reconciliation_mismatch():
    gateway = PaymentGateway(mode="test")
    system = InternalSystem()
    
    system.process_application("student3", "job3", gateway)
    
    # Artificially create a mismatch
    idemp_key = "idem_app_student3_job3"
    
    # 1. Gateway succeeded, but internal thinks it failed
    system.payment_records[idemp_key]["status"] = "failed"
    
    mismatches = reconcile_payments(system.payment_records, gateway.transactions)
    assert len(mismatches) == 1
    assert "Gateway claims success" in mismatches[0]
