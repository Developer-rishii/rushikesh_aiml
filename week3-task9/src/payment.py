import uuid

class PaymentGateway:
    """Mock external payment gateway."""
    def __init__(self, mode="test"):
        self.mode = mode
        self.transactions = {}
        
    def charge(self, amount, user_id, idempotency_key):
        if idempotency_key in self.transactions:
            return self.transactions[idempotency_key]
            
        # Simulate mid-transaction failure based on some condition or random chance
        # For our purposes, we will allow forcing a failure via user_id
        if "fail" in user_id:
            raise Exception("Gateway error: Connection dropped mid-transaction.")
            
        record = {
            "amount": amount,
            "status": "success",
            "user_id": user_id,
            "idempotency_key": idempotency_key
        }
        self.transactions[idempotency_key] = record
        return record

class InternalSystem:
    def __init__(self):
        self.applications = {}
        self.payment_records = {}
        self.test_mode = True

    def process_application(self, student_id, job_id, gateway, amount=10.0):
        print(f"[{'TEST MODE' if self.test_mode else 'LIVE MODE'}] Processing application for {student_id}...")
        
        app_id = f"app_{student_id}_{job_id}"
        idemp_key = f"idem_{app_id}"
        
        # Pre-payment state
        self.applications[app_id] = {"status": "pending_payment"}
        self.payment_records[idemp_key] = {"status": "initiated", "amount": amount}
        
        try:
            gateway_res = gateway.charge(amount, student_id, idemp_key)
            
            if gateway_res["status"] == "success":
                self.applications[app_id]["status"] = "unlocked"
                self.payment_records[idemp_key]["status"] = "completed"
                return True
                
        except Exception as e:
            # Payment failed mid-transaction
            print(f"Payment failed for {student_id}: {str(e)}")
            self.applications[app_id]["status"] = "payment_failed"
            self.payment_records[idemp_key]["status"] = "failed"
            return False

def reconcile_payments(internal_records, gateway_transactions):
    """
    Gateway vs Internal mismatch check.
    Returns a list of mismatches.
    """
    mismatches = []
    
    # Check for payments we think are completed but gateway doesn't have
    for key, record in internal_records.items():
        if record["status"] == "completed":
            if key not in gateway_transactions or gateway_transactions[key]["status"] != "success":
                mismatches.append(f"Mismatch: Internal claims completed, Gateway missing or failed for {key}")
                
    # Check for payments gateway thinks are successful but we don't
    for key, record in gateway_transactions.items():
        if record["status"] == "success":
            if key not in internal_records or internal_records[key]["status"] != "completed":
                mismatches.append(f"Mismatch: Gateway claims success, Internal missing or not completed for {key}")
                
    return mismatches
