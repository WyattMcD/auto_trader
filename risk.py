# risk.py
from config import MAX_OPEN_CSP_POSITIONS, MAX_BP_AT_RISK

def approve_csp_intent(intent, account, open_csp_count: int, est_bpr: float) -> bool:
    if open_csp_count >= MAX_OPEN_CSP_POSITIONS:
        return False
    # Rough guard: est BPR vs buying power
    try:
        bp = float(getattr(account, "buying_power", 0))
        if est_bpr > bp * MAX_BP_AT_RISK:
            return False
    except Exception:
        pass
    return True
