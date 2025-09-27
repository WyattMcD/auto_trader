# positions.py
def is_option_symbol(sym: str) -> bool:
    # e.g., AAPL251018P00190000
    return len(sym) > 15 and (sym[-9] in ("C","P"))

def closing_side(open_side: str) -> str:
    return "buy" if open_side == "sell" else "sell"
