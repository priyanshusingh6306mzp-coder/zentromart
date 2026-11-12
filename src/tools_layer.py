# src/tools_layer.py
import importlib.util

import data
import dates
from config import DATASET_DIR


def _load_dataset_tools():
    path = DATASET_DIR / "tools" / "tools.py"
    if not path.is_file():
        raise SystemExit(f"tools/tools.py not found at {path}. "
                         "Set ZM_DATASET_DIR to the folder containing kb/, data/ and tools/.")
    spec = importlib.util.spec_from_file_location("zm_dataset_tools", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_tools = _load_dataset_tools()
lookup_order = _tools.lookup_order          # the provided tools
create_ticket = _tools.create_ticket

DELIVERED_STATUSES = ["delivered", "return_picked_up", "return_refund_initiated", "returned_refunded"]


def _verified_order(order_id, phone_last4=None, pincode=None):
    """Call the provided lookup_order (it takes ONE value per call).
    Returns the full tool result when the caller is verified, otherwise None."""
    for value in (pincode, phone_last4):
        if value:
            try:
                res = lookup_order(order_id, value)
            except Exception:
                return None
            if isinstance(res, dict) and res.get("verified") is True:
                return res
    return None


def verify_caller(order_id, phone_last4=None, pincode=None):
    if not (phone_last4 or pincode):
        return False, "needs_verification"
    if _verified_order(order_id, phone_last4, pincode):
        return True, ""
    return False, "verification_failed"      # the tool hides whether the order exists


def safe_lookup_order(order_id, phone_last4=None, pincode=None):
    if not (phone_last4 or pincode):
        return {"ok": False, "reason": "needs_verification", "order": None}
    res = _verified_order(order_id, phone_last4, pincode)
    if res is None:
        return {"ok": False, "reason": "verification_failed", "order": None}
    return {"ok": True, "reason": "", "order": res}


def order_facts(order_id, phone_last4=None, pincode=None):
    """Facts computed in code from the provided lookup_order. The LLM never does date math."""
    lookup = safe_lookup_order(order_id, phone_last4, pincode)
    if not lookup["ok"]:
        return {"error": lookup["reason"]}
    order = lookup["order"]
    status = order.get("status")
    is_delivered = status in DELIVERED_STATUSES
    is_shipped = is_delivered or status == "shipped"
    delivered_on = order.get("delivery_date") or ""

    facts = {
        "order_id": order.get("order_id") or order_id,
        "status": status,
        "shipped": is_shipped,
        "delivered": is_delivered,
        "days_late": dates.days_late(order.get("promised_date") or ""),
        "is_plus": bool(order.get("is_plus")),
        "items": [],
    }
    for item in order.get("items") or []:
        cat = str(item.get("return_category") or "").strip().lower() or "default"
        facts["items"].append({
            "sku": item.get("sku"),
            "category": cat,
            "within_return_window": dates.is_within_return_window(delivered_on, cat) if is_delivered else False,
        })
    return facts


def product_info(sku):
    """Catalog lookup: no verification required."""
    prod = data.get_product(sku)
    return prod if prod else {"error": "product_not_found"}


def pincode_info(pincode):
    """Delivery zone lookup: no verification required."""
    if data.pincodes.empty:
        return {"error": "pincode_data_missing"}
    res = data.pincodes[data.pincodes["pincode"] == pincode]
    return res.iloc[0].to_dict() if not res.empty else {"error": "pincode_not_found"}


def safe_create_ticket(summary, order_id=None, reason=None, intent=None):
    """Wrapper around the provided create_ticket. Success only when the tool says created=True."""
    text = (summary or "").strip()
    if reason:
        text += f" Escalation reason: {reason}."
    if len(text) < 20:                          # the tool rejects summaries under 20 characters
        text += " Escalated by the AI assistant for human review."
    try:
        res = create_ticket(text, order_id=order_id, reason=reason, intent=intent)
    except Exception as e:
        return {"created": False, "ticket_id": None, "error": str(e)[:120]}
    if not isinstance(res, dict):
        return {"created": False, "ticket_id": None, "error": "unexpected tool result"}
    return {"created": res.get("created") is True, "ticket_id": res.get("ticket_id"),
            "error": res.get("error")}