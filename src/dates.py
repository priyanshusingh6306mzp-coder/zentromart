# src/dates.py
from datetime import datetime, timedelta
import data
from policy_params import POLICY_PARAMS

# HARD CAPSTONE RULE: Today is always this exact date
TODAY = datetime(2026, 10, 20)

def _parse(d_str):
    if not d_str: return None
    return datetime.strptime(d_str.split()[0], "%Y-%m-%d")

def add_business_days(start_date_str, n):
    current = _parse(start_date_str)
    if not current: return None
    
    holidays = [_parse(h).date() for h in data.get_holidays()]
    added = 0
    
    while added < int(n):
        current += timedelta(days=1)
        # Weekday 6 is Sunday. Skip Sundays and Holidays.
        if current.weekday() != 6 and current.date() not in holidays:
            added += 1
            
    return current.strftime("%Y-%m-%d")

def days_late(promised_date_str):
    promised = _parse(promised_date_str)
    if not promised: return 0
    late = (TODAY - promised).days
    return max(0, late)

def is_within_return_window(delivery_date_str, category):
    delivery = _parse(delivery_date_str)
    if not delivery: return False
    
    window = POLICY_PARAMS["return_windows_calendar_days"].get(
        category.lower(), 
        POLICY_PARAMS["return_windows_calendar_days"]["default"]
    )
    
    end_date = delivery + timedelta(days=window)
    return TODAY <= end_date