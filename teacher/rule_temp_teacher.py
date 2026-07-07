# rule_temp_teacher.py
from datetime import datetime, timedelta

def get_adaptive_setpoint(price, t_out, q_sol, t_low, t_high,
                          avg_price=None,future_q_sol_avg=None,
                          future_q_sol_max=None,future_tout_max=None,
                          time_seconds=None):

    is_away = (t_low < 20.5)


    if t_out < 10.0:
        base_target = (t_low + t_high) / 2.0- 0.5
    elif 10.0 <= t_out < 12.0:
        base_target = (t_low + t_high) / 2.0 - 1.0
    else:
        base_target = (t_low + t_high) / 2.0 - 1.5

    adjustment = 0.0


    if avg_price is None:
        ref_high = 0.26
        ref_low = 0.24
    else:
        ref_high = avg_price * 1.05
        ref_low  = avg_price * 0.95

    if price >= ref_high:
        if t_out < 10.0:
            adjustment -= 0.0
        else:
            adjustment -= 0.5 if is_away else 1.0  
    elif price <= ref_low:
        if t_out > 10.0:
            adjustment += 0.2
            if future_q_sol_avg > 300:
                adjustment -= 0.5
        else:
            adjustment += 0.5
    else:
        adjustment += 0.0


    if q_sol > 0:
        solar_impact = (q_sol / 100.0) * 0.16
        if t_out > 15.0:
            solar_impact *= 2.0
        if t_out > 20.0:
            solar_impact *= 2.3


        if t_out < 5.0:
            adjustment -= min(0.5, solar_impact)
        else:
            adjustment -= min(2.5, solar_impact)


    if t_out < -15.0:
        adjustment += 0.5


    if is_away and (time_seconds is not None):
        base_date = datetime(2019, 1, 1)
        current_dt = base_date + timedelta(seconds=float(time_seconds))
        weekday = current_dt.weekday()
        hour = current_dt.hour + current_dt.minute / 60.0

        if is_away and (14.0 <= hour < 18.0):

            if avg_price is not None and price < avg_price:
                adjustment += 1.0
            else:
                adjustment += 0.5



    is_solar_danger = (
            (future_q_sol_max is not None) and (future_tout_max is not None) and
            (future_q_sol_max > 700) and (future_tout_max > 10.0)
    )

    if (future_q_sol_max is not None) and (future_tout_max is not None):
        if (future_q_sol_max > 600) and (future_tout_max > 12.0):
            adjustment -= 1.0
        elif (future_q_sol_max > 450) and (future_tout_max > 10.0):
            adjustment -= 0.5


    if is_solar_danger:
        if adjustment > 0:
            adjustment = 0.0
        adjustment -= 1.5


    final_target = base_target + adjustment

    if is_away:
        safe_low = t_low
  
        safe_high = t_high - 1.5 if is_solar_danger else t_high - 0.5
    else:
        buffer = 0.0 if t_out > 10.0 else 0.2
        safe_low = t_low + buffer
     
        safe_high = t_high - 1.0 if is_solar_danger else t_high - 0.1

    final_target = max(safe_low, min(safe_high, final_target))
    return final_target




