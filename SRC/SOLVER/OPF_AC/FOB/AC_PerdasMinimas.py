import pyomo.environ as pyo

def AC_PerdasMinimas(_self):
    perdas_minimas = sum(_self.var_lists['PGER_UTE'][g]
                         for g in range(_self.NUTE))

    custo_deficit = sum(10000 * _self.var_lists['deficit'][b]
                        for b in range(_self.NBAR))


    custo_bess_op = 0.0
    if _self.NBESS > 0:

        for t in range(_self.horizon_time):
            for i, bus in enumerate(_self.battery_buses):
                custo_bess_op += float(getattr(_self.sistema,
                                               'BATTERY_COST_OP', 0.01)) \
                                 * _self.BatteryOperation_dict[(t, bus)]

        for t in range(_self.horizon_time):
            for i, bus in enumerate(_self.battery_buses):
                custo_bess_op += 0.001 * (
                    _self.CHARGE_dict[(t, bus)]
                    + _self.DISCHARGE_dict[(t, bus)]
                )

    return perdas_minimas + custo_deficit + custo_bess_op