import numpy as np
import pandas as pd
from datetime import time
import matplotlib.pyplot as plt
from pyomo.environ import *
import pytz

def model_p2p(data):
    model = AbstractModel()

    # Sets
    model.T = Set() # Time period (e.g., hour, half-an-hour)
    model.H = Set() # Households
    model.H_pv = Set() # Set of Households with PV
    model.H_bat = Set() # Set of Households with Batteries
    model.P = model.H*model.H # Subset of network in the community (dim=2, H, H) (eg. P={(1,2), (1,3), (2,1), (2,3), (3,1), (3,2)})

    # Parameters
    model.P_spot = Param(model.T) # Spot price for electricity
    model.PV = Param(model.T) # PV production at each time and household with PV panels
    model.Dem = Param(model.T, model.H) # Demand at each time and household

    model.PV_cap = Param(model.H_pv) # Installed capacity PV for each house with pv

    model.Psi = Param()  # Losses in the community lines The local trade assumes losses of 7.6% through the local network (see [40]) in luth.

    model.Mu_c = Param()  # charging efficiency
    model.Mu_d = Param() # discharge efficiency
    model.Alpha = Param()  # charging rate 2.5 kW -> 2.5 kWh/hour at constant rate
    model.Beta = Param()  # discharging rate 2.5 kW -> 2.5 kWh/hour at constant rate
    model.Smax = Param()  # capacity batteries [kWh]
    model.Smin = Param()  # [kWh] here 20% of the maximum capacity
    model.S_init = Param()  # [kWh]

    # Variables
    model.x_g = Var(model.T, model.H, within=NonNegativeReals)  # sold power to community c
    model.d = Var(model.T, model.H_bat, within=NonNegativeReals)  # discharge from batteries
    model.c = Var(model.T, model.H_bat, within=NonNegativeReals)  # charging from batteries
    model.s = Var(model.T, model.H_bat, within=NonNegativeReals)  # state of battery

    model.x = Var(model.T, model.H, within=NonNegativeReals)  # total exports house h
    model.x_p = Var(model.T, model.P, within=NonNegativeReals)  # exports from house h to house p
    model.i = Var(model.T, model.H, within=NonNegativeReals)  # total imports house h
    model.i_p = Var(model.T, model.P, within=NonNegativeReals)  # imports of house h from house p

    model.r_charge = Var(model.T, model.H_bat, within=NonNegativeReals) # reserved charging capacity, how much i reduce of my charge
    model.r_discharge = Var(model.T, model.H_bat, within=NonNegativeReals) # reserved discharged capacity, how much is left of discharging
    model.z = Var(within=NonNegativeReals)

    # Objective function
    def objective_function(model):
        return sum(model.P_spot[t] * model.x_g[t, h] for t in model.T for h in model.H)

    model.objective_function = Objective(rule=objective_function, sense=minimize)

    def balance_equation(model, t, h): # For each time and household
        return (model.x_g[t, h] + (model.PV[t] if h in model.H_pv else 0)  + (model.d[t,h] if h in model.H_bat else 0)
                + model.i[t, h] >= model.Dem[t, h] + model.x[t, h] + (model.c[t, h] if h in model.H_bat else 0))

    model.balance_equation = Constraint(model.T, model.H, rule=balance_equation)

    # P2P constraints
    def sum_exports_household(model, h, t):
        return model.x[t, h] == sum(model.x_p[t, p] for p in model.P if p[0] == h)

    model.sum_exports_household = Constraint(model.H, model.T, rule=sum_exports_household)

    def sum_imports_household(model, h, t):
        return model.i[t, h] == sum(model.i_p[t, p] for p in model.P if p[0] == h)

    model.sum_imports_household = Constraint(model.H, model.T, rule=sum_imports_household)

    def balance_exports_imports(model, t):
        return sum(model.i[t, h] for h in model.H) == model.Psi * sum(model.x[t, h] for h in model.H)

    model.balance_exports_imports = Constraint(model.T, rule=balance_exports_imports)

    def balance_exports_imports_household(model, t, h0, h1):
        return model.i_p[t, h0, h1] == model.Psi * model.x_p[t, h1, h0]

    model.balance_exports_imports_household = Constraint(model.T, model.P, rule=balance_exports_imports_household)

    # Battery constraints

    def time_constraint(model, t, h):
        if t.time() == time(0,0): # when the hour is 00:00
            return model.s[t, h] == model.S_init + model.Mu_c * model.c[t, h] - 1/model.Mu_d * model.d[t, h]
        else:
            t_previous = t - pd.Timedelta(minutes=30)  # Calculate your previous t, change depending on your delta time
            return model.s[t, h] == model.s[t_previous, h] + model.Mu_c * model.c[t, h] - 1/model.Mu_d * model.d[t, h]

    model.time_constraint = Constraint(model.T, model.H_bat, rule=time_constraint)

    def min_SoC(model, t, h):
        return model.s[t, h] >= model.Smin

    model.min_SoC = Constraint(model.T, model.H_bat, rule=min_SoC)

    def charging_rate(model, t, h):
        return model.c[t, h] <= model.Alpha

    model.charging_rate = Constraint(model.T, model.H_bat, rule=charging_rate)

    def discharge_rate(model, t, h):
        return model.d[t, h] <= model.Beta

    model.discharge_rate = Constraint(model.T, model.H_bat, rule=discharge_rate)

    def max_SoC(model, t, h):
        return model.s[t, h] <= model.Smax

    model.max_SoC = Constraint(model.T, model.H_bat, rule=max_SoC)

    instance = model.create_instance(data)
    results = SolverFactory("gurobi", Verbose=True).solve(instance, tee=True)
    results.write()
    instance.solutions.load_from(results)

    return instance

def model_p2p_stratified(data):
    model = AbstractModel()

    # Sets
    model.T = Set() # Time period (e.g., hour, half-an-hour)
    model.H = Set() # Households
    model.H_pv = Set() # Set of Households with PV
    model.H_bat = Set() # Set of Households with Batteries
    model.P = model.H*model.H # Subset of network in the community (dim=2, H, H) (eg. P={(1,2), (1,3), (2,1), (2,3), (3,1), (3,2)})

    # Parameters
    model.P_spot = Param(model.T) # Spot price for electricity
    model.PV = Param(model.T) # PV production at each time and household with PV panels
    model.Dem = Param(model.T, model.H) # Demand at each time and household

    model.PV_cap = Param(model.H_pv) # Installed capacity PV for each house with pv

    model.Psi = Param()  # Losses in the community lines The local trade assumes losses of 7.6% through the local network (see [40]) in luth.

    model.Mu_c = Param()  # charging efficiency
    model.Mu_d = Param() # discharge efficiency
    model.Alpha = Param(model.H_bat)  # charging rate 2.5 kW -> 2.5 kWh/hour at constant rate
    model.Beta = Param(model.H_bat)  # discharging rate 2.5 kW -> 2.5 kWh/hour at constant rate
    model.Smax = Param(model.H_bat)  # capacity batteries [kWh]
    model.Smin = Param(model.H_bat)  # [kWh] here 20% of the maximum capacity
    model.S_init = Param(model.H_bat)  # [kWh]

    model.c_FFR = Param() # Price in the FFR market pence/kWh

    # Variables
    model.x_g = Var(model.T, model.H, within=NonNegativeReals)  # sold power to community c
    model.d = Var(model.T, model.H_bat, within=NonNegativeReals)  # discharge from batteries
    model.c = Var(model.T, model.H_bat, within=NonNegativeReals)  # charging from batteries
    model.s = Var(model.T, model.H_bat, within=NonNegativeReals)  # state of battery

    model.x = Var(model.T, model.H, within=NonNegativeReals)  # total exports house h
    model.x_p = Var(model.T, model.P, within=NonNegativeReals)  # exports from house h to house p
    model.i = Var(model.T, model.H, within=NonNegativeReals)  # total imports house h
    model.i_p = Var(model.T, model.P, within=NonNegativeReals)  # imports of house h from house p

    model.r_charge = Var(model.T, model.H_bat, within=NonNegativeReals) # reserved charging capacity, how much i reduce of my charge
    model.r_discharge = Var(model.T, model.H_bat, within=NonNegativeReals) # reserved discharged capacity, how much is left of discharging
    model.z = Var(within=NonNegativeReals)

    # Objective function
    def objective_function(model):
        return sum(model.P_spot[t] * model.x_g[t, h] for t in model.T for h in model.H) #- model.c_FFR * model.z * 48

    model.objective_function = Objective(rule=objective_function, sense=minimize)

    def balance_equation(model, t, h): # For each time and household
        return (model.x_g[t, h] + (model.PV[t] if h in model.H_pv else 0)  + (model.d[t,h] if h in model.H_bat else 0)
                + model.i[t, h] >= model.Dem[t, h] + model.x[t, h] + (model.c[t, h] if h in model.H_bat else 0))

    model.balance_equation = Constraint(model.T, model.H, rule=balance_equation)

    # P2P constraints
    def sum_exports_household(model, h, t):
        return model.x[t, h] == sum(model.x_p[t, p] for p in model.P if p[0] == h)

    model.sum_exports_household = Constraint(model.H, model.T, rule=sum_exports_household)

    def sum_imports_household(model, h, t):
        return model.i[t, h] == sum(model.i_p[t, p] for p in model.P if p[0] == h)

    model.sum_imports_household = Constraint(model.H, model.T, rule=sum_imports_household)

    def balance_exports_imports(model, t):
        return sum(model.i[t, h] for h in model.H) == model.Psi * sum(model.x[t, h] for h in model.H)

    model.balance_exports_imports = Constraint(model.T, rule=balance_exports_imports)

    def balance_exports_imports_household(model, t, h0, h1):
        return model.i_p[t, h0, h1] == model.Psi * model.x_p[t, h1, h0]

    model.balance_exports_imports_household = Constraint(model.T, model.P, rule=balance_exports_imports_household)

    # Battery constraints

    def time_constraint(model, t, h):
        if t.time() == time(0,0): # when the hour is 00:00
            return model.s[t, h] == model.S_init[h] + model.Mu_c * model.c[t, h] - 1/model.Mu_d * model.d[t, h]
        else:
            t_previous = t - pd.Timedelta(minutes=30)  # Calculate your previous t, change depending on your delta time
            return model.s[t, h] == model.s[t_previous, h] + model.Mu_c * model.c[t, h] - 1/model.Mu_d * model.d[t, h]

    model.time_constraint = Constraint(model.T, model.H_bat, rule=time_constraint)

    def min_SoC(model, t, h):
        return model.s[t, h] >= model.Smin[h]

    model.min_SoC = Constraint(model.T, model.H_bat, rule=min_SoC)

    def charging_rate(model, t, h):
        return model.c[t, h] <= model.Alpha[h]

    model.charging_rate = Constraint(model.T, model.H_bat, rule=charging_rate)

    def discharge_rate(model, t, h):
        return model.d[t, h] <= model.Beta[h]

    model.discharge_rate = Constraint(model.T, model.H_bat, rule=discharge_rate)

    def max_SoC(model, t, h):
        return model.s[t, h] <= model.Smax[h]

    model.max_SoC = Constraint(model.T, model.H_bat, rule=max_SoC)

    # FFR Constraints
    # def reserved_charge(model, t, h):
    #     return model.c[t, h] >= model.r_charge[t,h]
    #
    # model.reserved_charge = Constraint(model.T, model.H_bat, rule=reserved_charge)
    #
    # def reserved_discharge(model, t, h):
    #     return model.d[t, h] + model.r_discharge[t, h] <= model.Beta
    #
    # model.reserved_discharge = Constraint(model.T, model.H_bat, rule=reserved_discharge)
    #
    # def reserved_bound(model, t):
    #     return sum(model.r_charge[t,h] + model.r_discharge[t,h] for h in model.H_bat) >= model.z
    #
    # model.reserved_bound = Constraint(model.T, rule=reserved_bound)

    instance = model.create_instance(data)
    results = SolverFactory("gurobi", Verbose=True).solve(instance, tee=True)
    results.write()
    instance.solutions.load_from(results)

    return instance
