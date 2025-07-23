# SPONCHpop project
# Physical disk model
#
# by Mihkel Kama and Oliver Shorttle
# 2020

import sys
import matplotlib.pyplot as plt
import numpy as np
import physcon
from scipy import integrate

debug        = False

print("")
print("---------------------------")
print("---------------------------")
print("-- Disk structure module --")
print("---------------------------")
print("---------------------------")
print("   All units are SI unless stated otherwise.")
print("")

### Physical constants ###
M_sun                   = 1.989e+30            # kg
R_sun                   = 6.957e+8             # m
gamma                   = 1.4                  # adiabatic index
auSI                    = 1.496e+11            # m
yr                      = 365.2425*24*3600     # s
M_e                     = 5.972e24             # kg