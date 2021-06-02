import numpy as np
import pandas as pd

d_filename = "small_molecule_drug_sim.xlsx"
filename = "association.xls"
v_filename = "virus_sim(2020.2.12).xlsx"

dsim = pd.read_excel(d_filename, header=None)
vsim = pd.read_excel(v_filename, header=None)
association = pd.read_excel(filename)

print(dsim, vsim, association.to_numpy()[:, 1:])

np.save("dsim", dsim.to_numpy())
np.save("vsim", vsim.to_numpy())
np.save("association", association.to_numpy()[:, 1:])
