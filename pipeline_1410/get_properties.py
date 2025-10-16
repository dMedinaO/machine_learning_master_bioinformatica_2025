import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from modlamp.descriptors import GlobalDescriptor
import numpy as np
import sys

def mw(sequence):
    gd = GlobalDescriptor(sequence)
    gd.calculate_MW(amide=True)
    val = float(np.round(gd.descriptor[0], 5))
    return val

def isoelectric_point(sequence):
    gd = GlobalDescriptor(sequence)
    gd.isoelectric_point(amide=True)
    val = float(np.round(gd.descriptor[0], 5))
    return val

def charge_density(sequence):
    gd = GlobalDescriptor(sequence)
    gd.charge_density(amide=True)
    val = float(np.round(gd.descriptor[0], 5))
    return val

def instability_index(sequence):
    gd = GlobalDescriptor(sequence)
    gd.instability_index()
    val = float(np.round(gd.descriptor[0], 5))
    return val

def boman_index(sequence):
    gd = GlobalDescriptor(sequence)
    gd.boman_index()
    val = float(np.round(gd.descriptor[0], 5))
    return val

def hydrophobic_ratio(sequence):
    gd = GlobalDescriptor(sequence)
    gd.hydrophobic_ratio()
    val = float(np.round(gd.descriptor[0], 5))
    return val

print("Collecting input data")
df_data = pd.read_csv(sys.argv[1])
name_sequence = sys.argv[2]
output = sys.argv[3]

print("Applying function")
df_data["mw"] = df_data[name_sequence].apply(mw)
df_data["hydrophobic_ratio"] = df_data[name_sequence].apply(hydrophobic_ratio)
df_data["boman_index"] = df_data[name_sequence].apply(boman_index)
df_data["instability_index"] = df_data[name_sequence].apply(instability_index)
df_data["charge_density"] = df_data[name_sequence].apply(charge_density)
df_data["isoelectric_point"] = df_data[name_sequence].apply(isoelectric_point)

print("Exporting data")
df_data.to_csv(output, index=False)