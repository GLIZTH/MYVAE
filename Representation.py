import pandas as pd
import numpy as np
from pymatgen.io.vasp import Poscar
import torch
import os
def read_from_poscar(filename,max_atom):
    poscar = Poscar.from_file(filename)
    structure = poscar.structure
    species=structure.species
    if(len(species)>max_atom):
        return torch.Tensor(0),torch.Tensor(0),torch.Tensor(0),False
    coords=structure.frac_coords
    lattice=np.array([[structure.lattice.a/30,structure.lattice.b/30,structure.lattice.c/30],[structure.lattice.alpha/180,structure.lattice.beta/180,structure.lattice.gamma/180]])
    return species,coords,lattice,True
def get_coords_tensor(max_atom,coords):
    coords_tensor=torch.Tensor(max_atom,3)
    for i in range(len(coords)):
        coords_tensor[i]=torch.Tensor(coords[i])
    for i in range(len(coords),max_atom):
        coords_tensor[i]=torch.rand(1,3)
    return coords_tensor
def get_coords_feature(max_atom,coords):
    coords_10000=(get_coords_tensor(max_atom,coords).detach().numpy()*10000).astype(int)

    coords_feature=torch.zeros(max_atom,42)
    for i in range(max_atom):
        bin_number1=bin(coords_10000[i][0])[2:]
        for j in range(0,len(bin_number1)):
            coords_feature[i][13-j]=torch.Tensor([float(bin_number1[len(bin_number1)-j-1])])
        bin_number2=bin(coords_10000[i][1])[2:]
        for j in range(0,len(bin_number2)):
            coords_feature[i][27-j]=torch.Tensor([float(bin_number2[len(bin_number2)-j-1])])
        bin_number3=bin(coords_10000[i][2])[2:]
        for j in range(0,len(bin_number3)):
            coords_feature[i][41-j]=torch.Tensor([float(bin_number3[len(bin_number3)-j-1])])
    return coords_feature.view(-1,max_atom,42)
def get_species_feature(data,max_atom,species):
    species_feature=torch.zeros(max_atom,7)
    for i in range(len(species)):
        bin_number=bin(int(data[str(species[i])]))[2:]
        for j in range(len(bin_number)):
            species_feature[i][6-j]=torch.Tensor([float(bin_number[len(bin_number)-j-1])])
    return species_feature.view(-1,max_atom,7)
def get_lattice_feature(lattice):
    lattice_feature=torch.zeros(2,49)
    lattice_10000=(lattice*10000).astype(int)
    for i in range(2):
        bin_number1=bin(lattice_10000[i][0])[2:]
        for j in range(0,len(bin_number1)):
                lattice_feature[i][13-j]=torch.Tensor([float(bin_number1[len(bin_number1)-j-1])])
        bin_number2=bin(lattice_10000[i][1])[2:]
        for j in range(0,len(bin_number2)):
            lattice_feature[i][27-j]=torch.Tensor([float(bin_number2[len(bin_number2)-j-1])])
        bin_number3=bin(lattice_10000[i][2])[2:]
        for j in range(0,len(bin_number3)):  
            lattice_feature[i][41-j]=torch.Tensor([float(bin_number3[len(bin_number3)-j-1])])
    return lattice_feature.view(-1,2,49)