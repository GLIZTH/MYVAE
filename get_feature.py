from Representation import *
def generatre_feature(pos_dir,data,max_atom):
    file_list = os.listdir(pos_dir)
    file_list = sorted(file_list, key=lambda x: int(x))
    species_feature=torch.Tensor(0)
    coords_feature=torch.Tensor(0)
    lattice_feature=torch.Tensor(0)

    for j in file_list[0:1000]:
        print(os.path.join(pos_dir,j))
        species,coords,lattice,value=read_from_poscar(os.path.join(pos_dir,j),max_atom)
        if value==True:
            feature1=get_species_feature(data,max_atom,species)
            feature2=get_coords_feature(max_atom,coords)
            feature3=get_lattice_feature(lattice)
            species_feature=torch.cat((species_feature,feature1))
            coords_feature=torch.cat((coords_feature,feature2))
            lattice_feature=torch.cat((lattice_feature,feature3))

    torch.save( species_feature,'species_feature.pt')
    torch.save( coords_feature,'coords_feature.pt')
    torch.save( lattice_feature,'lattice_feature.pt')

    return species_feature,coords_feature,lattice_feature
def merge_feature(max_atom,pos_dir):
    data=pd.read_json('atom.json')
    species_feature,coords_feature,lattice_feature=generatre_feature(pos_dir,data,max_atom)
    feature=torch.cat((species_feature,coords_feature),dim=2)
    feature=torch.cat((lattice_feature,feature),dim=1)
    torch.save( feature,'feature.pt')
    return feature
max_atom=40
pos_dir='tclposfile'
feature=merge_feature(max_atom,pos_dir)

