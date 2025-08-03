import Bio.PDB
import os
from tqdm import tqdm
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import SDWriter
from scipy.spatial import KDTree
from Bio.PDB.PDBIO import Select
import argparse

def read_pdb(pdb_file, name=None):
    """Read pdb file into Biopython structure."""
    if name is None:
        name = os.path.basename(pdb_file)
    parser = Bio.PDB.PDBParser(QUIET=True)
    return parser.get_structure(name, pdb_file)

def get_ligand_code(path):
    """Extract 4-character PDB ID code from full path."""
    filename = os.path.splitext(os.path.basename(path))[0]
    return filename.split('_', 2)[1]

def get_ligand_pdb(ligand_pdb):
    """Read ligand into RDKit Mol."""
    lig = Chem.MolFromPDBFile(str(ligand_pdb), removeHs=True)
    if lig is None:
        print('failed')
        return None
    return lig

def get_pdb_code(path):
    """Extract 4-character PDB ID code from full path."""
    return str(path).split('/')[-1][:4].lower()

def get_pocket_res(rna, ligand, dist):
    """Extract residues within specified distance of ligand."""
    prot_atoms = [a for a in rna.get_atoms()]
    prot_coords = [atom.get_coord() for atom in prot_atoms]

    lig_coords = []
    for i in range(ligand.GetNumAtoms()):
        pos = ligand.GetConformer().GetAtomPosition(i)
        lig_coords.append([pos.x, pos.y, pos.z])

    kd_tree = KDTree(prot_coords)
    key_pts = kd_tree.query_ball_point(lig_coords, r=dist, p=2.0)
    key_pts = set([k for l in key_pts for k in l])

    key_residues = set()
    for i in key_pts:
        atom = prot_atoms[i]
        res = atom.get_parent()
        if res.get_resname() != 'HOH':
            key_residues.add(res)
    return key_residues

class PocketSelect(Select):
    """Selection class for subsetting RNA to key binding residues."""
    def __init__(self, reslist):
        self.reslist = reslist

    def accept_residue(self, residue):
        return residue in self.reslist

def write_files(pdbid, ligandid, rna, ligand, pocket, out_path):
    """Writes cleaned structure files for RNA, ligand, and pocket."""
    io = Bio.PDB.MMCIFIO()
    io.set_structure(rna)
    io.save(os.path.join(out_path, f"{pdbid}_rna.cif"))

    io.save(os.path.join(out_path, f"{pdbid}_{ligandid}_pocket.cif"), PocketSelect(pocket))

    writer = Chem.SDWriter(os.path.join(out_path, f"{pdbid}_{ligandid}_ligand.sdf"))
    writer.write(ligand)

def produce_cleaned_dataset(structure_dict, out_path, dist=10.0):
    """Generate and save cleaned dataset."""
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for pdb, data in tqdm(structure_dict.items(), desc='writing to files'):
        rna = structure_dict[pdb]['rna']
        for ligand_id in list(data.keys())[1:]:
            ligand = structure_dict[pdb][ligand_id]
            if ligand is None:
                continue
            pocket_res = get_pocket_res(rna, ligand, dist)
            write_files(pdb, ligand_id, rna, ligand, pocket_res, out_path)

def main(ligands_path, rna_path, output_dir, dist=6.0):
    ligands_files = [str(path) for path in Path(ligands_path).rglob('*.pdb')]
    rna_files = [str(path) for path in Path(rna_path).rglob('*.pdb')]

    structure_dict = {}

    for f in tqdm(rna_files, desc='RNA pdb files'):
        pdb_id = get_pdb_code(f)
        rna = read_pdb(f)
        structure_dict[pdb_id] = {'rna': rna}

    for f in tqdm(ligands_files, desc='Ligand pdb files'):
        pdb_id = get_pdb_code(f)
        ligand_id = get_ligand_code(f)
        if pdb_id in structure_dict:
            structure_dict[pdb_id][ligand_id] = get_ligand_pdb(f)

    produce_cleaned_dataset(structure_dict, output_dir, dist)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PDB files to generate cleaned datasets.")
    parser.add_argument('--ligands_path', type=str, required=True, help="Path to ligand PDB files")
    parser.add_argument('--rna_path', type=str, required=True, help="Path to RNA PDB files")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save output files")
    parser.add_argument('--dist', type=float, default=6.0, help="Distance cutoff for defining binding pocket")

    args = parser.parse_args()
    main(args.ligands_path, args.rna_path, args.output_dir, args.dist)

#python process_pdb.py --ligands_path processed_data/ligands/ --rna_path processed_data/rna/ --output_dir output_dir/ --dist 6.0
