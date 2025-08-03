import pandas as pd
import os
from pathlib import Path
import subprocess
import lmdb
import json
import tqdm
import io
import gzip
from rdkit import Chem
import torch
from torch.utils.data import Dataset
import argparse
import lba.datasets.datasets as da
import lba.protein.sequence as seq
import lba.util.file as fi
import lba.util.formats as fo


class Scores(object):
    def __init__(self, score_path):
        self._scores = pd.read_csv(
            score_path,
            delimiter=",",
            engine="python",
            index_col="pdb",
            usecols=["pdb", "neglog_aff"],
        ).dropna()

    def _lookup(self, pdbcode):
        if pdbcode in self._scores.index:
            return self._scores.loc[pdbcode].to_dict()
        return None

    def __call__(self, x, error_if_missing=False):
        x["scores"] = self._lookup(x["id"])
        if x["scores"] is None and error_if_missing:
            raise RuntimeError(f'Unable to find scores for {x["id"]}')
        return x


def find_files(path, suffix, relative=False):
    path = Path(path)
    find_cmd = f"find {path} -maxdepth 1 -type f -name '*{suffix}' | sort"

    out = subprocess.Popen(
        find_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    stdout, stderr = out.communicate()

    if out.returncode != 0:
        raise Exception(f"Error finding files: {stderr.decode().strip()}")

    name_list = stdout.decode().strip().split("\n")
    name_list = [x for x in name_list if x]

    if relative:
        return [os.path.basename(x) for x in name_list]
    else:
        return [os.path.basename(x) for x in name_list]


def get_pdb_code(path):
    return str(path).split("/")[-1][:4].lower()


class SequenceReader(object):
    def __init__(self, protein_dir):
        self._protein_dir = protein_dir

    def _lookup(self, file_path):
        return seq.get_chain_sequences(fo.bp_to_df(fo.read_any(file_path)))

    def __call__(self, x, error_if_missing=False):
        x["seq"] = self._lookup(x["file_path"])
        del x["file_path"]
        if x["seq"] is None and error_if_missing:
            raise RuntimeError(f'Unable to find AA sequence for {x["id"]}')
        return x


class SmilesReader(object):
    def _lookup(self, file_path):
        ligand = fo.read_sdf_to_mol(
            file_path, sanitize=False, add_hs=False, remove_hs=True
        )[0]
        return Chem.MolToSmiles(ligand)

    def __call__(self, x, error_if_missing=False):
        x["smiles"] = self._lookup(x["file_path"])
        del x["file_path"]
        if x["smiles"] is None and error_if_missing:
            raise RuntimeError(f'Unable to find SMILES for {x["id"]}')
        return x


class LMDBDataset(Dataset):
    """
    Creates a dataset from an lmdb file. Adapted from `TAPE <https://github.com/songlab-cal/tape/blob/master/tape/datasets.py>`_.

    :param data_file: path to LMDB file containing dataset
    :type data_file: Union[str, Path]
    :param transform: Transformation function to apply to each item.
    :type transform: Function, optional

    """

    def __init__(self, data_file, transform=None):
        """constructor"""
        if type(data_file) is list:
            if len(data_file) != 1:
                raise RuntimeError("Need exactly one filepath for lmdb")
            data_file = data_file[0]

        self.data_file = Path(data_file).absolute()
        if not self.data_file.exists():
            raise FileNotFoundError(self.data_file)

        env = lmdb.open(
            str(self.data_file),
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        with env.begin(write=False) as txn:
            self._num_examples = int(txn.get(b"num_examples"))
            self._serialization_format = txn.get(b"serialization_format").decode()
            self._id_to_idx = deserialize(
                txn.get(b"id_to_idx"), self._serialization_format
            )

        self._env = env
        self._transform = transform

    def __len__(self) -> int:
        return self._num_examples

    def get(self, id: str):
        idx = self.id_to_idx(id)
        return self[idx]

    def id_to_idx(self, id: str):
        if id not in self._id_to_idx:
            raise IndexError(id)
        idx = self._id_to_idx[id]
        return idx

    def ids_to_indices(self, ids):
        return [self.id_to_idx(id) for id in ids]

    def ids(self):
        return list(self._id_to_idx.keys())

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)

        with self._env.begin(write=False) as txn:

            compressed = txn.get(str(index).encode())
            buf = io.BytesIO(compressed)
            with gzip.GzipFile(fileobj=buf, mode="rb") as f:
                serialized = f.read()
            try:
                item = deserialize(serialized, self._serialization_format)
            except:
                return None
        # Recover special data types (currently only pandas dataframes).
        if "types" in item.keys():
            for x in item.keys():
                if item["types"][x] == str(pd.DataFrame):
                    item[x] = pd.DataFrame(**item[x])
        else:
            logging.warning(
                "Data types in item %i not defined. Will use basic types only." % index
            )

        if "file_path" not in item:
            item["file_path"] = str(self.data_file)
        if "id" not in item:
            item["id"] = str(index)
        if self._transform:
            item = self._transform(item)
        return item


class PDBDataset(Dataset):
    """
    Creates a dataset from a list of PDB files.

    :param file_list: path to LMDB file containing dataset
    :type file_list: list[Union[str, Path]]
    :param transform: transformation function for data augmentation, defaults to None
    :type transform: function, optional
    """

    def __init__(self, file_list, transform=None, store_file_path=True):
        """constructor"""
        self._file_list = [Path(x).absolute() for x in file_list]
        self._num_examples = len(self._file_list)
        self._transform = transform
        self._store_file_path = store_file_path

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)

        file_path = self._file_list[index]

        item = {"atoms": fo.bp_to_df(fo.read_any(file_path)), "id": file_path.name}
        if self._store_file_path:
            item["file_path"] = str(file_path)
        if self._transform:
            item = self._transform(item)
        return item


class SDFDataset(Dataset):
    """
    Creates a dataset from directory of SDF files.

    :param file_list: list containing paths to SDF files. Assumes one structure per file.
    :type file_list: list[Union[str, Path]]
    :param transform: transformation function for data augmentation, defaults to None
    :type transform: function, optional
    :param read_bonds: flag for whether to process bond information from SDF, defaults to False
    :type read_bonds: bool, optional
    """

    def __init__(self, file_list, transform=None, read_bonds=False, add_Hs=False):
        """constructor"""
        self._file_list = [Path(x) for x in file_list]
        self._num_examples = len(self._file_list)
        self._transform = transform
        self._read_bonds = read_bonds
        self._add_Hs = add_Hs

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)
        # Read biopython structure
        file_path = self._file_list[index]
        structure = fo.read_sdf(
            str(file_path), sanitize=True, add_hs=self._add_Hs, remove_hs=False
        )
        # assemble the item (no bonds)
        item = {
            "atoms": fo.bp_to_df(structure),
            "id": structure.id,
            "file_path": str(file_path),
        }
        # Add bonds if included
        if self._read_bonds:
            mol = fo.read_sdf_to_mol(
                str(file_path), sanitize=False, add_hs=False, remove_hs=False
            )
            bonds_df = fo.get_bonds_list_from_mol(mol[0])
            item["bonds"] = bonds_df
        if self._transform:
            item = self._transform(item)
        return item


def serialize(x, serialization_format):
    """
    Serializes dataset `x` in format given by `serialization_format` (pkl, json, msgpack).
    """
    if serialization_format == "pkl":
        # Pickle
        # Memory efficient but brittle across languages/python versions.
        return pkl.dumps(x)
    elif serialization_format == "json":
        # JSON
        # Takes more memory, but widely supported.
        serialized = json.dumps(
            x,
            default=lambda df: json.loads(
                df.to_json(orient="split", double_precision=6)
            ),
        ).encode()
    elif serialization_format == "msgpack":
        # msgpack
        # A bit more memory efficient than json, a bit less supported.
        serialized = msgpack.packb(x, default=lambda df: df.to_dict(orient="split"))
    else:
        raise RuntimeError("Invalid serialization format")
    return serialized


def deserialize(x, serialization_format):
    """
    Deserializes dataset `x` assuming format given by `serialization_format` (pkl, json, msgpack).
    """
    if serialization_format == "pkl":
        return pkl.loads(x)
    elif serialization_format == "json":
        serialized = json.loads(x)
    elif serialization_format == "msgpack":
        serialized = msgpack.unpackb(x)
    else:
        raise RuntimeError("Invalid serialization format")
    return serialized


class LBADataset(Dataset):
    def __init__(self, input_file_path, pdbcodes, ligcodes, transform=None):
        self._rna_dataset = None
        self._pocket_dataset = None
        self._ligand_dataset = None
        self._load_datasets(input_file_path, pdbcodes, ligcodes)
        self._num_examples = len(self._rna_dataset)
        self._transform = transform

    def _load_datasets(self, input_file_path, pdbcodes, ligcodes):
        rna_list = []
        pocket_list = []
        ligand_list = []
        for ligcode in ligcodes:
            pdbcode = ligcode.split("/")[-1][:4]
            ligcode = ligcode.split("/")[-1][5:-11]
            rna_path = os.path.join(input_file_path, f"{pdbcode}_rna.cif")
            pocket_path = os.path.join(
                input_file_path, f"{pdbcode}_{ligcode}_pocket.cif"
            )
            ligand_path = os.path.join(
                input_file_path, f"{pdbcode}_{ligcode}_ligand.sdf"
            )
            if (
                os.path.exists(rna_path)
                and os.path.exists(pocket_path)
                and os.path.exists(ligand_path)
            ):
                rna_list.append(rna_path)
                pocket_list.append(pocket_path)
                ligand_list.append(ligand_path)

        self._rna_dataset = load_dataset(
            rna_list, "pdb", transform=SequenceReader(input_file_path)
        )
        self._pocket_dataset = load_dataset(pocket_list, "pdb", transform=None)
        self._ligand_dataset = load_dataset(
            ligand_list,
            "sdf",
            include_bonds=True,
            add_Hs=False,
            transform=SmilesReader(),
        )

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)

        protein = self._rna_dataset[index]
        pocket = self._pocket_dataset[index]
        ligand = self._ligand_dataset[index]
        pdbcode = get_pdb_code(protein["id"])

        item = {
            "atoms_protein": protein["atoms"],
            "atoms_pocket": pocket["atoms"],
            "atoms_ligand": ligand["atoms"],
            "bonds": ligand["bonds"],
            "id": pdbcode,
            "seq": protein["seq"],
            "smiles": ligand["smiles"],
        }
        if self._transform:
            item = self._transform(item)
        return item


def load_dataset(
    file_list, filetype, transform=None, include_bonds=False, add_Hs=False
):
    if filetype == "lmdb":
        dataset = LMDBDataset(file_list, transform=transform)
    elif (filetype == "pdb") or (filetype == "pdb.gz"):
        dataset = PDBDataset(file_list, transform=transform)
    elif filetype == "sdf":
        dataset = SDFDataset(
            file_list, transform=transform, read_bonds=include_bonds, add_Hs=add_Hs
        )
    else:
        raise RuntimeError(f"Unrecognized filetype {filetype}.")
    return dataset


def make_lmdb_dataset(dataset, output_lmdb, serialization_format="json"):
    num_examples = len(dataset)
    env = lmdb.open(str(output_lmdb), map_size=int(1e13))

    with env.begin(write=True) as txn:
        try:
            id_to_idx = {}
            i = 0
            for x in tqdm.tqdm(dataset, total=num_examples):
                if x["id"] == "3mxh":
                    continue
                x["types"] = {key: str(type(val)) for key, val in x.items()}
                x["types"]["types"] = str(type(x["types"]))
                buf = io.BytesIO()
                with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as f:
                    f.write(serialize(x, serialization_format))
                compressed = buf.getvalue()
                result = txn.put(str(i).encode(), compressed, overwrite=False)
                if not result:
                    raise RuntimeError(
                        f"LMDB entry {i} in {str(output_lmdb)} already exists"
                    )
                id_to_idx[x["id"]] = i
                i += 1
        finally:
            txn.put(b"num_examples", str(i).encode())
            txn.put(b"serialization_format", serialization_format.encode())
            txn.put(b"id_to_idx", serialize(id_to_idx, serialization_format))


def split_lmdb_dataset(lmdb_path, train_txt, val_txt, test_txt, split_dir):
    lmdb_ds = load_dataset(lmdb_path, "lmdb")

    def _write_split_indices(split_txt, lmdb_ds, output_txt):
        with open(split_txt, "r") as f:
            split_set = set([x.strip() for x in f.readlines()])
        split_ids = list(filter(lambda id: id in split_set, lmdb_ds.ids()))
        split_indices = lmdb_ds.ids_to_indices(split_ids)
        with open(output_txt, "w") as f:
            f.write(str("\n".join([str(i) for i in split_indices])))
        return split_indices

    os.makedirs(os.path.join(split_dir, "indices"), exist_ok=True)
    os.makedirs(os.path.join(split_dir, "data"), exist_ok=True)

    indices_train = _write_split_indices(
        train_txt, lmdb_ds, os.path.join(split_dir, "indices/train_indices.txt")
    )
    indices_val = _write_split_indices(
        val_txt, lmdb_ds, os.path.join(split_dir, "indices/val_indices.txt")
    )
    indices_test = _write_split_indices(
        test_txt, lmdb_ds, os.path.join(split_dir, "indices/test_indices.txt")
    )

    train_dataset = torch.utils.data.Subset(lmdb_ds, indices_train)
    val_dataset = torch.utils.data.Subset(lmdb_ds, indices_val)
    test_dataset = torch.utils.data.Subset(lmdb_ds, indices_test)

    make_lmdb_dataset(train_dataset, os.path.join(split_dir, "data/train"))
    make_lmdb_dataset(val_dataset, os.path.join(split_dir, "data/val"))
    make_lmdb_dataset(test_dataset, os.path.join(split_dir, "data/test"))


def main(
    input_file_path,
    output_root,
    score_path,
    train_txt=None,
    val_txt=None,
    test_txt=None,
    split=False,
):
    lmdb_path = os.path.join(output_root, "data")
    os.makedirs(lmdb_path, exist_ok=True)

    # Check if LMDB dataset already exists
    lmdb_file = os.path.join(lmdb_path, "data.mdb")
    if not os.path.exists(lmdb_file):
        scores = Scores(score_path) if score_path else None
        pdbcodes = find_files(input_file_path, "cif")
        ligcodes = find_files(input_file_path, "sdf")

        dataset = LBADataset(input_file_path, pdbcodes, ligcodes, transform=scores)
        make_lmdb_dataset(dataset, lmdb_path)
    else:
        print(f"LMDB dataset found at {lmdb_file} Skipping LMDB creation.")

    if split:
        if not (train_txt, val_txt, test_txt):
            raise ValueError(
                "To use the split option, please provide train_txt, val_txt, and test_txt files containing PDB IDs."
            )
        split_dir = os.path.join(output_root, "split")
        split_lmdb_dataset(lmdb_path, train_txt, val_txt, test_txt, split_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process LMDB datasets.")
    parser.add_argument(
        "--input_file_path",
        required=True,
        help="Path to the input directory containing PDB files.",
    )
    parser.add_argument(
        "--output_root", required=True, help="Root directory for output LMDB files."
    )
    parser.add_argument("--score_path", required=True, help="Path to the score file.")
    parser.add_argument(
        "--train_txt", help="Path to the train PDB ID list (required if split is True)."
    )
    parser.add_argument(
        "--val_txt",
        help="Path to the validation PDB ID list (required if split is True).",
    )
    parser.add_argument(
        "--test_txt", help="Path to the test PDB ID list (required if split is True)."
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Flag to indicate if the dataset should be split.",
    )

    args = parser.parse_args()
    main(
        args.input_file_path,
        args.output_root,
        args.score_path,
        args.train_txt,
        args.val_txt,
        args.test_txt,
        args.split,
    )

# python prepare_lmdb.py --input_file_path './output_dir/' --output_root './output_mdb_/' --score_path './pdbbind_NL_cleaned.csv'
# python prepare_lmdb.py --input_file_path './pdbbind_output_dir/' --output_root './pdbbind_output_mdb/' --score_path '.pdbbind_rna_labels.csv' --train_txt './train_list.txt' --val_txt './val_list.txt' --test_txt './test_list.txt'  --split
