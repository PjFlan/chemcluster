from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import BRICS

from rdkit.Chem import MACCSkeys

class Fingerprint():
    
    def __init__(self,mol=None):
        self.fp_map = {"MACCS":self.calc_MACCS,"morgan":self.calc_morgan,
                  "rdk":self.calc_rdk,"feature":self.calc_morgan_feat}
        self._mol = mol
        
    def calc_fingerprints(self,fp_type='all'):
        self.fp_dict = {}
        if fp_type == 'all':
            for fp_func in self.fp_map.values():
                fp_func()
        else:
            fp_func = self.fp_map[fp_type]
            fp_func()
        return self.fp_dict
    
    def calc_MACCS(self):
        self.MACCS = MACCSkeys.GenMACCSKeys(self._mol)
        print(list(self.MACCS.GetOnBits()))
        self.fp_dict['MACCS'] = self.MACCS
        
    def calc_morgan(self):
        self.morgan = Chem.GetMorganFingerprintAsBitVect(self._mol,2,nBits=1024)
        self.fp_dict['morgan'] = self.morgan
        
    def calc_rdk(self):
        self.rdk = Chem.RDKFingerprint(self._mol)
        self.fp_dict['rdk'] = self.rdk
        
    def calc_morgan_feat(self):
        self.morgan_feat = Chem.GetMorganFingerprintAsBitVect(self._mol,2,nBits=1024,useFeatures=True)
        self.fp_dict['morgan_feat'] = self.morgan_feat
        
class Structure:
    
    def __init__(self,smiles):
        self.smiles = smiles
        self.size = 0
        self._mol = None
        
    def fingerprint(self,fp_type = 'all'):
        self.fingerprint = Fingerprint(self.get_rdk_mol())
        fps = self.fingerprint.calc_fingerprints(fp_type)
        return fps
    
    def get_rdk_mol(self):
        if self._mol:
            return self._mol
        self._mol = Chem.MolFromSmiles(self.smiles)
        return self._mol
    
    def get_size(self):
        if self.size:
            return self.size
        self.size = Descriptors.HeavyAtomCount(self.get_rdk_mol())
        return self.size

class Fragment(Structure):
    is_leaf = False
    occurrence = 1
    
    def __init__(self,smiles,mol):
        super().__init__(smiles)
        self.parent_mol = mol
    
class Molecule(Structure):
    
    def __init__(self,smiles,mol_rdk=None):
        super().__init__(smiles)
        self.lambda_max = 0
        self.strength_max = 0
        if mol_rdk:
            self._mol = mol_rdk

    def fragment(self):
        if self.get_rdk_mol():
            all_frags = BRICS.BRICSDecompose(self.get_rdk_mol(),keepNonLeafNodes=True)
            leaf_frags = BRICS.BRICSDecompose(self.get_rdk_mol(),keepNonLeafNodes=False)
        else:
            all_frags = leaf_frags = []
        return (all_frags,leaf_frags)
        
    def set_conjugation(self,conj_bonds):
        self.conjugation = conj_bonds
        
    
    
