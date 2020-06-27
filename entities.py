import re

from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Descriptors as Descriptors_
from rdkit.Chem import rdMolDescriptors as Descriptors
from rdkit.Chem import BRICS

from rdkit.Chem import MACCSkeys

class Fingerprint:
    
    def __init__(self, mol=None):
        self.fp_map = {"MACCS": self.calc_MACCS, "morgan": self.calc_morgan,
                  "rdk": self.calc_rdk, "feature": self.calc_morgan_feat,
                  "topology": self.calc_topology}
        self._mol = mol
        
    def calc_fingerprints(self, fp_type='all'):
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
        self.fp_dict['MACCS'] = self.MACCS
        
    def calc_morgan(self):
        self.morgan = Chem.GetMorganFingerprintAsBitVect(self._mol, 2, nBits=1024)
        self.fp_dict['morgan'] = self.morgan
        
    def calc_rdk(self):
        self.rdk = Chem.RDKFingerprint(self._mol)
        self.fp_dict['rdk'] = self.rdk
        
    def calc_morgan_feat(self):
        self.morgan_feat = Chem.GetMorganFingerprintAsBitVect(
            self._mol ,2, nBits=1024, useFeatures=True)
        self.fp_dict['morgan_feat'] = self.morgan_feat
        
    def calc_topology(self):
        self.topology = Chem.GetMorganFingerprintAsBitVect(
            self._mol, 2, nBits=1024, invariants=[1]*self._mol.GetNumAtoms())
        self.fp_dict['topology'] = self.topology
     
        
class Entity:
    
    id_ = 0
    SMARTS_mols = {'c1ccccc1': Chem.MolFromSmarts('c1ccccc1')}
    
    def __init__(self, smiles, id_):
        self.smiles = smiles
        self.id_ = id_
        self.size = 0
        self._mol = None
        
    def fingerprint(self, fp_type='all'):
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
        self.size = Descriptors_.HeavyAtomCount(self.get_rdk_mol())
        return self.size
    
    def set_id(self, id_):
        self.id_ = id_
        
    def get_id(self):
        return self.id_
        

class Fragment(Entity):

    occurrence = 1
    is_leaf = False
    
    def __init__(self, smiles, id_):
        super().__init__(smiles, id_)
        self._parent_mols = []
        
    def _remove_redundant(self):
        new_smiles = self._remove_link_atoms(self.smiles)
        new_smiles = self._remove_stereo(new_smiles)
        new_mol = Chem.MolFromSmiles(new_smiles)
        if not new_mol:
            print(new_smiles,self.smiles)
        new_mol = self._shrink_alkyl_chains(new_mol)
        new_smiles = Chem.MolToSmiles(new_mol)
        if new_smiles == 'C(=O)C':
            print('how')
        return new_smiles
    
    def _shrink_alkyl_chains(self, mol):
        mol = Chem.RWMol(mol)
        patt = Chem.MolFromSmarts('[C;R0;D1;$(C-[#6;-0])]')
        match = mol.GetSubstructMatch(patt)
        while match:
            mol.RemoveAtom(match[0])
            smi = Chem.MolToSmiles(mol)
            mol = Chem.MolFromSmiles(smi)
            if not mol:
                print(self.id_)
            mol = Chem.RWMol(mol)
            match = mol.GetSubstructMatch(patt)
        return mol
        
    def _remove_link_atoms(self, smiles):
        new = re.sub('\[[0-9]+\*\]','C',smiles)
        #new = re.sub('\(\)','',new)
        return new
    
    def _remove_stereo(self, smiles):
        new = re.sub('\[(\w+)@+H\]',r'\1',smiles)
        return new
       
    def _is_isolated_benzene(self):
        mol = self.get_rdk_mol()
        test_mol = self.SMARTS_mols['c1ccccc1']
        if Descriptors.CalcNumRings(mol) != 1:
            return 0
        if mol.HasSubstructMatch(test_mol):
            return 1
        return 0
    
    def get_substituents(self):
        if not self._is_isolated_benzene():
            return 0
        core = self.SMARTS_mols['c1ccccc1']
        subs = Chem.ReplaceCore(self.get_rdk_mol(), core)
        smiles = Chem.MolToSmiles(subs)
        smiles = re.sub('\[[0-9]+\*\]', '', smiles)
        res = smiles.split('.')
        return res
    
    def get_core_structure(self):
        core = self._remove_redundant()
        return core
    
    def add_parent(self, mol):
        self._parent_mols.append(mol)
        
    def get_parents(self):
        return self._parent_mols
    
    def assign_group(self, group_id):
        self._group = group_id

    def get_group(self):
        try:
            return self._group
        except AttributeError:
            return None
    

class FragmentGroup(Fragment):
    
    def __init__(self, smiles, id_, tier=0):
        super().__init__(smiles, id_)
        self.tier = 0
        

class Molecule(Entity):
    
    def __init__(self, smiles, id_, mol_rdk=None):
        super().__init__(smiles, id_)
        self.lambda_max = 0
        self.strength_max = 0
        if mol_rdk:
            self._mol = mol_rdk

    def fragment(self):
        if self.get_rdk_mol():
            all_frags = BRICS.BRICSDecompose(self.get_rdk_mol(),
                                             keepNonLeafNodes=True)
            leaf_frags = BRICS.BRICSDecompose(self.get_rdk_mol(),
                                              keepNonLeafNodes=False)
        else:
            all_frags = leaf_frags = []
        return (all_frags, leaf_frags)   
        
    def set_conjugation(self, conj_bonds):
        self.conjugation = conj_bonds
        
    
