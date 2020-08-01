import re
from copy import deepcopy

from rdkit.Chem import AllChem as Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import Descriptors as Descriptors_
from rdkit.Chem import rdMolDescriptors as Descriptors
from rdkit.Chem import BRICS

from fingerprint import BasicFingerprint
from helper import FingerprintNotSetError, draw_to_svg_stream
        
class Entity:
    
    SMARTS_mols = {}
    occurrence = None
    
    def __init__(self, smiles, id_):
        self.smiles = smiles
        self.id_ = id_
        self._size = 0
        self._mol = None
        
    def basic_fingerprint(self, fp_type='all'):
        try:
            return self.fp.get_fp(fp_type)
        except AttributeError:
            self.base_fp = BasicFingerprint(self.get_rdk_mol())
            fps = self.base_fp.get_fp(fp_type)
        return fps
    
    def get_rdk_mol(self):
        if self._mol:
            return self._mol
        self._mol = Chem.MolFromSmiles(self.smiles)
        return self._mol
    
    def remove_stereo(self, smiles):
        new = re.sub('\[(\w+)@+H*\]',r'\1',smiles)
        return new
    
    def remove_link_atoms(self, smiles):
        new = re.sub('\[[0-9]+\*\]','C',smiles)
        #new = re.sub('\(\)','',new)
        return new
    
    def shrink_alkyl_chains(self, mol):
        
       alkyl_SMARTS = '[C;R0;D1;$(C-[#6,#7,#8;-0])]'
       try:
           patt = self.SMARTS_mols[alkyl_SMARTS]
       except KeyError:
           patt = Chem.MolFromSmarts(alkyl_SMARTS)
           self.SMARTS_mols[alkyl_SMARTS] = patt
       matches = mol.GetSubstructMatches(patt)
       if not matches:
           return mol
       mol = Chem.RWMol(mol)
       for atom in matches:
           mol.ReplaceAtom(atom[0], Chem.Atom(1))
       smi = Chem.MolToSmiles(mol)
       mol = Chem.MolFromSmiles(smi)
       mol = self.shrink_alkyl_chains(mol)
       return mol  
    
    def pattern_count(self, pattern):
        matches = self.get_rdk_mol().GetSubstructMatches(pattern)
        return len(matches)
    
    def has_pattern(self, pattern):
        return self.get_rdk_mol().HasSubstructMatch(pattern)
    
    def get_size(self):
        if self._size:
            return self._size
        self._size = Descriptors_.HeavyAtomCount(self.get_rdk_mol())
        return self._size
    
    def set_id(self, id_):
        self.id_ = id_
        
    def get_id(self):
        return self.id_
    
    def _repr_svg_(self):
        return draw_to_svg_stream(self.get_rdk_mol())
    
        
class Fragment(Entity):

    _parent_group = None
    
    def __init__(self, smiles, id_):
        super().__init__(smiles, id_)
        self._parent_mols = []
        self._group = None
        
    def _remove_redundant(self):
        new_smiles = self.remove_link_atoms(self.smiles)
        new_smiles = self.remove_stereo(new_smiles)
        new_mol = Chem.MolFromSmiles(new_smiles)
        new_mol = self.shrink_alkyl_chains(new_mol)
        new_smiles = Chem.MolToSmiles(new_mol)
        return new_smiles
    
    def get_direct_subs(self):
        benzene_SMARTS = 'c1ccccc1'
        try:
            patt = self.SMARTS_mols[benzene_SMARTS]
        except KeyError:
            patt = Chem.MolFromSmarts(benzene_SMARTS)
            self.SMARTS_mols[benzene_SMARTS] = patt
        core = self.get_core()
        core_mol = Chem.MolFromSmiles(core)
        if not core_mol:
            print('no_core')
            return []
        if Descriptors.CalcNumRings(core_mol) != 1:
            return []
        if not core_mol.HasSubstructMatch(patt):
            return []
        subs = Chem.ReplaceCore(core_mol, patt)
        smiles = Chem.MolToSmiles(subs)
        smiles = re.sub('\[[0-9]+\*\]', '', smiles)
        res = smiles.split('.')
        try:
            res.remove('')
        except ValueError:
            pass
        return res
    
    def remove_subs(self, direct_subs):
        core_mol = self.get_core(as_mol=True)
        for sub, patt in direct_subs.items():
            rm = Chem.DeleteSubstructs(core_mol, patt)
            rm_smi = Chem.MolToSmiles(rm)
            if '.' in rm_smi:
                continue
            core_mol = Chem.MolFromSmiles(rm_smi)
        can_smi = Chem.MolToSmiles(core_mol)
        return can_smi
    
    def get_core(self, as_mol=False):
        try:
            core = self._core
        except AttributeError:
            self.set_core()
            core = self._core
        if as_mol:
            return Chem.MolFromSmiles(core)
        return core
    
    def set_core(self):
        self._core = self._remove_redundant()
    
    def set_group(self, group_id):
        self._group = group_id
        
    def get_group(self):
        return self._group


class Group(Entity):
    
    def __init__(self, smiles, id_):
        super().__init__(smiles, id_)
        self._cluster = None
        self.is_taa = False
    
    
    def get_subs_from_core(self, core):
        core = self.SMARTS_mols[core]
        subs = Chem.ReplaceCore(self.get_rdk_mol(), core)
        smiles = Chem.MolToSmiles(subs)
        smiles = re.sub('\[[0-9]+\*\]', '', smiles)
        res = smiles.split('.')
        try:
            res.remove('')
        except ValueError:
            pass
        return res
    
    def remove_subs(self, sub_mols_dict):
        subs = deepcopy(self._direct_subs)
        mol = self.get_rdk_mol()
        if not subs:
            return self.smiles
        for sub in subs:
            patt = sub_mols_dict[sub]
            rm = Chem.DeleteSubstructs(mol, patt)
            rm_smi = Chem.MolToSmiles(rm)
            if '.' in rm_smi:
                self._direct_subs.remove(sub)
                continue
            mol = Chem.MolFromSmiles(rm_smi)
        can_smi = Chem.MolToSmiles(mol)
        return can_smi

    def set_cluster(self, cluster):
    
        self._cluster = cluster
        
    def get_cluster(self):
        
        return self._cluster
        
    
class Molecule(Entity):
    
    def __init__(self, smiles, id_, mol_rdk=None):
        super().__init__(smiles, id_)
        self.lambda_max = 0
        self.strength_max = 0
        if mol_rdk:
            self._mol = mol_rdk

    def fragment(self):
        leaf_frags = []
        if self.get_rdk_mol():
            leaf_frags = BRICS.BRICSDecompose(self.get_rdk_mol(),
                                              keepNonLeafNodes=False)
        return leaf_frags
        
    def set_conjugation(self, conj_bonds):
        self._conjugation = conj_bonds
        
    def get_conjugation(self):
        return self._conjugation
    
    def set_abs_fp(self, afp):
        self._afp = afp
        
    def get_abs_fp(self):
        try:
            return self._afp
        except AttributeError:
            raise FingerprintNotSetError()


class Bridge(Entity):
    
    def __init__(self, smiles, id_):
        smiles = self.remove_stereo(smiles)
        super().__init__(smiles, id_)
        

class Substituent(Entity):
    
    def __init__(self, smiles, id_):
        smiles = self.remove_stereo(smiles)
        super().__init__(smiles, id_)