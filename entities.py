import re
from copy import deepcopy

from IPython.display import display, SVG

from rdkit.Chem import AllChem as Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import Descriptors as Descriptors_
from rdkit.Chem import rdMolDescriptors as Descriptors
from rdkit.Chem import BRICS
from rdkit.Chem.Draw.IPythonConsole import ShowMols
from rdkit.Chem import MACCSkeys

class Fingerprint:
    
    def __init__(self, mol=None):
        self._fp_dict = {}
        self.fp_map = {"MACCS": self._calc_MACCS, "morgan": self._calc_morgan,
                  "rdk": self._calc_rdk, "feature": self._calc_morgan_feat,
                  "topology": self._calc_topology}
        self._mol = mol
    
    def _calc_all(self):
        for method, fp_func in self.fp_map.items():
            if not method in self._fp_dict:
                fp_func()

    def _calc_MACCS(self):
        self.MACCS = MACCSkeys.GenMACCSKeys(self._mol)
        self._fp_dict['MACCS'] = self.MACCS
        
    def _calc_morgan(self):
        self.morgan = Chem.GetMorganFingerprintAsBitVect(self._mol, 
                                                         2, nBits=1024)
        self._fp_dict['morgan'] = self.morgan
        
    def _calc_rdk(self):
        self.rdk = Chem.RDKFingerprint(self._mol)
        self._fp_dict['rdk'] = self.rdk
        
    def _calc_morgan_feat(self):
        self.morgan_feat = Chem.GetMorganFingerprintAsBitVect(
            self._mol ,2, nBits=1024, useFeatures=True)
        self._fp_dict['morgan_feat'] = self.morgan_feat
        
    def _calc_topology(self):
        self.topology = Chem.GetMorganFingerprintAsBitVect(
            self._mol, 2, nBits=1024, 
            invariants=[1]*self._mol.GetNumAtoms())
        self._fp_dict['topology'] = self.topology
        
    def get_fp(self, fp_type):
        if fp_type == "all":
            self._calc_all()
            return self._fp_dict
        try:
            return self._fp_dict[fp_type]
        except KeyError:
            fp_func = self.fp_map[fp_type]
            fp_func()
        return self._fp_dict[fp_type]
        
        
class Entity:
    
    SMARTS_mols = {}
    
    def __init__(self, smiles, id_):
        self.smiles = smiles
        self.id_ = id_
        self.size = 0
        self._mol = None
        
    def fingerprint(self, fp_type='all'):
        try:
            return self.fp.get_fp(fp_type)
        except AttributeError:
            self.fp = Fingerprint(self.get_rdk_mol())
            fps = self.fp.get_fp(fp_type)
        return fps
    
    def get_rdk_mol(self):
        if self._mol:
            return self._mol
        self._mol = Chem.MolFromSmiles(self.smiles)
        return self._mol
    
    def pattern_count(self, pattern):
        matches = self.get_rdk_mol().GetSubstructMatches(pattern)
        return len(matches)
    
    def has_pattern(self, pattern):
        return self.get_rdk_mol().HasSubstructMatch(pattern)
    
    def draw_to_svg_stream(self):
        mol = self.get_rdk_mol()
        d2svg = rdMolDraw2D.MolDraw2DSVG(300,300)
        d2svg.DrawMolecule(mol)
        d2svg.FinishDrawing()
        return d2svg.GetDrawingText()
    
    def get_size(self):
        if self.size:
            return self.size
        self.size = Descriptors_.HeavyAtomCount(self.get_rdk_mol())
        return self.size
    
    def set_id(self, id_):
        self.id_ = id_
        
    def get_id(self):
        return self.id_
    
    def _repr_svg_(self):
        return self.draw_to_svg_stream()


        
class Fragment(Entity):

    occurrence = None
    is_leaf = False
    _parent_group = None
    
    def __init__(self, smiles, id_):
        super().__init__(smiles, id_)
        self._parent_mols = []
        
    def _remove_redundant(self):
        new_smiles = self._remove_link_atoms(self.smiles)
        new_smiles = self._remove_stereo(new_smiles)
        new_mol = Chem.MolFromSmiles(new_smiles)
        new_mol = self._shrink_alkyl_chains(new_mol)
        new_smiles = Chem.MolToSmiles(new_mol)
        return new_smiles
    
    def _shrink_alkyl_chains(self, mol):
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
        mol = self._shrink_alkyl_chains(mol)
        return mol
        
    def _remove_link_atoms(self, smiles):
        new = re.sub('\[[0-9]+\*\]','C',smiles)
        #new = re.sub('\(\)','',new)
        return new
    
    def _remove_stereo(self, smiles):
        new = re.sub('\[(\w+)@+H*\]',r'\1',smiles)
        return new
    
    def get_core_structure(self):
        core = self._remove_redundant()
        return core
    
    def add_parent_mol(self, mol):
        self._parent_mols.append(mol)
        
    def set_parent_mols(self, mols):
        self._parent_mols = mols
        
    def get_parent_mols(self):
        return self._parent_mols
    
    def set_parent_group(self, group_id):
        self._parent_group = group_id
        
    def get_parent_group(self):
        return self._parent_group


class FragmentGroup(Fragment):
    
    def __init__(self, smiles, id_, tier=0):
        super().__init__(smiles, id_)
        self._tier = tier
        self._direct_subs = []
        self._leaf_frags= []
        self._cluster = None
        self.is_taa = False
    
    def get_tier(self):
        return self._tier
    
    def get_substituents(self):
        return self._direct_subs
        
    def set_substituents(self, direct_subs):
        
        subs = []
        for smi, patt in direct_subs.items():
            if not self.get_rdk_mol().HasSubstructMatch(patt):
                continue
            subs.append(smi)
        self._direct_subs = subs
 
    def is_isolated_benzene(self):
        mol = self.get_rdk_mol()
        benzene_SMARTS = 'c1ccccc1'
        try:
            patt = self.SMARTS_mols[benzene_SMARTS]
        except KeyError:
            patt = Chem.MolFromSmarts(benzene_SMARTS)
            self.SMARTS_mols[benzene_SMARTS] = patt
        if Descriptors.CalcNumRings(mol) != 1:
            return 0
        if mol.HasSubstructMatch(patt):
            return 1
        return 0
    
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
    
    def add_leaf_frag(self, leaf_id):
        self._leaf_frags.append(leaf_id)
        
    def get_leaf_frags(self):
        return self._leaf_frags
    
    def check_is_azo(self):
        azo_SMARTS = '[#6]-[N;R0]=[N;R0]-[#6]'
        try:
            patt = self.SMARTS_mols[azo_SMARTS]
        except KeyError:
            patt = Chem.MolFromSmarts(azo_SMARTS)
            self.SMARTS_mols[azo_SMARTS] = patt
        if self.get_rdk_mol().HasSubstructMatch(patt):
            return True
        return False
    
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
        if self.get_rdk_mol():
            all_frags = BRICS.BRICSDecompose(self.get_rdk_mol(),
                                             keepNonLeafNodes=True)
            leaf_frags = BRICS.BRICSDecompose(self.get_rdk_mol(),
                                              keepNonLeafNodes=False)
        else:
            all_frags = leaf_frags = []
        return (all_frags, leaf_frags)   
        
    def set_conjugation(self, conj_bonds):
        self._conjugation = conj_bonds
        
    def get_conjugation(self):
        return self._conjugation


class Bridge(Fragment):
    
    def __init__(self, smiles, id_):
        super().__init__(smiles, id_)
        

class Substituent(Fragment):
    
    def __init__(self, smiles, id_):
        super().__init__(smiles, id_)