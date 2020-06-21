import os.path
import math
from collections import defaultdict
import re
import pandas as pd

from rdkit import Chem
from pymongo import MongoClient
from rdkit.Chem import MACCSkeys

from entities import Molecule,Fragment
from helper import MyConfig,MyLogger,MyFileHandler,MyConfigParamError

class Mongo:

    
    def __init__(self):
        self._configure()
        
    def _configure(self):
        self._logger = MyLogger().get_child(type(self).__name__)
        self._fh = MyFileHandler()
        self._config = MyConfig()

    def _connect(self,conn_info=None):
        self._client = MongoClient()
        self._db = self._client[conn_info["database"]]
        self._collection = self._db[conn_info["collection"]]
        
    def _close_connection(self):
        self._client.close()
        
class MongoLoad(Mongo):
    
    def __init__(self):
        super().__init__()
        self._conn_info = self._config.get_db_source()

        
    def __enter__(self): 
        self._connect(self._conn_info)
        return self
      
    def __exit__(self, exc_type, exc_value, exc_traceback): 
        self._close_connection()
        
    def resolve_query(self,record_type):
        query_func = 'self.get_' + record_type + '()'
        query_result = eval(query_func)
        return query_result
    
    def get_smiles(self):

        smiles = []
        smiles_file = None
        regen = self._config.get_regen('smiles')
        
        try:
            smiles_file = self._config.get_directory('smiles')
        except MyConfigParamError as e:
            self._logger.warning(e)
            
        if os.path.isfile(smiles_file) and not regen:
            smiles = self._fh.load_from_text(smiles_file)
        else:
            smiles = self._collection.distinct("PRISTINE.SMI")
            if smiles_file:
                self._fh.output_to_text(smiles,smiles_file) 
        smiles_df = pd.Series(smiles,name='smiles')
        return smiles_df
    
    def get_comp_uvvis(self):
        cursor = self._collection.find({},{
            '_id':0,
            'PRISTINE.SMI':1,
            'FILTERED.orca.excited_states.orbital_energy_list':{'$slice':3},
            'FILTERED.orca.excited_states.orbital_energy_list.amplitude':1,
            'FILTERED.orca.excited_states.orbital_energy_list.oscillator_strength':1
        })
        
        lambdas = []
        strengths = []
        smiles = []
        for record in cursor:
            try:
                smi = record['PRISTINE'][0]['SMI']
                comp = record['FILTERED'][0]['orca'][0]['excited_states']['orbital_energy_list']
                lam_1,lam_2,lam_3 = comp[0]['amplitude'],comp[1]['amplitude'],comp[2]['amplitude']
                osc_1,osc_2,osc_3 = comp[0]['oscillator_strength'],comp[1]['oscillator_strength'],comp[2]['oscillator_strength']
            except KeyError:
                continue
            smiles.append(smi)
            osc = [osc_1,osc_2,osc_3]
            lam = [lam_1,lam_2,lam_3]
            max_idx = osc.index(max(osc))
            lambdas.append(lam[max_idx])
            strengths.append(osc[max_idx])
            
        df = pd.DataFrame({'smiles':smiles,'lambda':lambdas,'strength':strengths})
        df = df.set_index('smiles')
        return df
    
class MoleculeData():
    
    def __init__(self):
        self._configure()
        
    def _configure(self):
        self.molecules = pd.Series([],dtype=object)
        self.fragments = pd.Series([],dtype=object)
        self.comp_flag = False
        
        self._logger = MyLogger().get_child(type(self).__name__)
        self._fh = MyFileHandler()
        self._config = MyConfig()
        self._get_patterns()
        
    def _get_patterns(self):
        file = self._config.get_directory("patterns")
        patterns = self._fh.load_from_text(file,delim='|')
        self.PATTERNS = {p[0]:p[1] for p in patterns}
    
    def _set_smiles_id_map(self,reset=False):
        smiles_map_file = self._config.get_directory('smiles_ids')
        regen = self._config.get_regen('smiles_ids')
        if os.path.isfile(smiles_map_file) and not regen:
            smiles_id_map = self._fh.load_from_json(smiles_map_file)
        else:
            unique_smi = self.fragments.index.tolist()
            smiles_id_map = {smi:idx for idx,smi in enumerate(unique_smi)}
        id_smiles_map = {v: k for k, v in smiles_id_map.items()}
        self._smi_id_map = smiles_id_map
        self._id_smi_map = id_smiles_map
        if regen or not os.path.isfile(smiles_map_file):
            self._fh.output_to_json(smiles_id_map,smiles_map_file)
            
    def _get_frags_file(self):
        try:
            frags_dir = self._config.get_directory('fragments')
            self.frags_file = os.path.join(frags_dir,'frags.json')
    
        except MyConfigParamError as e:
            self._logger.warning(e)
            self.frags_file = None
            
    def _parse_frags_file(self):
        frags_dict = {}
        
        if os.path.isfile(self.frags_file):
            frags_dict = self._fh.load_from_json(self.frags_file)
    
        return frags_dict

    def _save_frags_to_file(self):
        frags_dict = defaultdict(lambda : {'all_frags':[],'leaf_frags':[]})
        for frag in self.fragments.tolist():
            frags_dict[frag.parent_mol.smiles]['all_frags'].append(frag.smiles)
            if frag.is_leaf:
                frags_dict[frag.parent_mol.smiles]['leaf_frags'].append(frag.smiles)
        self._fh.output_to_json(frags_dict,self.frags_file)
        
    def get_molecules(self,reset=False):
        if not self.molecules.empty and not reset:
            return self.molecules
        with MongoLoad() as mongo:
            smiles = mongo.resolve_query('smiles')
        pickled_file = self._config.get_directory('pickle') +  'molecules.pickle'
        regen = self._config.get_regen('rdk_mols')
        mols_dict = {}
        if os.path.isfile(pickled_file) and not regen:
            mols_dict = self._fh.load_from_pickle(pickled_file)
        mols = []
        for smi in smiles.tolist():
            mol_rdk = mols_dict.get(smi,0)
            if mol_rdk:
                mol = Molecule(smi,mol_rdk)
            else:
                mol = Molecule(smi)
            mols.append(mol)
    
        if regen:
            for mol in mols:
                mols_dict[mol.smiles] = mol.get_rdk_mol()
            self._fh.dump_to_pickle(mols_dict,pickled_file)
        self.molecules = pd.Series(mols,index=smiles)
        return self.molecules
    
    def get_fragments(self):
        regen = self._config.get_regen('fragments')
        if not self.fragments.empty and not regen:
            return self.fragments
        
        self._get_frags_file()
        if not self.frags_file:
            regen = True
            
        saved_frags = {}
        if not regen:
            saved_frags = self._parse_frags_file()
                   
        molecules = self.get_molecules()
        frags_dict = {}
        for mol in molecules:
            frags = saved_frags.get(mol.smiles)
            if frags:
                frags = (frags['all_frags'],frags['leaf_frags'])
            else:
                frags = mol.fragment()
            for frag in frags[0]:
                if frag in frags_dict:
                    frags_dict[frag].occurrence += 1
                    continue
                new_frag = Fragment(frag,mol)
                if frag in frags[1]:
                    new_frag.is_leaf = True
                frags_dict[frag] = new_frag
        self.fragments = pd.Series(frags_dict)
        self._set_smiles_id_map()
        if regen and self.frags_file:
            self._save_frags_to_file()
        return self.fragments
    
    def get_id_from_smiles(self,smiles):
        return self._smi_id_map[smiles]

    def get_smiles_from_id(self,smi_id):
        return self._id_smi_map[smi_id]
    
    def get_parent_mols(self,smi_id):
        molecules = self.get_molecules()
        fragments = self.get_fragments()
        smiles = self.get_smiles_from_id(smi_id)
        fragments = fragments[smiles]
        mols_filter = fragments.apply(lambda f: f.parent_mol.smiles).tolist()
        parents = molecules[mols_filter]
        return parents
    
    def get_conjugation(self,molecules=None):
        if not molecules:
            molecules = self.get_molecules()
        conj_file = os.path.join(self._config.get_directory('conjugation'),'conjugation.csv')
        conjugation = self._fh.load_from_text(conj_file)
        zipped = list(zip(*conjugation))
        conjugation = pd.Series(zipped[2],index=zipped[1])

        for m in molecules:
            try:
                conj = int(conjugation.loc[m.smiles])
            except KeyError:
                conj = None
            m.set_conjugation(conj)
            
    def find_mols_with_pattern(self,group):
        if group == 'NoGroup':
            return self.find_mols_no_pattern()
        molecules = self.get_molecules()
        pattern = self.PATTERNS[group]
        pattern = Chem.MolFromSmarts(pattern)
        matches = molecules.apply(lambda x: x if x.get_rdk_mol().HasSubstructMatch(pattern) else None)
        matches = matches.dropna()
        return matches
    
    def find_mols_no_pattern(self):
        molecules = self.get_molecules()
        patterns = [Chem.MolFromSmarts(smarts) for smarts in list(self.PATTERNS.values())]
        mols = []
        for mol in molecules.tolist():
            found = False
            for pattern in patterns:
                if mol.get_rdk_mol().HasSubstructMatch(pattern):
                    found = True
                    break
            if not found:
                mols.append(mol.smiles)
        matches = molecules.filter(mols)
        return matches
    
    def groups_in_combo(self,mol):
        groups = []
        for pattern,smarts in self.PATTERNS.items():
            test = Chem.MolFromSmarts(smarts)
            if mol.get_rdk_mol().HasSubstructMatch(test):
                groups.append(pattern)
        return groups
        
    def clean_mols(self,mols=None):
        molecules = self.get_molecules()
        self.set_comp_data()
        try:
            thresholds = self._config.get_comp_thresholds('molecules')
            f_min = thresholds['f_min']
            lambda_min = thresholds['lambda_min']
            lambda_max = thresholds['lambda_max']
        except MyConfigParamError as e:
            f_min = 0
            lambda_min = 0
            lambda_max = math.inf
            self._logger.warning(e)
            f_min = 0
            lambda_min = 0
            lambda_max = math.inf
        molecules = molecules.apply(lambda x: x if (x.lambda_max and x.lambda_max > lambda_min and x.lambda_max < lambda_max
                                                    and x.strength_max > f_min) else None)
        molecules = molecules.dropna()
        self.molecules = molecules
        return molecules
    
    def set_comp_data(self):
        if self.comp_flag:
            return
        mols = self.get_molecules()
        with MongoLoad() as mongo:
            comp_df = mongo.get_comp_uvvis()
        for m in mols:
            try:
                record = comp_df.loc[m.smiles]
                m.lambda_max = record['lambda']
                m.strength_max = record['strength']
            except KeyError:
                m.lambda_max = m.strength_max = None
        self.comp_flag = True
        
            
class FragmentData(MoleculeData):
    
    def __init__(self):
        self.fragments= None
        super().__init__()
    
    def _filter(self,fragment):
        if not fragment.is_leaf:
                return 0
        raw = re.sub('\[.*?\]','',fragment.smiles)
        raw = re.sub('[()]','',raw)
        if bool(re.match('^[C]+$',raw)):
            return 0
        if bool(re.match('^[c1ccccc1]+$',raw)):
            return 0
        return fragment
            
    def clean_frags(self,frags=None):
        if not frags:
            frags = self.get_fragments()
        clean_frags = frags[frags.apply(self._filter) != 0]
        return clean_frags
    
    def frags_to_mols(self,frag_smiles=None):
        if not frag_smiles:
            frag_smiles = [f.smiles for f in self.fragments]
        mols = [Chem.MolFromSmiles(f) for f in frag_smiles]
        return mols



    
    