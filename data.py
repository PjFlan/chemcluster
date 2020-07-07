import os.path
import math
from collections import defaultdict
import re
import pandas as pd

from rdkit import Chem
from rdkit import DataStructs
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker

from pymongo import MongoClient

from entities import Molecule, Fragment, FragmentGroup
from helper import MyConfig, MyLogger, MyFileHandler, MyConfigParamError

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
                self._fh.output_to_text(smiles, smiles_file) 
        smiles_df = pd.Series(smiles, name='smiles')
        return smiles_df
    
    def get_comp_uvvis(self):
        cursor = self._collection.find({}, {
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
                lam_1, lam_2, lam_3 = comp[0]['amplitude'], \
                    comp[1]['amplitude'], comp[2]['amplitude']
                osc_1, osc_2, osc_3 = comp[0]['oscillator_strength'], \
                    comp[1]['oscillator_strength'], comp[2]['oscillator_strength']
            except KeyError:
                continue
            smiles.append(smi)
            osc = [osc_1, osc_2, osc_3]
            lam = [lam_1, lam_2, lam_3]
            max_idx = osc.index(max(osc))
            lambdas.append(lam[max_idx])
            strengths.append(osc[max_idx])
            
        df = pd.DataFrame({'smiles': smiles,'lambda': lambdas,'strength': strengths})
        df = df.set_index('smiles')
        return df
    

class EntityData:
        
    def _configure(self):
        self._logger = MyLogger().get_child(type(self).__name__)
        self._fh = MyFileHandler()
        self._config = MyConfig()
        self._get_patterns()
        
    def _get_patterns(self):
        file = self._config.get_directory("patterns")
        patterns = self._fh.load_from_text(file, delim='|')
        self.PATTERNS = {p[0]: p[1] for p in patterns}
        
    def _create_entity_df(self, entities):
        entity_df = pd.Series(entities)
        index = entity_df.apply(lambda e: e.id_)
        entity_df.index = index
        return entity_df
    
    
class MoleculeData(EntityData):
    
    def __init__(self):
        self._configure()
        self._set_up()
        
    def _set_up(self):
        self.molecules = pd.Series([], dtype=object)
        self.comp_flag = False
        
    def _comp_filter(self, mol, f_min, lambda_max, lambda_min):
        if not mol.lambda_max:
            return None
        if (lambda_min < mol.lambda_max < lambda_max) \
        and mol.strength_max > f_min:
            return mol
        return None
            
    def get_molecule(self, id_):
        molecules = self.get_molecules()
        mol = molecules[id_]
        return mol
    
    def get_molecules(self, reset=False):
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
        id_ = 0
        for smi in smiles.tolist():
            mol_rdk = mols_dict.get(smi, 0)
            if mol_rdk:
                mol = Molecule(smi, id_, mol_rdk)
            else:
                mol = Molecule(smi, id_)
            mols.append(mol)
            id_ += 1
        if regen:
            for mol in mols:
                mols_dict[mol.smiles] = mol.get_rdk_mol()
            self._fh.dump_to_pickle(mols_dict, pickled_file)

        self.molecules = self._create_entity_df(mols)
        return self.molecules
    
    def get_conjugation(self, molecules=None):
        if not molecules:
            molecules = self.get_molecules()
        conj_file = os.path.join(self._config.get_directory('conjugation'),'conjugation.csv')
        conjugation = self._fh.load_from_text(conj_file)
        zipped = list(zip(*conjugation))
        conjugation = pd.Series(zipped[2], index=zipped[1])

        for m in molecules:
            try:
                conj = int(conjugation.loc[m.smiles])
            except KeyError:
                conj = None
            m.set_conjugation(conj)
            
    def find_mols_with_pattern(self, group):
        if group == 'NoGroup':
            return self.find_mols_no_pattern()
        molecules = self.get_molecules()
        pattern = self.PATTERNS[group]
        pattern = Chem.MolFromSmarts(pattern)
        matches = molecules.apply(lambda x: x if 
                                  x.get_rdk_mol().HasSubstructMatch(pattern) else None)
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
    
    def groups_in_combo(self, mol):
        groups = []
        for pattern, smarts in self.PATTERNS.items():
            test = Chem.MolFromSmarts(smarts)
            if mol.get_rdk_mol().HasSubstructMatch(test):
                groups.append(pattern)
        return groups
        
    def clean_mols(self, mols=None):
        try:
            return self.clean_molecules
        except AttributeError:
            pass
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
        molecules = molecules.apply(self._comp_filter, args=(f_min, lambda_max, lambda_min))
        molecules = molecules.dropna()
        self.clean_molecules = molecules
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
            except KeyError:
                m.lambda_max = m.strength_max = None
                continue
            m.lambda_max = record['lambda']
            m.strength_max = record['strength']
        self.comp_flag = True
        
            
class FragmentData(EntityData):
    
    SMARTS_mols = {smarts: Chem.MolFromSmarts(smarts) 
                   for smarts in ['[#6;!X4]']}
    def __init__(self):
        self._configure()
        self._set_up()
        
    def _set_up(self):
        self.fragments = pd.Series([], dtype=object)
        self.mol_data = MoleculeData()
        
    def _filter(self, fragment):
        if not fragment.is_leaf:
            return 0
        raw = re.sub('\[.*?\]', '', fragment.smiles)
        raw = re.sub('[()]', '', raw)
        if bool(re.match('^[C]+$', raw)):
            return 0
        if bool(re.match('^[c1ccccc1]+$', raw)):
            return 0
        if '.' in raw:
            return 0
        if not fragment.get_rdk_mol().HasSubstructMatch(
                self.SMARTS_mols['[#6;!X4]']):
            return 0
        return fragment
    
    def _create_sub_pattern(self, sub):
        sub = Chem.MolFromSmiles(sub)
        sub = Chem.MolToSmarts(sub)
        sub = re.sub(r'\[(.+?)\]', r'[\1;!R]', sub)
        sub = re.sub(r'\[(.+?)\](?!.+\[(.+?)\])', r'[\1;D1]', sub)
        sub = re.sub(r'\[(.+?)\]', r'[\1;$([\1]-[#6;R])]', sub, count=1)

        return Chem.MolFromSmarts(sub)
    
    def _set_mol_frag_link(self):
        link_table = []
        frags = self.get_fragments()
        link_table = pd.DataFrame([[mol_id,f_id] 
                     for f_id, m_list in frags.apply(
                             lambda x: x.get_parents()).iteritems()
                     for mol_id in m_list], columns=['mol_id','frag_id'])
        self._mol_frag_link = link_table
        return link_table  
    
    def _get_mol_frag_link(self):
        try:
            link_table = self._mol_frag_link
        except AttributeError:
            link_table = self._set_mol_frag_link()
        return link_table
            
    def _get_frags_file(self):
        try:
            frags_dir = self._config.get_directory('fragments')
            self.frags_file = os.path.join(frags_dir, 'frags.json')
    
        except MyConfigParamError as e:
            self._logger.warning(e)
            self.frags_file = None
            
    def _parse_frags_file(self):
        frags_dict = {}
        
        if os.path.isfile(self.frags_file):
            frags_dict = self._fh.load_from_json(self.frags_file)
    
        return frags_dict

    def _save_frags_to_file(self):
        frags_dict = defaultdict(lambda : {'all_frags': [],'leaf_frags': []})
        for frag in self.fragments.tolist():
            for mol in frag.parent_mols:
                frags_dict[mol.smiles]['all_frags'].append(frag.smiles)
                if frag.is_leaf:
                    frags_dict[mol.smiles]['leaf_frags'].append(frag.smiles)
        self._fh.output_to_json(frags_dict, self.frags_file)
        
    def get_fragment(self, id_):
        fragments = self.get_fragments()
        frag = fragments[id_]
        return frag    
    
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
                   
        molecules = self.mol_data.get_molecules()
        frags_dict = {}
        id_ = 0
        for mol in molecules:
            frags = saved_frags.get(mol.smiles)
            if frags:
                frags = (frags['all_frags'], frags['leaf_frags'])
            else:
                frags = mol.fragment()
            for frag in frags[0]:
                if frag in frags_dict:
                    existing = frags_dict[frag]
                    existing.occurrence += 1
                    existing.add_parent(mol.get_id())
                    continue
                new_frag = Fragment(frag, id_)
                new_frag.add_parent(mol.get_id())
                if frag in frags[1]:
                    new_frag.is_leaf = True
                frags_dict[frag] = new_frag
                id_ += 1

        self.fragments = self._create_entity_df(frags_dict)
        self._set_mol_frag_link()
        
        if regen and self.frags_file:
            self._save_frags_to_file()
        return self.fragments
    
    def get_parent_mols(self, id_):
        mols = self.mol_data.get_molecules()
        link_table = self._get_mol_frag_link()
        parent_ids = link_table[id_ == link_table['frag_id']]['mol_id']
        parents = mols.loc[parent_ids.values]
        return parents
           
    def has_comp_data(self, frag):
        parent_mols = self.get_parent_mols(frag.get_id())
        comp_mols = self.mol_data.clean_mols()
        parent_ids = set(parent_mols.index)
        comp_mol_ids = set(comp_mols.index)
        if parent_ids.intersection(comp_mol_ids):
            return 1
        return 0
    
    def clean_frags(self, frags=None):
        if not frags:
            frags = self.get_fragments()
        clean_frags = frags[frags.apply(self._filter) != 0]
        return clean_frags
    
    def frags_to_mols(self, frag_smiles=None):
        if not frag_smiles:
            frag_smiles = self.fragments.apply(lambda f: f.smiles)
        mols = frag_smiles.apply(lambda f: Chem.MolFromSmiles(f))
        return mols


class FragmentGroupData(FragmentData):
    
    def __init__(self):
        super().__init__()
        super()._set_up()

    def _set_up(self):
        self._frag_groups_dict = {}   
        self._tier_link_tables = {}
        
    def _set_frag_group_link(self):
        link_table = []
        frag_groups = self.get_frag_groups(tier=0)
        link_table = pd.DataFrame([[fg_id, frag_id] 
                     for fg_id, f_list in frag_groups.apply(
                             lambda fg: fg.get_leaf_frags()).iteritems()
                     for frag_id in f_list], columns=['group_id','frag_id'])
        self._frag_group_link = link_table
        return link_table
    
    def _set_tier_link(self, parent_tier=1):
        link_table = []
        child_groups = self.get_frag_groups(parent_tier - 1)
        link_table = pd.DataFrame([[child_id, parent_id] 
                     for child_id, parent_id in child_groups.apply(
                             lambda fg: fg.get_parent_group()).iteritems()])
        link_table.columns = ['child_id', 'parent_id']
        self._tier_link_tables[parent_tier] = link_table
        return link_table  
    
    def _get_frag_group_link(self):
        try:
            link_table = self._frag_group_link
        except AttributeError:
            link_table = self._set_frag_group_link()
        return link_table
    
    def _get_tier_link(self, parent_tier=0):
        try:
            link_table = self._tier_link_tables[parent_tier]
        except KeyError:
            link_table = self._set_tier_link(parent_tier)
        return link_table
    
    def _similarity_search(self, group, all_groups):
        pass
    
    def _diversify(self, fps, diverse_groups=100):
        picker = MaxMinPicker()
        nfps = fps.size
        def distij(i, j, fps=fps.tolist()):
            return 1-DataStructs.DiceSimilarity(fps[i],fps[j])
        pick_indices = picker.LazyPick(distij, nfps, diverse_groups, seed=23)
        return list(pick_indices)
    
    def _remove_azo_groups(self, groups):
        is_azo = groups.apply(lambda x: x.check_is_azo())
        return groups[~is_azo]
    
    def _parent_mol_count(self, group):
        parent_mols = self.get_group_parent_mols(group)
        return parent_mols.size
    
    def _get_parent_mol_ids(self, group):
        tier = group.get_tier()
        parent_mol_ids = set()
    
        if tier > 0:
            children = self.get_child_groups(tier, group.get_id())
            for child in children:
                parent_ids = self._get_parent_mol_ids(child)
                parent_mol_ids.update(parent_ids)
        else:
            frags = self.get_leaf_frags(group.get_id())
            for frag in frags:
                parent_ids = self.get_parent_mols(frag.get_id()).index
                parent_mol_ids.update(parent_ids)
        ids = list(parent_mol_ids)
        return ids
    
    def get_frag_group(self, id_, tier=0):
        frag_groups = self.get_frag_groups(tier)
        fg = frag_groups.loc[id_]
        return fg
        
    def get_frag_groups(self, tier=0):
        try:
            return self._frag_groups_dict[tier]
        except KeyError:
            pass
        
        grouping_func = 'self.get_tier_' + str(tier) + '()'
        grouping_result = eval(grouping_func)
        frag_groups = self._create_entity_df(grouping_result)
        self._frag_groups_dict[tier] = frag_groups
        if tier > 0:
            self._set_tier_link(tier)
        else:
            self._set_frag_group_link()
        return frag_groups 
    
    def get_tier_0(self):
        fragments = super().clean_frags()
        groups_dict = {}
        id_ = 0
        for frag in fragments:
            if not self.has_comp_data(frag):
                continue
            core = frag.get_core_structure()
            group = groups_dict.get(core)
            if group:
                group.occurrence += 1
                group.add_leaf_frag(frag.get_id())
                frag.set_parent_group(group.get_id())
                continue
            new_group = FragmentGroup(core, id_, tier=0)
            new_group.add_leaf_frag(frag.get_id())
            groups_dict[core] = new_group
            frag.set_parent_group(new_group.get_id())
            id_ += 1
        return groups_dict
    
    def get_child_groups(self, tier, id_):
        children = self.get_frag_groups(tier - 1)
        link_table = self._get_tier_link(tier)
        child_ids = link_table[id_ == link_table['parent_id']]['child_id']
        children = children.loc[child_ids]
        return children
    
    def get_group_parent_mols(self, group):
        parent_ids = self._get_parent_mol_ids(group)
        mols = self.mol_data.get_molecules()
        return mols[parent_ids]
    
    def get_leaf_frags(self, id_):
        '''
        Return the leaf fragments belonging to the tier 0 group with id = id_

        Parameters
        ----------
        id_ : TYPE
            DESCRIPTION.

        Returns
        -------
        leaf_frags : TYPE
            DESCRIPTION.

        '''
        leaf_frags = super().clean_frags()
        link_table = self._get_frag_group_link()
        leaf_frag_ids = link_table[id_ == link_table['group_id']]['frag_id']
        leaf_frags = leaf_frags.loc[leaf_frag_ids]
        return leaf_frags
    
    def get_common_subs(self):
        try:
            return self._common_subs
        except AttributeError:
            pass
        frags = self.get_frag_groups(tier=0)
        all_subs = set()
        for f in frags:
            if not f.is_isolated_benzene():
                continue
            subs = f.get_subs_from_core('c1ccccc1')
            if subs:
                all_subs.update(subs)
        self._common_subs = {sub: self._create_sub_pattern(sub) for 
                         sub in all_subs}
    
    def get_tier_1(self):
        prev_tier = self.get_frag_groups(tier=0)
        self.get_common_subs()
        self.set_frag_subs()
        groups_dict = {}
        id_ = prev_tier.index[-1] + 1
        for fg in prev_tier:
            core = fg.remove_subs(self._common_subs)
            if not core:
                continue
            group = groups_dict.get(core)
            if group:
                group.occurrence += 1
                fg.set_parent_group(group.get_id())
                continue
            new_group = FragmentGroup(core, id_, tier=1)
            fg.set_parent_group(new_group.get_id())
            groups_dict[core] = new_group
            id_ += 1
        return groups_dict
    
    def set_frag_subs(self):
        frags = self.get_frag_groups(tier=0)
        frag_subs = frags.apply(lambda fg: fg.set_substituents(self._common_subs))
        frag_subs = frag_subs.dropna()
        return frag_subs
    
    def get_tier_2(self):
        prev_tier = self.get_frag_groups(tier=1)
        groups_dict = {}
        id_ = prev_tier.index[-1] + 1
        no_azo_groups = self._remove_azo_groups(prev_tier)
        azo_smi = 'C-N=N-C'
        azo_group = FragmentGroup(azo_smi, id_, tier=2)
        azo_group.occurrence = prev_tier.size - no_azo_groups.size
        groups_dict[azo_smi] = azo_group
        id_ += 1
        
        mols_per_group = no_azo_groups.apply(self._parent_mol_count)
        indices = mols_per_group[mols_per_group >= 3].index
        common_groups = no_azo_groups.loc[indices]

        fps = common_groups.apply(lambda x: 
                                  x.fingerprint(fp_type='MACCS')['MACCS'])
        diverse_indices = self._diversify(fps)
        diverse_groups = no_azo_groups.iloc[diverse_indices]
        return diverse_groups

    def find_groups_with_pattern(self, pattern):
        group_ids = []
        frag_groups = self.get_frag_groups(tier=0)
        for fg in frag_groups:
            if fg.has_pattern(pattern):
                group_ids.append(fg.get_id())
        return group_ids
    