import os.path
import math
from itertools import chain
from functools import reduce
from collections import defaultdict
import re
import statistics
import pandas as pd


from rdkit import Chem
from rdkit import DataStructs
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
from rdkit.ML.Cluster import Butina

from pymongo import MongoClient

from entities import Molecule, Fragment, FragmentGroup, Substituent, Bridge
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
    
    SMARTS_mols = {}
        
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
        index = entity_df.apply(lambda e: e.get_id())
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
    
    def set_conjugation(self, molecules=None):
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
    
    def __init__(self):
        self._configure()
        self._set_up()
        
    def _set_up(self):
        self.fragments = pd.Series([], dtype=object)
        self.mol_data = MoleculeData()
        
    def _process_link_file(self, filename, link_array=None):
        link_dir = self._config.get_directory("link_tables")
        link_file = os.path.join(link_dir, filename)
        regen = self._config.get_regen('grouping')
        if link_array:
            if not os.path.isfile(link_file) or regen:
                self._fh.output_to_text(link_array, link_file)
            return 0
        if os.path.isfile(link_file) and not regen:
            link_array = self._fh.load_from_text(link_file)
            link_table = pd.DataFrame(link_array)
            link_table = link_table.apply(pd.to_numeric, errors='coerce')
            return link_table
        return pd.DataFrame([])
        
    def _filter_leaf(self, fragment):
        if not fragment.is_leaf:
            return 0
        return 1
        
    def _filter_all(self, fragment):
        if not fragment.is_leaf:
            return 0
        raw = re.sub('\[.*?\]', '', fragment.smiles)
        raw = re.sub('[()]', '', raw)
        if bool(re.match('^[C]+$', raw)):
            return 0
        # if bool(re.match('^[c1ccccc1]+$', raw)):
        #     return 0
        if '.' in raw:
            return 0
        
        conj_SMARTS = '[#6;!X4]'
        ring_SMARTS = '[*;R]'
        try:  
            patt1 = self.SMARTS_mols[conj_SMARTS]
            patt2 = self.SMARTS_mols[ring_SMARTS]
        except KeyError:
            patt1 = Chem.MolFromSmarts(conj_SMARTS)
            patt2 = Chem.MolFromSmarts(ring_SMARTS)
            self.SMARTS_mols[conj_SMARTS] = patt1
            self.SMARTS_mols[ring_SMARTS] = patt2
        mol = fragment.get_rdk_mol()
        if mol.HasSubstructMatch(patt1) and mol.HasSubstructMatch(patt2):
            return fragment
        return 0
    
    def _sub_pattern(self, sub):
        sub = Chem.MolFromSmiles(sub)
        sub = Chem.MolToSmarts(sub)
        sub = re.sub(r'\[(.+?)\]', r'[\1;!R]', sub)
        sub = re.sub(r'\[(.+?)\](?!.+\[(.+?)\])', r'[\1;D1]', sub)
        return sub
    
    def _direct_sub_pattern(self, sub):
        sub = self._sub_pattern(sub)
        direct_sub = re.sub(r'\[(.+?)\]', r'[\1;$([\1]-[#6;R])]', sub, count=1)

        return Chem.MolFromSmarts(direct_sub)
    
    def _set_mol_frag_link(self):
        filename = "mol_frag_link.txt"
        link_table = self._process_link_file(filename)
        if link_table.empty:
            link_table = []
            frags = self.get_fragments()
            link_table = pd.DataFrame([[mol_id,f_id] 
                         for f_id, m_list in frags.apply(
                                 lambda x: x.get_parent_mols()).iteritems()
                         for mol_id in m_list])
        link_table.columns = ['mol_id','frag_id']
        self._mol_frag_link = link_table
        self._process_link_file(filename, link_table.values.tolist())
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
        
    def set_occurrence(self):
        frags = self.get_fragments()
        
        for frag in frags:
            if frag.occurrence:
                break
            num_parents = self.parent_mol_count(frag)
            frag.occurrence = num_parents
        
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
                    existing.add_parent_mol(mol.get_id())
                    continue
                new_frag = Fragment(frag, id_)
                new_frag.add_parent_mol(mol.get_id())
                if frag in frags[1]:
                    new_frag.is_leaf = True
                frags_dict[frag] = new_frag
                id_ += 1

        self.fragments = self._create_entity_df(frags_dict)
        self._set_mol_frag_link()
        self.set_occurrence()
        
        if regen and self.frags_file:
            self._save_frags_to_file()
        return self.fragments
    
    def get_parent_mols(self, id_):
        mols = self.mol_data.get_molecules()
        link_table = self._get_mol_frag_link()
        parent_ids = link_table[id_ == link_table['frag_id']]['mol_id']
        parents = mols.loc[parent_ids.values]
        return parents
    
    def get_mol_frags(self, id_):
        self.mol_data.get_molecules()
        frags = self.get_fragments()
        link_table = self._get_mol_frag_link()
        frag_ids = link_table[id_ == link_table['mol_id']]['frag_id']
        frags = frags[frag_ids]
        frags = frags[frags.apply(lambda x: x.is_leaf)]
        return frags
    
    def parent_mol_count(self, frag):
        parent_mols = self.get_parent_mols(frag.get_id())
        return parent_mols.size
           
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
        clean_frags = frags[frags.apply(self._filter_all) != 0]
        return clean_frags
    
    def leaf_frags(self, frags=None):
        if not frags:
            frags = self.get_fragments()
        leaf_frags = frags[frags.apply(self._filter_leaf) != 0]
        return leaf_frags
    
    def frags_to_mols(self, frag_smiles=None):
        if not frag_smiles:
            frag_smiles = self.fragments.apply(lambda f: f.smiles)
        mols = frag_smiles.apply(lambda f: Chem.MolFromSmiles(f))
        return mols


class FragmentGroupData(FragmentData):
    
    MANUAL_GROUPS = ['triarylamine']
    
    def __init__(self):
        super().__init__()
        super()._set_up()

    def _set_up(self):
        
        self._frag_groups_dict = {}   
        self._tier_link_tables = {}
    
    def _set_frag_group_link(self):
        
        filename = "frag_group_link.txt"
        link_table = self._process_link_file(filename)
        if link_table.empty:
            link_table = []
            frag_groups = self.get_frag_groups(tier=0)
            link_table = pd.DataFrame([[fg_id, frag_id] 
                         for fg_id, f_list in frag_groups.apply(
                                 lambda fg: fg.get_leaf_frags()).iteritems()
                         for frag_id in f_list])
        link_table.columns = ['group_id','frag_id']
        self._frag_group_link = link_table
        self._process_link_file(filename, link_table.values.tolist())
        return link_table
    
    def _set_tier_link(self, parent_tier=1):
        
        filename = f"tier_{parent_tier}_link.txt"
        link_table = self._process_link_file(filename)
        if link_table.empty:
            link_table = []
            child_groups = self.get_frag_groups(parent_tier - 1)
            link_table = pd.DataFrame([[child_id, parent_id] 
                         for child_id, parent_id in child_groups.apply(
                                 lambda fg: fg.get_parent_group()).iteritems()])
        link_table.columns = ['child_id', 'parent_id']
        self._tier_link_tables[parent_tier] = link_table
        self._process_link_file(filename, link_table.values.tolist())
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
    
    def _first_sim_grouping(self, all_groups, diverse_groups, groups_dict):
        
        all_groups.drop(diverse_groups.index, inplace=True)
        for group in diverse_groups:
            fp = group.fingerprint('MACCS')
            similarities = all_groups.apply(self._compute_similarity, args=(fp,))
            sim_groups = similarities[similarities >= 0.8]
            groups_dict[group.get_id()] = all_groups.loc[sim_groups.index]
            all_groups.drop(sim_groups.index, inplace=True)
        return all_groups
    
    def _second_sim_grouping(self, remain_groups, mols_per_group, groups_dict):
        
        mols_per_group.sort_values(ascending=True, inplace=True)
        remain_groups = remain_groups[mols_per_group.index]
        remain_copy = remain_groups.copy()
        i = 0
        while not remain_copy.empty or i >= remain_groups.size:
            group = remain_groups.iloc[i]
            if group.get_id() not in remain_copy:
                i += 1
                continue
            remain_copy.drop(group.get_id(), inplace=True)
            fp = group.fingerprint('MACCS')
            similarities = remain_copy.apply(self._compute_similarity, args=(fp,))
            sim_groups = similarities[similarities >= 0.8]
            groups_dict[group.get_id()] = remain_copy.loc[sim_groups.index]
            remain_copy.drop(sim_groups.index, inplace=True)
            i += 1
            
    def _diversify(self, fps, diverse_groups=100):
        
        picker = MaxMinPicker()
        nfps = fps.size
        def distij(i, j, fps=fps.tolist()):
            return 1-DataStructs.DiceSimilarity(fps[i],fps[j])
        pick_indices = picker.LazyPick(distij, nfps, diverse_groups, seed=23)
        return list(pick_indices)
    
    def _compute_similarity(self, group, fp2):
        
        fp1 = group.fingerprint('MACCS')
        return DataStructs.DiceSimilarity(fp1, fp2)
    
    def _remove_azo_groups(self, groups):
        
        is_azo = groups.apply(lambda x: x.check_is_azo())
        return groups[~is_azo]
    
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
            
    def _identify_substits(self):

        frags = self.get_frag_groups(tier=0)
        all_subs = set()
        for f in frags:
            if not f.is_isolated_benzene():
                continue
            subs = f.get_subs_from_core('c1ccccc1')
            if subs:
                all_subs.update(subs)
        self._direct_subs = {sub: self._direct_sub_pattern(sub) for 
                         sub in all_subs}
        
    def _check_manual_groups(self, mol):
        
        manual_groups_dict = {}
        for group in self.MANUAL_GROUPS:
            group_SMARTS = self.PATTERNS[group]
            try:
                patt = self.SMARTS_mols[group_SMARTS]
            except KeyError:
                patt = Chem.MolFromSmarts(group_SMARTS)
                self.SMARTS_mols[group_SMARTS] = patt
            group_count = mol.pattern_count(patt)
            if not group_count > 0:
                continue
            if group == 'methine':
                group_count += 1
            manual_groups_dict[group] = group_count
        return manual_groups_dict
            
    def _set_cluster_group_map(self, cutoff=0.2):
        
        groups = self.get_frag_groups(tier=1)
        fps = groups.apply(lambda fg: fg.fingerprint('MACCS')).tolist()
        nfps = groups.size
        dists = []
        for i in range(1, nfps):
            sims = DataStructs.BulkDiceSimilarity(fps[i], fps[:i])
            dists.extend([1-x for x in sims])
        clusters = Butina.ClusterData(dists, nfps, cutoff, isDistData=True)
        
        cgm = []
        for i, cluster in enumerate(clusters):
            clust_groups = groups.iloc[list(cluster)]
            clust_groups = clust_groups.apply(lambda x: x.set_cluster(i))
            group_idx = clust_groups.index
            cluster = [i]*len(group_idx)
            cgm.extend(zip(cluster, group_idx))
        cgm = pd.DataFrame(cgm)
        cgm.columns = ['Cluster', 'Group']
        self._cgm = cgm
        
    def _taa_parents(self):
        
        taa_SMARTS = self.PATTERNS['triarylamine']
        patt = Chem.MolFromSmarts(taa_SMARTS)
        mols = self.mol_data.get_molecules()
        has_taa = mols.apply(lambda x: x.has_pattern(patt))
        return mols[has_taa]
    
    def pickle_groups(self):
        
        tiers = [0, 1]
        for tier in tiers:
            groups = self.get_frag_groups(tier)
            pickled_file = self._config.get_directory('pickle') +  f'tier_{tier}.pickle'
            self._fh.dump_to_pickle(groups, pickled_file)
            
    def set_occurrence(self, tier=0):
        
        groups = self.get_frag_groups(tier)
        for group in groups:
            if group.occurrence:
                break
            if group.is_taa:
                group.occurrence = self._taa_parents().count
                continue
            num_parents = self.parent_mol_count(group)
            group.occurrence = num_parents

    def get_frag_group(self, id_):
        
        tier = 0
        frag_groups = self.get_frag_groups(tier)
        while not id_ in frag_groups:
            tier += 1
            frag_groups = self.get_frag_groups(tier)
        fg = frag_groups.loc[id_]
        return fg
        
    def get_frag_groups(self, tier=0):
        
        try:
            return self._frag_groups_dict[tier]
        except KeyError:
            pass
        pickled_file = self._config.get_directory('pickle') +  f'tier_{tier}.pickle'
        regen = self._config.get_regen(f'grouping')
        
        if os.path.isfile(pickled_file) and not regen:
            frag_groups = self._fh.load_from_pickle(pickled_file)
            self._frag_groups_dict[tier] = frag_groups
            return frag_groups

        grouping_func = 'self.get_tier_' + str(tier) + '()'
        grouping_result = eval(grouping_func)
        frag_groups = self._create_entity_df(grouping_result)
        self._frag_groups_dict[tier] = frag_groups
        if tier > 0:
            self._set_tier_link(tier)
        else:
            self._set_frag_group_link()
        self.set_occurrence(tier)
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
    
    def get_group_parent_mols(self, id_):
        
        group = self.get_frag_group(id_)
        if len(group.get_parent_mols()) == 0:
            self.set_group_parent_mols(group.get_tier())
        parent_ids = group.get_parent_mols()
        mols = self.mol_data.get_molecules()
        return mols[parent_ids]
    
    def set_group_parent_mols(self, tier=1):
        
        groups = self.get_frag_groups(tier)
        for group in groups:
            if group.is_taa:
                print(group.get_id())
                parent_mols = self._taa_parents()
                group.set_parent_mols(parent_mols.index.tolist())
            elif len(group.get_parent_mols()) == 0:
                parent_mol_ids = self._get_parent_mol_ids(group)
                group.set_parent_mols(parent_mol_ids)
    
    def parent_mol_count(self, group):
        
        parent_mols = self.get_group_parent_mols(group.get_id())
        return parent_mols.size
    
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
        frags = super().clean_frags()
        link_table = self._get_frag_group_link()
        leaf_frag_ids = link_table[id_ == link_table['group_id']]['frag_id']
        leaf_frags = frags.loc[leaf_frag_ids]
        return leaf_frags
    
    def get_tier_1(self):
        
        prev_tier = self.get_frag_groups(tier=0)
        self.set_frag_subs(prev_tier)
        groups_dict = {}
        id_ = prev_tier.index[-1] + 1
        for fg in prev_tier:
            core = fg.remove_subs(self._direct_subs)
            if not core:
                continue
            group = groups_dict.get(core)
            if group:
                fg.set_parent_group(group.get_id())
                continue
            new_group = FragmentGroup(core, id_, tier=1)
            fg.set_parent_group(new_group.get_id())
            groups_dict[core] = new_group
            id_ += 1
        taa_SMILES = self.PATTERNS['triarylamine']
        taa_group = FragmentGroup(taa_SMILES, id_, tier=1)
        taa_group.is_taa = True
        groups_dict[taa_SMILES] = taa_group
        return groups_dict
    
    def set_frag_subs(self, groups):
        
        self._identify_substits()
        for group in groups:
            group.set_substituents(self._direct_subs)
        
    def get_group_clusters(self):
        
        try:
            return self._cgm
        except AttributeError:
            self._set_cluster_group_map()
        return self._cgm
    
    def get_tier_2(self):
        
        prev_tier = self.get_frag_groups(tier=1)
        groups_dict = {}
        
        id_ = prev_tier.index[-1] + 1
        no_azo_groups = self._remove_azo_groups(prev_tier)
        azo_smi = 'C-N=N-C'
        azo_children = prev_tier.drop(no_azo_groups.index)
        azo_group = FragmentGroup(azo_smi, id_, tier=2)
        for azo in azo_children:
            azo.set_parent_group(azo_group.get_id())
        groups_dict[azo_smi] = azo_group
        id_ += 1
        
        substit_groups = no_azo_groups.apply(lambda fg: fg.is_substituent)
        all_groups = no_azo_groups[~substit_groups]
        
        mols_per_group = all_groups.apply(self.parent_mol_count)
        common_indices = mols_per_group[mols_per_group >= 3].index
        common_groups = all_groups.loc[common_indices]

        fps = all_groups.apply(lambda x: 
                                  x.fingerprint('MACCS'))
        diverse_indices = self._diversify(fps.loc[common_indices])
        diverse_groups = common_groups.iloc[diverse_indices]
        
        sim_groups_dict = {}
        remaining_groups = self._first_sim_grouping(all_groups, diverse_groups, sim_groups_dict)
        mols_per_group = mols_per_group.loc[remaining_groups.index]
        self._second_sim_grouping(remaining_groups, mols_per_group, sim_groups_dict)
        
        for group_id, children in sim_groups_dict.items():
            group = prev_tier.loc[group_id]
            new_group = FragmentGroup(group.smiles, id_, tier=2)
            group.set_parent_group(new_group.get_id())
            for child in children:
                child.set_parent_group(new_group.get_id())
            groups_dict[group.smiles] = new_group
            id_ += 1
        self._set_tier_link(parent_tier=2)
        return groups_dict
        
    def get_mol_groups(self, id_, tier=0):
        frags = self.get_mol_frags(id_)
        group_ids = []
        frag_link = self._get_frag_group_link()
        tier_link = self._get_tier_link(1)
        isin = frag_link['frag_id'].isin(frags.index)
        group_ids = frag_link['group_id'][isin]
        
        groups = self.get_frag_groups(tier=0)
        groups = groups[group_ids]
        if tier == 0:
            return groups

        isin = tier_link['child_id'].isin(groups.index)
        group_ids = tier_link['parent_id'][isin].dropna()
        groups = self.get_frag_groups(tier=1)
        groups = groups[group_ids]
        groups.drop_duplicates(inplace=True)
        return groups

    def find_groups_with_pattern(self, pattern):
        
        group_ids = []
        frag_groups = self.get_frag_groups(tier=0)
        for fg in frag_groups:
            if fg.has_pattern(pattern):
                group_ids.append(fg.get_id())
        return group_ids
    
    def get_mol_fp(self, id_):
        
        mol = self.mol_data.get_molecule(id_)
        groups = self.get_mol_groups(id_, tier=1)
        grouping_dict = defaultdict(int)
        cgm = self.get_group_clusters()
        num_clusters = cgm['Cluster'].max() + 1
        vect_size = num_clusters + len(self.MANUAL_GROUPS)
        fp_vect = [0] * vect_size
        for group in groups:
            count = mol.pattern_count(Chem.MolFromSmarts(group.smiles))
            cluster = cgm.loc[group.get_id() == cgm['Group'], 'Cluster'].iloc[0]
            fp_vect[cluster] += count
        manual_groups = self._check_manual_groups(mol)
        grouping_dict.update(manual_groups)
        
        return fp_vect
    
    def order_cluster_by_sim(self, cluster):

        #Take each molecule in a cluster and group by vector FP distance
        #Find all mols that have identical vector FP and within this, if
        #groups in certain bit are not the same, find the one that is closest
        #based on MACCS fp of the groups
        pass
    
    def populous_clusters(self):
        
        cgm = self.get_group_clusters()
        num_clusters = cgm['Cluster'].max()
        groups = self.get_frag_groups(tier=1)
        pop_clust = []
        for clust in range(0, num_clusters + 1):
            group_idx = cgm[clust == cgm['Cluster']]['Group']
            parent_mols = groups[group_idx].apply(self.parent_mol_count)
            if parent_mols[parent_mols > 2].empty:
                continue
            pop_clust.append(clust)
        return pop_clust
    
    def comp_deviation(self):
        
        cgm = self.get_group_clusters()
        pop_clust = self.populous_clusters()
        groups = self.get_frag_groups(1)
        comp_dict = {}
        for group in groups:
            parents = self.get_group_parent_mols(group.get_id())
            self.mol_data.set_comp_data()
            lambdas = parents.apply(lambda x: x.lambda_max)
            comp_dict[group.get_id()] = lambdas.tolist()
        group_comp = pd.Series(comp_dict)
        
        comp_dict = {}
        for clust in pop_clust:
            groups = cgm[clust == cgm['Cluster']]['Group']
            lambdas = group_comp[groups]
            lambdas = list(chain.from_iterable(lambdas.values))
            comp_dict[clust] = statistics.stdev(lambdas)
        lambdas_dev = pd.Series(comp_dict).sort_values(ascending=True)
        lambdas_dev.dropna(inplace=True)
        return lambdas_dev
            
    def fingerprint_mols(self):
        
        mols = self.mol_data.get_molecules()
        fp_dict = {}
        for mol in mols:
            fp = self.get_mol_fp(mol.get_id())
            fp_dict[mol.get_id()] = fp
        return pd.Series(fp_dict)
    
    def get_on_bits(self, fp):
        return {i: bit for i, bit in enumerate(fp) if bit > 0}
    
    def create_sub_objects(self, subs, groups, existing_subs):
        obj_subs = []
        str_subs = []
        for sub in subs:    
            mol = Chem.MolFromSmiles(sub)
            sub_flag = True
            for group in groups:
                if mol.HasSubstructMatch(group):
                    sub_flag = False
                    break
            if not sub_flag:
                continue
            str_subs.append(sub)
            test_sub = re.sub('\[*[0-9]*\*+\]*', '[*]', sub)
            try:
                sub_obj = existing_subs[test_sub]
                obj_subs.append(sub_obj)
            except KeyError:
                id_ = len(existing_subs)
                new_sub = Substituent(test_sub, id_)
                existing_subs[test_sub] = new_sub
                obj_subs.append(new_sub)
        return obj_subs, str_subs    
    
    def create_bridge_objects(self, bridges, existing_bridges):
        
        obj_bridges = []
        for bridge in bridges:
            try:
                bridge_obj = existing_bridges[bridge]
                obj_bridges.append(bridge_obj)
            except KeyError:
                id_ = len(existing_bridges)
                new_bridge = Bridge(bridge, id_)
                existing_bridges[bridge] = new_bridge
                obj_bridges.append(new_bridge)
        return obj_bridges
    
    def get_subs_and_bridges(self):
        mol_subs_map = []
        mol_bridge_map = []
        subs_dict = {}
        bridges_dict = {}
        my_mols = self.mol_data.get_molecules()
        self.mol_data.set_comp_data()
        for my_mol in my_mols:
            if '.' in my_mol.smiles:
                continue
            if not my_mol.lambda_max:
                continue
            mol = my_mol.get_rdk_mol()
            groups = self.get_mol_groups(my_mol.get_id(), tier=1)
            group_size = groups.apply(lambda x: x.get_size())
            groups = pd.concat([groups, group_size], axis=1)
            my_groups = groups.sort_values(1, ascending=True).drop(1, 1)[0]
            groups = my_groups.apply(lambda x: x.get_rdk_mol())
            matches = []
    
            for group in groups:
                g_matches = mol.GetSubstructMatches(group)
                g_matches = [set(match) for match in g_matches]
                matches.append(g_matches)
                
            mol_subs = []
            for i in range(0, len(matches)):
                group_rdk = groups.iloc[i]
                my_group = my_groups.iloc[i]
                if i < len(matches) - 1:
                    flat_list = chain.from_iterable(matches[i+1:])
                    try:
                        other_atoms = reduce(lambda s1, s2: s1.union(s2), flat_list)
                    except TypeError:
                        other_atoms = {}
                else:
                    other_atoms = set()
                for j in range(0, len(matches[i])):
                    match = matches[i][j]
                    if match.issubset(other_atoms):
                        continue
                    rm = Chem.ReplaceCore(mol, group_rdk, matches=tuple(match), labelByIndex=True)
                    smiles = Chem.MolToSmiles(rm)
                    if not smiles:
                        continue
                    branches = smiles.split('.')

                    obj_subs, str_subs = self.create_sub_objects(branches, groups, subs_dict)
                    new_records = [[ my_mol.get_id(), my_group.get_id(), 
                                    j, sub.get_id() ] for sub in obj_subs]
                    mol_subs_map.extend(new_records)
                    mol_subs.extend(str_subs)
            trimmed_mol = mol
            for group in groups[::-1]:
                trimmed_mol = Chem.ReplaceCore(trimmed_mol, group, labelByIndex=True)
                while trimmed_mol:
                     tmp = Chem.MolToSmiles(trimmed_mol)
                     trimmed_mol = Chem.ReplaceCore(trimmed_mol, group, labelByIndex=True)
                trimmed_mol = Chem.MolFromSmiles(tmp)
            smiles = Chem.MolToSmiles(trimmed_mol)
            branches = smiles.split('.')
            branches = [re.sub('\[*[0-9]*\*+\]*', '[*]', branch) for 
                        branch in branches]
            branches = [branch for branch in branches if not 
                        re.match('^(\[*\*\]*)+$', branch)]
            actual_subs = [re.sub('\[*[0-9]*\*+\]*', '[*]', sub) for
                           sub in mol_subs]
            for sub in actual_subs:
                if sub in branches:
                    branches.remove(sub)
            patt = Chem.MolFromSmarts(self.PATTERNS['triarylamine'])
            num_tri = my_mol.pattern_count(patt)
            for i in range(0, num_tri):
                try:
                    branches.remove('[*]N([*])[*]')
                except ValueError:
                    break
            obj_bridges = self.create_bridge_objects(branches, bridges_dict)
            new_records = [[my_mol.get_id(), bridge.get_id()] for bridge in obj_bridges]
            mol_bridge_map.extend(new_records)
            
            self.mol_sub_link = pd.DataFrame(mol_subs_map, 
                                             columns=['Molecule', 'Group', 'Group Instance', 'Sub'])
            self.mol_bridge_link = pd.DataFrame(mol_bridge_map, 
                                 columns=['Molecule', 'Bridge'])
            self.substituents = pd.Series(subs_dict)
            self.bridges = pd.Series(bridges_dict)             
