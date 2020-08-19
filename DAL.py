#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path
import re
import math
from collections import defaultdict
from itertools import combinations
from functools import reduce

import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
from rdkit.ML.Cluster import Butina

from database import MongoLoad, LinkTable
from helper import MyConfig, MyLogger, MyFileHandler, MyConfigParamError
from entities import Molecule, Fragment, Group, Substituent, Bridge

    
class EntityData:
    """
    superclass for the Data child classes.
    
    this class contains a few a helper methods that
    are used by all child classes, such as pickling
    and setting the link tables. This class should
    never be instantiated directly.
    
    Attributes
    ----------
    SMARTS_mols : dict
        {SMARTS string : RDKit.Mol representation}
        Updated dynamically as new patterns are defined. 
        Saves having to create the Mol object again.e
    
    Methods
    -------
    pickle(entities, name)
        pickle entity objects to prevent regeneration
    get_link_table()
        interface for the LinkTable object
    """
    
    SMARTS_mols = {}
    _linker = LinkTable()
    
    def _configure(self):
        self._logger = MyLogger().get_child(type(self).__name__)
        self._fh = MyFileHandler()
        self._config = MyConfig()
        self._get_patterns()
        
    def _get_patterns(self):
        """
        load the user-defined SMARTS patterns from file

        Returns
        -------
        None.

        """
        file = self._config.get_directory("patterns")
        patterns = self._fh.load_from_text(file, delim='|')
        self.PATTERNS = {p[0]: p[1] for p in patterns}
        
    def _create_entity_df(self, entities):
        """
        convert a list of entities into a pandas Series

        Parameters
        ----------
        entities : list
            a list of entity objects (molecules, fragments, groups etc.)

        Returns
        -------
        entity_df : pandas.Series
            a pandas Series object with the entities as values and the
            IDs as the index

        """
        entity_df = pd.Series(entities)
        index = entity_df.apply(lambda e: e.get_id())
        entity_df.index = index
        return entity_df
    
    def pickle(self, entities, name):
        """
        pickle a series of entities for future convenience

        Parameters
        ----------
        entities : pandas.Series
            a series of entities
        name : str
            name of the entity (group, fragment etc.)

        Returns
        -------
        None.

        """
        pickled_file = self._config.get_directory('pickle') +  f'{name}.pickle'
        self._fh.dump_to_pickle(entities, pickled_file)
            
    def get_link_table(self, table_name):
        """
        return link table for two entities
        
        the link tables store many-to-many entity relationships.
        For example, a fragment object can be found in multiple molecules,
        and a molecule can contain multiple fragments.

        Parameters
        ----------
        table_name : str
            name of the link table. Should be labelled as
            [first_entity_name]_[second_entity_name] e.g.
            molecule_group.

        Returns
        -------
        pandas.DataFrame
            a DataFrame with two columns of entity IDs

        """
        return self._linker.get_link_table(table_name)


class MoleculeData(EntityData):
    """
    a class that performs data processing and algorithms on Molecules
    
    this class contains methods to retrieve the molecules from
    the database as SMILES strings, create the corresponding Molecule 
    objects and transform, filter or add data to these objects.
    
    Methods
    -------
    get_molecule(id)
        get a Molecule object by its ID
    get_molecules()
        return a pandas Series of all Molecule objects
    set_conjugation(molecules)
        retrieve and set the number of conjugated bonds
        in each Molecule
    pattern_count()
        the number of molecules containing each of a list
        of user-defined patterns 
    find_mols_with_pattern(patt, clean_only=False)
        return only the molecules that contain the 
        substructure 'patt'
    find_mols_no_pattern(clean_only=False)
        return the molecules that dont match any of the
        user-defined patterns
    patterns_in_combo()
        return the number of times any combination of
        two user-defined patterns occur in the same Molecule
    clean_mols()
        return only the Molecules that have realistic computational
        data
    set_comp_data()
        retrieve and store the sTDA wavelengths and strengths
        for each molecule
    """
    def __init__(self):
        self._configure()
        self._set_up()
        
    def _set_up(self):
        self._molecules = pd.Series([], dtype=object)
        self._comp_flag = False
        
    def _comp_filter(self, mol, f_min, lambda_max, lambda_min):
        """
        filter for Molecules with valid computational data
        
        this method assumes the the computational data has
        already been retrieved and set before being called.

        Parameters
        ----------
        mol : Molecule
            the Molecule whose computational data to check
        f_min : float
            the minimum permissable strength
        lambda_max : float
            the maximum permissable wavelength
        lambda_min : float
            the minimum permissable wavelength

        Returns
        -------
        mol/None : Molecule/None
            returns the Molecule object if its data is valid or
            else None

        """
        if not mol.get_lambda_max():
            return None
        if '.' in mol.smiles:
            return None
        if (lambda_min < mol.get_lambda_max() < lambda_max) \
        and mol.get_strength_max() > f_min:
            return mol
        return None
            
    def get_molecule(self, mol_id):
        """
        retrieve a particular Molecule object

        Parameters
        ----------
        mol_id : int
            ID of the Molecule to retrieve

        Returns
        -------
        mol : Molecule
            the Molecule with ID 'mol_id'

        """
        molecules = self.get_molecules()
        mol = molecules[mol_id]
        return mol
    
    def get_molecules(self, regen=False):
        """
        generate and return a series of Molecule objects
        
        the Molecules can either be generated from scratch
        by retrieving the list of unique molecule SMILES from
        the database, or else the objects can be reloaded from
        a pickle, depending on the preference specified in the config
        file.

        Parameters
        ----------
        regen : boolean, optional
            if True ignore the pickle file or any in-memory  
            data and regenerate from the raw SMILES. The default is False.

        Returns
        -------
        pandas.Series
            a series of Molecule objects indexed by their IDs

        """
        if not self._molecules.empty and not regen:
            return self._molecules
        
        with MongoLoad() as mongo:
            smiles = mongo.resolve_query('smiles')
        pickled_file = self._config.get_directory('pickle') +  'molecules.pickle'
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

        self._molecules = self._create_entity_df(mols)
        return self._molecules
    
    def set_conjugation(self, molecules=None):
        if not molecules:
            molecules = self.get_molecules()
        conj_file = os.path.join(self._config.get_directory('conjugation'),'conjugation.csv')
        conjugation = self._fh.load_from_text(conj_file)
        smi_conj = list(zip(*conjugation))
        conjugation = pd.Series(smi_conj[2], index=smi_conj[1])

        for m in molecules:
            try:
                conj = int(conjugation.loc[m.smiles])
            except KeyError:
                conj = None
            m.set_conjugation(conj)
            
    def pattern_count(self):
        for patt in self.PATTERNS:
            num_mols = self.find_mols_with_pattern(patt).size
            print(f'{patt}: {num_mols}')
            
    def find_mols_with_pattern(self, patt, clean_only=False):
        """
        return only the molecules that contain the 
        substructure 'patt'

        Parameters
        ----------
        patt : str
            the name of the pattern in the user-defined
            patterns file e.g. triarylamine.
        clean_only : boolean, optional
            if true only search molecules with valid computational
            data. The default is False.

        Returns
        -------
        matches : pandas.Series
            a series of Molecules that contain the pattern

        """
        if patt == 'NoGroup':
            return self.find_mols_no_pattern(clean_only)
        if clean_only:
            mols = self.clean_mols()
        else:
            mols = self.get_molecules()
        pattern = self.PATTERNS[patt]
        pattern = Chem.MolFromSmarts(pattern)
        
        matches = mols.apply(lambda x: x if 
                                  x.has_pattern(pattern) else None)
        matches = matches.dropna()
        return matches
    
    def find_mols_no_pattern(self, clean_only=False):
        """
        return the molecules that contain the 
        none of the user-defined patterns

        Parameters
        ----------
        clean_only : boolean, optional
            if true only search molecules with valid computational
            data. The default is False.

        Returns
        -------
        matches : pandas.Series
            a series of Molecules that contain no pattern

        """
        if clean_only:
            mols = self.clean_mols()
        else:
            mols = self.get_molecules()
        patterns = [Chem.MolFromSmarts(smarts) for smarts in list(self.PATTERNS.values())]
        tmp_mols = []
        for mol in mols.tolist():
            found = False
            for pattern in patterns:
                if mol.has_pattern(pattern):
                    found = True
                    break
            if not found:
                tmp_mols.append(mol.smiles)
        matches = mols.filter(tmp_mols)
        return matches
    
    def patterns_in_combo(self):
        """
        number of times patterns occur in same Molecule
        
        output the number of times any combination of
        two user-defined patterns occur in the same Molecule
        as a dictionary of {(pattern_1, pattern_2) : num_mols}

        Returns
        -------
        None.

        """
        combo_dict = defaultdict(int)
        mols = self.md.clean_mols()
        for mol in mols:
            patterns = []
            for pattern, SMARTS in self.PATTERNS.items():
                try:  
                    patt = self.SMARTS_mols[SMARTS]
                except KeyError:
                    patt = Chem.MolFromSmarts(SMARTS)
                    self.SMARTS_mols[SMARTS] = patt
                if mol.has_pattern(patt):
                   patterns.append(pattern)
            combos = list(combinations(patterns, 2))
            for combo in combos:
                combo_dict[combo] += 1
        sorted_dict = {k: v for k, v in sorted(
            combo_dict.items(), reverse=True, key=lambda item: item[1])}
        for k,v in sorted_dict.items():
            print(k,v)
        
    def clean_mols(self):
        """
        return Molecules with valid sTDA absorption data

        Returns
        -------
        molecules: pandas.Series
            series of Molecules with valid sTDA absorption data

        """
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
            f_min = lambda_min = 0
            lambda_max = math.inf
            self._logger.warning(e)
        molecules = molecules.apply(self._comp_filter, 
                                    args=(f_min, lambda_max, lambda_min))
        molecules = molecules.dropna()
        self.clean_molecules = molecules
        return molecules
    
    def set_comp_data(self):
        """
        retrieve and set the sTDA absorption data for each Molecule

        Only retrieves the amplitudes and the strengths of the first
        3 excitations
        
        Returns
        -------
        None.

        """
        if self._comp_flag:
            return
        mols = self.get_molecules()
        with MongoLoad() as mongo:
            comp_df = mongo.get_comp_uvvis()
        for m in mols:
            try:
                record = comp_df.loc[m.smiles].iloc[0]
            except KeyError:
                continue
            m.set_comp_data(record)
        self._comp_flag = True
        
            
class FragmentData(EntityData):
    """
    a class that performs data processing and algorithms on Fragments
    
    this class contains methods to create Fragment objects from
    the entire set of Molecule objects, and to transform, 
    filter or add data these Fragment objects.
    
    Attributes
    ----------
    md : MoleculeData
        a MoleculeData instance for accessing the molecules needed
        to create fragments
    
    Methods
    -------
    set_occurrence()
        determine the number of molecules containing each Fragment
    get_fragment(frag_id)
        return the Fragment object with ID 'frag_id'
    get_fragments(regen=False)
        fragment a series of Molecule object and and generate
        Fragment objects from the resulting SMILES
    get_mol_frags(mol_id)
        return all the Fragment objects that are substructures
        of the Molecule with ID 'mol_id'
    get_frag_mols(frag_id)
        return all the Molecule objects that contain the Fragment
        with ID 'frag_id'
    clean_frags()
        return only those Fragment objects that could constitute
        a core group
    """
    def __init__(self, mol_data):
        self.md = mol_data
        self._configure()
        self._set_up()
        
    def _set_up(self):
        self._fragments = pd.Series([], dtype=object)
        
    def _filter(self, fragment):
        """
        check a Fragment meets criteria to be a Group
        
        in order to be a core group a fragment must have
        at least one ring and one conjugated carbon

        Parameters
        ----------
        fragment : Fragment
            the Fragment object to check

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        #SMARTS for a carbon that is not sp3 hybridized
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
            
    def _get_frags_file(self):
        """
        check if BRICS fragments stored on file and return file name
        """
        try:
            frags_dir = self._config.get_directory('fragments')
            frags_file = os.path.join(frags_dir, 'frags.json')
    
        except MyConfigParamError as e:
            self._logger.warning(e)
            frags_file = None
        return frags_file
            
    def _parse_frags_file(self, frags_file):
        """
        load BRICS fragments from a text file into a dictionary
        """
        frags_dict = {}
        
        if os.path.isfile(frags_file):
            frags_dict = self._fh.load_from_json(frags_file)
    
        return frags_dict

    def _save_frags_to_file(self, frags_file):
        """
        save BRICS fragments to file so dont have to repeat
        """
        frags_dict = defaultdict(list)
        lt = self._linker.get_link_table('mol_frag')
        frags = self.get_fragments()
        mols = self.md.get_molecules()
        for mol in mols:
            frag_ids = lt[lt['mol_id'] == mol.get_id()]['frag_id']
            mol_frags = frags[frag_ids].apply(lambda x: x.smiles)
            frags_dict[mol.smiles] = mol_frags.tolist()
        self._fh.output_to_json(frags_dict, frags_file)

    def set_occurrence(self):
        frags = self.get_fragments()
        for frag in frags:
            if frag.occurrence:
                break
            frag_mols = self.get_frag_mols(frag.get_id())
            frag.occurrence = frag_mols.size
            
    def get_fragment(self, frag_id):
        """
        retrieve a particular Fragment object

        Parameters
        ----------
        frag_id : int
            ID of the Fragment to retrieve

        Returns
        -------
        frag: Fragment
            the Fragment with ID 'frag_id'

        """
        fragments = self.get_fragments()
        frag = fragments.loc[frag_id][0]
        return frag    
    
    def get_fragments(self, regen=False):
        """
        return fragments of every Molecule in the database
        
        the Molecule object has its own fragment() method
        which applies the BRICS algorith. The fragments can 
        also be reloaded from a text file in their SMILES format 
        so the fragmentation algorithm does not have to be rerun 
        on every Molecule.

        Parameters
        ----------
        regen : boolean, optional
            ignore any in-memory data and rerun the entire
            process. The default is False.

        Returns
        -------
        pandas.Series
            a series of Fragment objects

        """
        if not self._fragments.empty and not regen:
            return self._fragments
        
        pickled_file = self._config.get_directory('pickle') + 'fragments.pickle'
   
        if os.path.isfile(pickled_file) and not regen:
            frags = self._fh.load_from_pickle(pickled_file)
            self._fragments = frags
            return frags
        
        frags_file = self._get_frags_file()
        if not frags_file:
            regen = True
            
        saved_frags = {}
        BRICS_regen = self._config.get_regen('BRICS')
        if not BRICS_regen:
            saved_frags = self._parse_frags_file(frags_file)
                   
        molecules = self.md.get_molecules()
        frags_dict = {}
        mol_frag_link = []
        id_ = 0
        for mol in molecules:
            try:
                frags = saved_frags[mol.smiles]
            except KeyError:
                frags = mol.fragment()
                
            for frag in frags:
                try:
                    frag_obj = frags_dict[frag]
                except KeyError:
                    frag_obj = Fragment(frag, id_)
                    frags_dict[frag] = frag_obj
                    id_ += 1
                mol_frag_link.append([mol.get_id(), frag_obj.get_id()])
                
        self._fragments = self._create_entity_df(frags_dict)
        self._linker.set_link_table('mol_frag', mol_frag_link)
        self.set_occurrence()
        self.pickle(self._fragments, 'fragments')
        if BRICS_regen and frags_file:
            self._save_frags_to_file(frags_file)
        return self._fragments
    
    def get_mol_frags(self, mol_id):
        """
        return the fragments contained by a certain Molecule
        """
        lt = self._linker.get_link_table(f'mol_frag')
        frag_ids = lt[mol_id == lt['mol_id']]['frag_id']
        frags = self.get_fragments().loc[frag_ids.values]
        return frags
    
    def get_frag_mols(self, frag_id):
        """
        return the molecules that contai a certain Fragment
        """       
        lt = self._linker.get_link_table(f'mol_frag')
        mol_ids = lt[frag_id == lt['frag_id']]['mol_id']
        mols = self.md.get_molecules().loc[mol_ids.values]
        return mols
    
    def clean_frags(self):
        """
        return only those Fragment objects that could constitute
        a core group
        """
        frags = self.get_fragments()
        clean_frags = frags[frags.apply(self._filter) != 0]
        return clean_frags   
    
    
class GroupData(EntityData):
    """
    a class that performs data processing and algorithms on Groups
    
    this class contains methods to create Group objects from
    the entire set of Fragment objects, and to transform, 
    filter or add data these Group objects.
    
    Attributes
    ----------
    SIMILARITIES : dict
        {metric_name : RDKit_implementation}
    md : MoleculeData
        a MoleculeData instance for accessing the molecules needed
        to create links
    fd : FragmentData
       a FragmentData instance for accessing the fragments needed
       to create groups  
       
    Methods
    -------
    set_occurrence()
        determine the number of molecules containing each Group
    get_group(group_id)
        return the Group object with ID 'group_id'
    get_groups(regen=False)
        strip each filtered Fragment object to its core and create a
        series of Group objects from these cores
    get_group_frags(group_id)
        return the Fragments that reduce to a particular Group
    get_group_cluster(group_id)
        return the cluster to which a particular Group belongs
    get_group_clusters(cutoff=0.2, similarity='dice', 
                       fp_type='MACCS', recluster=False, basic=False)
        cluster the Group objects according to their similarity using
        one of the existing fingerprinting schemes
    get_cluster_groups(cluster_id)
        get the Groups belonging to a particular cluster
    get_mol_groups(mol_id)
        return the groups contained by a particular Molecule
    get_group_mols(group_id)
        return the Molecules that contain a particular Group
    find_groups_with_pattern(pattern)
        return the Groups that have the user-defined
        substructure 'pattern'        
    
    """
    SIMILARITIES = {'dice': DataStructs.BulkDiceSimilarity,
                    'tanimoto': DataStructs.BulkTanimotoSimilarity}
    
    def __init__(self, mol_data, frag_data):
        self.md, self.fd = mol_data, frag_data
        self._configure()
        self._set_up()

    def _set_up(self):
        self._groups = pd.Series([], dtype=object)
    
    def _first_round_clustering(self, groups, diverse_groups, clusters):
        """
        part of basic clustering algorithm - not complete
        """
        remaining_groups = groups.drop(diverse_groups.index)
        cluster_id = 0
        for group in diverse_groups:
            fp = group.basic_fingerprint('MACCS')
            similarities = remaining_groups.apply(self._compute_similarity, args=(fp,))
            sim_groups = similarities[similarities >= 0.8]
            clust_groups = tuple(remaining_groups[sim_groups.index].index) + (group.get_id(),)
            clusters.append(clust_groups)
            remaining_groups = remaining_groups.drop(sim_groups.index)
            cluster_id += 1
        return remaining_groups
    
    def _second_round_clustering(self, remaining_groups, clusters):
        """
        part of basic clustering algorithm - not complete
        """
        remaining_copy = remaining_groups.copy()
        i = 0
        cluster_id = len(clusters)
        while not remaining_copy.empty:
            group = remaining_groups.iloc[i]
            if group.get_id() not in remaining_copy:
                i += 1
                continue
            remaining_copy.drop(group.get_id(), inplace=True)
            fp = group.basic_fingerprint('MACCS')
            similarities = remaining_copy.apply(self._compute_similarity, args=(fp,))
            sim_groups = similarities[similarities >= 0.8]
            clusters.append(tuple(remaining_copy.loc[sim_groups.index].index))
            remaining_copy.drop(sim_groups.index, inplace=True)
            i += 1
            cluster_id += 1
            
    def _diversify(self, fps, diverse_groups=100):
        """
        part of basic clustering algorithm - not complete
        """       
        picker = MaxMinPicker()
        nfps = fps.size
        def distij(i, j, fps=fps.tolist()):
            return 1-DataStructs.DiceSimilarity(fps[i],fps[j])
        pick_indices = picker.LazyPick(distij, nfps, diverse_groups, seed=23)
        return list(pick_indices)
    
    def _compute_similarity(self, group, fp2):
        """
        part of basic clustering algorithm - not complete
        """
        fp1 = group.basic_fingerprint('MACCS')
        return DataStructs.DiceSimilarity(fp1, fp2)
    
    def _direct_sub_pattern(self, sub):
        """
        create the SMARTS pattern to identify a direct sub
        
        this converts the SMILES version of the substituent into
        a SMARTS pattern that makes sure the substituent is terminal,
        is only connected to the ring via a single bond, and has no
        ring segments itself

        Parameters
        ----------
        sub : str
            the SMILES description of the substituent

        Returns
        -------
        RDKit.Mol
            the RDKit.Mol object representing the substituent 
            SMARTS pattern

        """
        sub = Chem.MolFromSmiles(sub)
        sub = Chem.MolToSmarts(sub)
        sub = re.sub(r'\[(.+?)\]', r'[\1;!R]', sub)
        sub = re.sub(r'\[(.+?)\](?!.+\[(.+?)\])', r'[\1;D1]', sub)
        direct_sub = re.sub(r'\[(.+?)\]', r'[\1;$([\1]-[#6;R])]', sub, count=1)

        return Chem.MolFromSmarts(direct_sub)
        
    def _get_direct_subs(self):
        """
        search all Fragments for direct substituents to create a repository
        
        extract the direct substituents from each Fragment object to build
        a list of known direct substituents. This list is used when reducing
        Fragments to their core

        Returns
        -------
        direct_subs : dict
            {subtituent SMILES : RDKit.Mol representation}

        """
        regen = self._config.get_regen('fragments')
        direct_subs_dir = self._config.get_directory('direct_subs')
        direct_subs_file = os.path.join(direct_subs_dir, 'ds.json')
        if not regen:
            try:
                direct_subs = self._fh.load_from_text(direct_subs_file)
                return direct_subs
            except FileNotFoundError:
                pass
        fragments = self.fd.clean_frags()
        direct_subs = set()

        for frag in fragments:
            subs = frag.get_direct_subs()
            direct_subs.update(subs)
        #sometimes the direct subs are connected to the
        #ring via an alkyl carbon e.g. -COH or -CBr
        #need to capture these too
        extended_subs = ['C' + sub for sub in direct_subs]
        direct_subs = list(direct_subs) + extended_subs
        direct_subs = {sub: self._direct_sub_pattern(sub) 
                             for sub in direct_subs}
        return direct_subs
    
    def _create_taa_group(self, groups_dict, mol_group_link, id_):
        """
        triarylamine needs to be created manually since its not a Fragment

        this method is called at the very end of the main Group
        creation procedure because triarylamine does not get picked
        up normally. All of the input arguments are simply information
        passed on from the grouping procedure
        
        Parameters
        ----------
        groups_dict : dict
            existing Group objects already created so can add the Group
            created here to this
        mol_group_link : list
            existing array of [mol_id, group_id] lists to which the IDs of
            Molecules of containing triarylamine will be added
        id_ : int
            the ID that needs to be assigned to the taa group

        Returns
        -------
        None.

        """
        taa_SMILES = self.PATTERNS['triarylamine']
        taa_group = Group(taa_SMILES, id_)
        taa_group.is_taa = True
        groups_dict[taa_SMILES] = taa_group
        group_mol_ids = self._taa_parents().index
        new_link = [[group_mol_id, taa_group.get_id()] for group_mol_id in group_mol_ids]
        mol_group_link.extend(new_link)     
        
    def _taa_parents(self):
        """
        find the Molecules that contain triarylamine
        """
        taa_SMARTS = self.PATTERNS['triarylamine']
        patt = Chem.MolFromSmarts(taa_SMARTS)
        mols = self.md.get_molecules()
        has_taa = mols.apply(lambda x: x.has_pattern(patt))
        return mols[has_taa]
    
    def _mol_comp_data(self):
        """
        return series with the lambda_max values of each Molecule
        """
        mols = self.md.get_molecules()
        self.md.set_comp_data()
        comp_df = mols.apply(lambda x: x.get_lambda_max())
        return comp_df
    
    def _has_comp_data(self, frag_id, comp_df):
        """
        make sure a fragment belongs to parent with absorption data
        
        if a fragment does not have any parents with valid sTDA data
        then this fragment is likely non-organic or in some way
        corrupted

        Parameters
        ----------
        frag_id : int
            ID of the Fragment to check
        comp_df : pandas.Series
            series with the lambda_max values of every Molecule

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        frag_mols = self.fd.get_frag_mols(frag_id)
        comp_tmp = comp_df[frag_mols.index]

        return (comp_tmp > 0).any()
    
    def _basic_clustering(self):
        """
        basic cluster algorithm - not yet complete
        """
        groups = self.get_groups()
        
        mols_per_group = groups.apply(lambda x: self.group_mol_count(x.get_id()))
        common_indices = mols_per_group[mols_per_group >= 3].index
        common_groups = groups.loc[common_indices]

        fps = groups.apply(lambda x: x.basic_fingerprint('MACCS'))
        diverse_indices = self._diversify(fps.loc[common_indices])
        diverse_groups = common_groups.iloc[diverse_indices]
        
        clusters = []
        remaining_groups = self._first_round_clustering(groups, diverse_groups, clusters)
        mols_per_group = mols_per_group.loc[remaining_groups.index]
        mols_per_group.sort_values(ascending=True, inplace=True)
        remaining_groups = remaining_groups[mols_per_group.index]
        self._second_round_clustering(remaining_groups, clusters)
        
        return clusters
            
    def set_occurrence(self):
        """ 
        calculate the number of Molecules that contain each Group
        """
        groups = self.get_groups()
        for group in groups:
            if group.occurrence:
                break
            if group.is_taa:
                group.occurrence = self._taa_parents().size
                continue
            group_mols = self.get_group_mols(group.get_id())
            group.occurrence = group_mols.size

    def get_group(self, group_id):
        """ 
        return a particular Group object
        """
        groups = self.get_groups()
        return groups.loc[group_id]
        
    def get_groups(self, regen=False):
        """
        create a series of Group objects
        
        the Group objects are created by reducing Fragment
        objects down to their core. This method only
        acts on filtered Fragments, that is Fragments that
        possess traits of being a core group (see Fragment class
        documentation)

        Parameters
        ----------
        regen : boolean, optional
            ignore pickle files or in-memory data and regenerate
            the Groups from scratch. The default is False.

        Returns
        -------
        pandas.Series
            a series of Group objects

        """
        if not self._groups.empty and not regen:
            return self._groups

        pickled_file = self._config.get_directory('pickle') +  'groups.pickle'
   
        if os.path.isfile(pickled_file) and not regen:
            groups = self._fh.load_from_pickle(pickled_file)
            self._groups = groups
            return groups

        direct_subs = self._get_direct_subs()
        frags = self.fd.clean_frags()

        groups_dict = {}
        #will be used to store the IDs of molecules that contain
        #each group
        mol_group_link = []
        comp_df = self._mol_comp_data()
        id_ = 0
        for frag in frags:

            if not self._has_comp_data(frag.get_id(), comp_df):
                continue
            core = frag.convert_to_group(direct_subs)
            try:
                group = groups_dict[core]
            except KeyError:
                group = Group(core, id_)
                groups_dict[core] = group
                id_ += 1
            frag.set_group(group.get_id())

            group_mols = self.fd.get_frag_mols(frag.get_id())
            group_mol_ids = group_mols.index
            new_link = [[group_mol_id, group.get_id()] for group_mol_id in group_mol_ids]
            mol_group_link.extend(new_link)
        self._create_taa_group(groups_dict, mol_group_link, id_)
        groups = self._create_entity_df(groups_dict)
        self._linker.set_link_table('mol_group', mol_group_link)
        self._groups = groups
        self.set_occurrence()
        self.pickle(self._groups, 'groups')
        return groups
    
    def get_group_frags(self, group_id):
        
        frags = self.fd.clean_frags()
        group_frags = frags.apply(lambda x: x.get_group() == group_id)
        group_frags = frags.loc[group_frags]
        return group_frags
    
    def get_group_cluster(self, group_id):
        
        cgm = self.get_group_clusters()
        return cgm[cgm['group_id'] == group_id]['cluster_id'].iloc[0]
    
    def get_group_clusters(self, cutoff=0.2, similarity='dice', 
                           fp_type='MACCS', recluster=False, basic=False):
        
        if not recluster:
            try:
                return self._cgm
            except AttributeError:
                pass
        groups = self.get_groups()
        if not basic:
            sim_func = self.SIMILARITIES[similarity]
            fps = groups.apply(lambda x: x.basic_fingerprint(fp_type)).tolist()
            num_fps = groups.size
            dists = []
            for i in range(1, num_fps):
                sims = sim_func(fps[i], fps[:i])
                dists.extend([1-x for x in sims])
            clusters = Butina.ClusterData(dists, num_fps, cutoff, isDistData=True)
        else:
            clusters = self._basic_clustering()
        
        cgm = []
        for i, cluster in enumerate(clusters):
            clust_groups = groups.iloc[list(cluster)]
            clust_groups = clust_groups.apply(lambda x: x.set_cluster(i))
            group_idx = clust_groups.index
            cluster = [i]*len(group_idx)
            cgm.extend(zip(cluster, group_idx))
        cgm = pd.DataFrame(cgm)
        cgm.columns = ['cluster_id', 'group_id']
        self._cgm = cgm
        return cgm
    
    def get_cluster_groups(self, cluster_id):
        
        cgm = self.get_group_clusters()
        return cgm[cgm['cluster_id'] == cluster_id]['group_id']    
    
    def get_mol_groups(self, mol_id):
        lt = self._linker.get_link_table(f'mol_group')
        group_ids = lt[mol_id == lt['mol_id']][f'group_id']
        groups = self.get_groups().loc[group_ids.values]
        return groups
    
    def get_group_mols(self, group_id=None, group_ids=[]):
        lt = self._linker.get_link_table(f'mol_group')
        if group_ids:
            mol_ids = lt[lt['group_id'] == group_ids]['mol_id'].unique
        else:
            mol_ids = lt[group_id == lt['group_id']]['mol_id']
        mols = self.md.get_molecules().loc[mol_ids.values]
        return mols
    
    def find_groups_with_pattern(self, pattern):
        patt = self.PATTERNS[pattern]
        patt = Chem.MolFromSmarts(patt)
        groups = self.get_groups()
        has_patt = groups.apply(lambda x: x.has_pattern(patt))
        return groups[has_patt]


class ChainData(EntityData):
    
    def __init__(self, mol_data, frag_data, group_data):
        self.md, self.fd, self.gd = mol_data, frag_data, group_data
        self._configure()
        self._set_up()

    def _set_up(self):
        self._substituents = pd.Series([], dtype=object)
        self._bridges = pd.Series([], dtype=object)
        
    def _create_sub_objects(self, subs, groups, existing_subs):
        obj_subs = []

        for sub in subs:  
            mol = Chem.MolFromSmiles(sub)
            sub_flag = True
            for group in groups:
                if mol.HasSubstructMatch(group):
                    sub_flag = False
                    break
            if not sub_flag:
                continue

            test_sub = re.sub('\[*[0-9]*\*+\]*', '[*]', sub)
            try:
                sub_obj = existing_subs[test_sub]
                obj_subs.append(sub_obj)
            except KeyError:
                id_ = len(existing_subs)
                new_sub = Substituent(test_sub, id_)
                existing_subs[test_sub] = new_sub
                obj_subs.append(new_sub)
        return obj_subs
    
    def _create_bridge_objects(self, bridges, existing_bridges):
        
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

    def _get_chains(self, mol, rdk_group, other_atoms, match):
        if match.issubset(other_atoms):
            return []
        rm = Chem.ReplaceCore(mol, rdk_group, matches=tuple(match), labelByIndex=True)
        smiles = Chem.MolToSmiles(rm)
        if not smiles:
            return []
        branches = smiles.split('.')
        return branches
    
    def _get_chain_df(self, reset=False):
        if not reset:
            try:
                return self._chain_df
            except AttributeError:
                pass
        my_mols = self.md.clean_mols()
        groups = self.gd.get_groups()
        groups_size = groups.apply(lambda x: x.get_size())
        groups = groups[groups_size.index]
        tmp = []
        for my_mol in my_mols:
            other_atoms = []
            mgs = self.gd.get_mol_groups(my_mol.get_id())
            is_taa = mgs.apply(lambda x: x.is_taa)
            mgs = mgs[~is_taa]
            #important: so that other atoms can be found in the case of being subset of larger group
            sizes = groups_size[mgs.index].sort_values(ascending=True) 
            mgs = mgs.loc[sizes.index]
            matches = [self._get_matches(my_mol.get_rdk_mol(), g.get_rdk_mol(), 
                                         other_atoms) for g in mgs]
            records = [[my_mol.get_id(), my_mol, g, g.get_rdk_mol(), 
                        matches[i], other_atoms[i]] for i, g in enumerate(mgs)]
            tmp.extend(records)
            
        df = pd.DataFrame(tmp, columns=['m_id', 'm', 'g', 'rdk_g','ms', 'oa'])
        self._chain_df = df
        return df
    
    def _get_bridges(self, my_mol, rdk_groups, mol_subs, bridges_dict):
        
        trimmed_mol = my_mol.get_rdk_mol()
        
        for rdk_group in rdk_groups[::-1]:
            while trimmed_mol:
                  tmp = Chem.MolToSmiles(trimmed_mol)
                  trimmed_mol = Chem.ReplaceCore(trimmed_mol, rdk_group, labelByIndex=True)
            trimmed_mol = Chem.MolFromSmiles(tmp)
        smiles = Chem.MolToSmiles(trimmed_mol)
        branches = smiles.split('.')
        branches = [re.sub('\[*[0-9]*\*+\]*', '[*]', branch) for 
                    branch in branches]
        branches = [branch for branch in branches if (not 
                    re.match('^(\[*\*\]*)+$', branch)) and (branch != '')]
        for sub in mol_subs:
            if sub in branches:
                branches.remove(sub)
        patt = Chem.MolFromSmarts(self.PATTERNS['triarylamine'])
        num_taa = my_mol.pattern_count(patt)
        for i in range(0, num_taa):
            try:
                branches.remove('[*]N([*])[*]')
            except ValueError:
                break
        obj_bridges = self._create_bridge_objects(branches, bridges_dict)
        new_records = [[my_mol.get_id(), bridge.get_id()] for bridge in obj_bridges]
        return new_records
    
    def _get_substituents(self, my_mol, my_group, rdk_groups, other_atoms, 
                  matches, mol_subs_map, subs_dict):
        rdk_mol = my_mol.get_rdk_mol()
        rdk_group = my_group.get_rdk_mol()
        for i in range(0, len(matches)):
            chains = self._get_chains(rdk_mol, rdk_group, other_atoms, matches[i])
            if not chains:
                continue
            obj_subs = self._create_sub_objects(chains, rdk_groups, subs_dict)
            new_records = [[my_mol.get_id(), my_group.get_id(), 
                            i, sub.get_id() ] for sub in obj_subs]
            mol_subs_map.extend(new_records)
            
    def _get_matches(self, mol, group, other_atoms):
        g_matches = mol.GetSubstructMatches(group)
        g_matches = [set(match) for match in g_matches]
        try:     
            all_matches = reduce(lambda s1, s2: s1.union(s2), g_matches)
        except TypeError:
            all_matches = {}
        for atoms in other_atoms:
            atoms.update(all_matches)
        other_atoms.append(set())
        return g_matches
    
    def set_sub_occurrence(self):

        subs = self.get_substituents()
        for sub in subs:
            if sub.occurrence:
                break
            sub_mols = self.get_sub_mols(sub.get_id())
            sub.occurrence = sub_mols.size  
            
    def set_bridge_occurrence(self):

        bridges = self.get_bridges()
        for bridge in bridges:
            if bridge.occurrence:
                break
            bridge_mols = self.get_bridge_mols(bridge.get_id())
            bridge.occurrence = bridge_mols.size  
            
    def get_bridge(self, bridge_id):
        bridges = self.get_bridges()
        return bridges.loc[bridge_id]

    def get_bridges(self, regen=False):
        if not self._bridges.empty and not regen:
            return self._bridges
        
        pickled_file = self._config.get_directory('pickle') +  'bridges.pickle'
        if os.path.isfile(pickled_file) and not regen:
            bridges = self._fh.load_from_pickle(pickled_file)
            self._bridges = bridges
            return bridges
        
        subs = self.get_substituents()
        my_mols = self.md.clean_mols()

        df = self._get_chain_df()
        mol_bridge_map = []
        bridges_dict = {}
        lt = self._linker.get_link_table('mol_group_sub')
        for my_mol in my_mols:
            rdk_gs = df[df['m_id'] == my_mol.get_id()]['rdk_g']
            sub_ids = lt[my_mol.get_id() == lt['mol_id']]['group_sub_id']
            mol_subs = list(subs[sub_ids].apply(lambda x: x.smiles))
            new_records = self._get_bridges(my_mol, rdk_gs, mol_subs, bridges_dict)
            mol_bridge_map.extend(new_records)
        self._linker.set_link_table('mol_bridge', mol_bridge_map)
        self._bridges = self._create_entity_df(bridges_dict)
        self.set_bridge_occurrence()
        self.pickle(self._bridges, 'bridges') #must be last line
        return self._bridges
    
    def find_subs_with_pattern(self, pattern):
        patt = Chem.MolFromSmarts(self.PATTERNS[pattern])
        subs = self.get_substituents()
        has_patt = subs.apply(lambda x: x.has_pattern(patt))
        return subs[has_patt]
    
    def get_substituent(self, sub_id):
        subs = self.get_substituents()
        return subs.loc[sub_id]
        
    def get_substituents(self, regen=False):
        if not self._substituents.empty and not regen:
            return self._substituents
        
        pickled_file = self._config.get_directory('pickle') +  'subs.pickle'
   
        if os.path.isfile(pickled_file) and not regen:
            subs = self._fh.load_from_pickle(pickled_file)
            self._substituents = subs
            return subs
        
        mol_subs_map = []
        subs_dict = {}

        df = self._get_chain_df()
        for idx, row in df.iterrows():

            my_mol, group, other_atoms, matches =\
            row['m'], row['g'], row['oa'], row['ms']
            mol_id = my_mol.get_id()
            groups = self.gd.get_mol_groups(mol_id)
            rdk_groups = groups.apply(lambda x: x.get_rdk_mol())
            self._get_substituents(my_mol, group, rdk_groups, other_atoms, 
                           matches, mol_subs_map, subs_dict)
        self._linker.set_link_table('mol_group_sub', mol_subs_map)
        self._substituents = self._create_entity_df(subs_dict)   
        self.set_sub_occurrence()
        self.pickle(self._substituents, 'subs')
        return self._substituents
    
    def get_mol_subs(self, mol_id):
        lt = self._linker.get_link_table(f'mol_group_sub')
        sub_ids = lt[mol_id == lt['mol_id']]['group_sub_id']
        subs = self.get_substituents().loc[sub_ids.values]
        return subs
    
    def get_mol_bridges(self, mol_id):
        lt = self._linker.get_link_table(f'mol_bridge')
        bridge_ids = lt[mol_id == lt['mol_id']]['bridge_id']
        bridges = self.get_bridges().loc[bridge_ids.values]
        return bridges
            
    def get_sub_mols(self, sub_id):
        lt = self._linker.get_link_table(f'mol_group_sub')
        mol_ids = lt[sub_id == lt['group_sub_id']]['mol_id']
        mols = self.md.get_molecules().loc[mol_ids.values]
        return mols
    
    def get_bridge_mols(self, bridge_id):
        lt = self._linker.get_link_table(f'mol_bridge')
        mol_ids = lt[bridge_id == lt['bridge_id']]['mol_id']
        mols = self.md.get_molecules().loc[mol_ids.values]
        return mols
        
def set_up():
    global md, fd, gd, cd
    md = MoleculeData()
    fd = FragmentData(md)
    gd = GroupData(md, fd)
    cd = ChainData(md, fd, gd)     

set_up()