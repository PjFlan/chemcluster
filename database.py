#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 13:30:31 2020

@author: padraicflanagan
"""
import os

import pandas as pd
from pymongo import MongoClient

from helper import MyConfig, MyLogger, MyFileHandler
from helper import MyConfigParamError, NoLinkTableError

class Mongo:

    def __init__(self):
        self._configure()
        
    def _configure(self):
        self._logger = MyLogger().get_child(type(self).__name__)
        self._fh = MyFileHandler()
        self._config = MyConfig()

    def _connect(self, conn_info=None):
        self._client = MongoClient()
        self._db = self._client[conn_info["database"]]
        self._collection = self._db[conn_info["collection"]]
        
    def _close_connection(self):
        self._client.close()
        
class MongoLoad(Mongo):
    
    def __init__(self):
        super().__init__()
        self._use_db = self._config.get_flag('db')
        self._conn_info = self._config.get_db_source()
 
    def __enter__(self): 

        if self._use_db:
            self._connect(self._conn_info)
        return self
      
    def __exit__(self, exc_type, exc_value, exc_traceback): 
        if self._use_db:
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
        regen = self._config.get_regen('comp')
        
        try:
            comp_file = self._config.get_directory('comp')
        except MyConfigParamError as e:
            self._logger.warning(e)
            
        if os.path.isfile(comp_file) and not regen:
            comp_json = self._fh.load_from_json(comp_file)
            comp = comp_json['comp']
            smiles = comp_json['smiles']
        else:
            cursor = self._collection.find({}, {
                '_id':0,
                'PRISTINE.SMI':1,
                'FILTERED.orca.excited_states.orbital_energy_list':{'$slice':3},
                'FILTERED.orca.excited_states.orbital_energy_list.amplitude':1,
                'FILTERED.orca.excited_states.orbital_energy_list.oscillator_strength':1
            })
            
            comp = []
            smiles = []
            for record in cursor:
                try:
                    smi = record['PRISTINE'][0]['SMI']
                    comp_tmp = record['FILTERED'][0]['orca'][0]['excited_states']['orbital_energy_list']
                    lam_1, lam_2, lam_3 = comp_tmp[0]['amplitude'], \
                        comp_tmp[1]['amplitude'], comp_tmp[2]['amplitude']
                    osc_1, osc_2, osc_3 = comp_tmp[0]['oscillator_strength'], \
                        comp_tmp[1]['oscillator_strength'], comp_tmp[2]['oscillator_strength']
                except KeyError:
                    continue
                smiles.append(smi)
                osc = [osc_1, osc_2, osc_3]
                lam = [lam_1, lam_2, lam_3]
                comp_dict = {'lambda': lam, 'strength': osc}
                comp.append(comp_dict)
            if comp_file:
                comp_json = {'smiles': smiles, 'comp': comp}
                self._fh.output_to_json(comp_json, comp_file)  
        df = pd.DataFrame({'smiles': smiles, 'comp': comp})
        df = df.set_index('smiles')
        return df
    
    
class LinkTable:
  
    _link_dict = {}
    COLUMNS = {'mol_frag': ['mol_id', 'frag_id'], 
               'mol_group': ['mol_id', 'group_id'],
               'mol_group_sub': ['mol_id', 'group_id', 'instance_id', 'group_sub_id'],
               'mol_bridge': ['mol_id', 'bridge_id']}
     
    def __init__(self):
        self._fh = MyFileHandler()
        self._config = MyConfig()
        self._link_dir = self._config.get_directory("link_tables")
        
    def _array_to_table(self, link_array):
        link_table = pd.DataFrame(link_array).apply(pd.to_numeric, errors='coerce')
        return link_table
    
    def _load_link_from_file(self, filename):
        link_file = os.path.join(self._link_dir, filename)
    
        try:
            link_array = self._fh.load_from_text(link_file)
        except FileNotFoundError:
            raise NoLinkTableError()
        link_array = self._fh.load_from_text(link_file)
        link_table = pd.DataFrame(link_array).apply(pd.to_numeric, errors='coerce')
        return link_table
    
    def _dump_link_to_file(self, filename, link_array):
        link_file = os.path.join(self._link_dir, filename)
        self._fh.output_to_text(link_array, link_file)
        
    def get_link_table(self, name):
        
        try:
            lt = self._link_dict[name]
        except KeyError:
            lt = self._load_link_from_file(f'{name}.txt')
            self._link_dict[name] = lt
            lt.columns = self.COLUMNS[name]
        return lt
    
    def set_link_table(self, name, link_array):
        filename = f'{name}.txt'
        self._dump_link_to_file(filename, link_array)
        lt = self._array_to_table(link_array)
        lt.columns = self.COLUMNS[name]
        self._link_dict[name] = lt
          