 # -*- coding: utf-8 -*-
"""
Created by: Padraic Flanagan
"""
import os.path
import json
try:
   import cPickle as pickle
except:
   import pickle
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

from pymongo import MongoClient
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Draw
from rdkit.Chem import BRICS
from rdkit import DataStructs
from rdkit import RDLogger

import statsmodels.api as sm
from patsy import dmatrices

from helper import MyConfig as Config
from data import MongoLoader

class ChemBase():

    def __init__(self):
        RDLogger.DisableLog('rdApp.*')
        self.config = Config().params
        self.fp_sim_map = {'tanimoto':'rdk','dice':'morgan'}
        self.get_molecules()

    def get_molecules(self,from_dump = True,save_dump=True):
        
        smiles_file = self.config['base_dir'] + self.config['smiles']
        smiles_df = self.extract_db('smiles')
        
        if from_dump:
            rd_mols = self.load_mols()
        else:
            suppl = Chem.SmilesMolSupplier(smiles_file,delimiter=',',titleLine=0)
            rd_mols = [x for x in suppl]
            if save_dump:
                self.dump_mols(rd_mols)
        rd_mols_df = pd.Series(rd_mols,name='molecule')
        
        comp_df = self.extract_db('comp_uvvis')
        self.mols_df = pd.concat([smiles_df,rd_mols_df],axis=1)
        self.mols_df.dropna()
        self.mols_df = pd.merge(self.mols_df,comp_df,on='smiles')
        
    def dump_mols(self,mols_list):
        pickle_file = self.config['base_dir'] + self.config['pickle']
        with open(pickle_file,'wb') as ph:
            pickle.dump(mols_list,ph)
            
    def load_mols(self):
        pickle_file = self.config['base_dir'] + self.config['pickle']
        with open(pickle_file,'rb') as ph:
            mols_list = pickle.load(ph)
        return mols_list
            
    def extract_db(self,record_type):
        db_broker = Mongo()
        return db_broker.resolve_query(record_type)

    def draw_molecules(self):
        Draw.MolToFile(self.mols[20],'cdk2_mol1.o.png')
        
    def similarities(self,sim_metric):
            fp_type = self.fp_sim_map[sim_metric]
            fps = self.get_fingerprints(fp_type)
            self.mols_df['fingerprint'] = fps
            self.sims_df = self.get_similarities(sim_metric)
                
    def compare_sim_uvvis(self,smi1,smi2):
        mol1 = self.mols_df.loc[self.mols_df['smiles']==smi1]
        mol2 = self.mols_df.loc[self.mols_df['smiles']==smi2]
        lambda1 = mol1['lambda'].values[0]
        lambda2 = mol2['lambda'].values[0]
        osc1 = mol1['strength'].values[0]
        osc2 = mol2['strength'].values[0]
        return lambda1,lambda2,osc1,osc2
          
    def get_fingerprints(self,fp_type):   
        if fp_type == 'rdk':
            fps = [Chem.RDKFingerprint(row['molecule']) for idx,row in self.mols_df.iterrows()]
        elif fp_type == 'morgan':
            fps = [Chem.GetMorganFingerprint(row['molecule'],2) for idx,row in self.mols_df.iterrows()]
        return pd.Series(fps,name='fingerprint')

    def get_similarities(self,metric):
        sims_df = pd.DataFrame({'mol1':[],'mol2':[],'similarity':[]})
        
        for idx_outer,row_outer in self.mols_df.iterrows():
            combinat_df = self.mols_df.loc[idx_outer+1:]
            if combinat_df.empty:
                continue
            sim = combinat_df.apply(lambda row: self.calc_sim(row_outer['fingerprint'],row['fingerprint'],metric),axis=1)
            temp_df = pd.DataFrame({'mol1':[row_outer['smiles']]*len(combinat_df),
                       'mol2':combinat_df['smiles'],
                       'similarity':sim})
            sims_df = pd.concat([sims_df,temp_df],ignore_index=True)
        return sims_df
            
    def calc_sim(self,fp1,fp2,metric):
        if metric == 'tanimoto':  
            sim = DataStructs.FingerprintSimilarity(fp1,fp2)
        elif metric == 'dice':
            sim = DataStructs.DiceSimilarity(fp1,fp2)
        return sim
    
    def analyse_sim_uvvis(self,threshold=0.5):
        temp = self.sims_df.loc[self.sims_df['similarity']>threshold]
        for idx,row in temp.iterrows():
            lam1,lam2,osc1,osc2 = self.compare_sim_uvvis(row['mol1'],row['mol2'])
            print('similarity:',row['similarity'],'\nlambda1: ',lam1,'\nlambda2:',lam2,
                  '\nsmiles1: ',row['mol1'],'\nsmiles2: ',row['mol2'],
                  '\noscillator1: ',osc1,'\noscillator2: ',osc2,'\n')
            
    def fragment(self,mol,from_dump=True):
        frag_dict = None
        if from_dump:
            pickle_file = self.config['base_dir'] + self.config['frag_pickle']
            with open(pickle_file,'rb') as ph:
                frag_dict = pickle.load(ph)
        frags = Fragment.fragment_mol(mol,frag_dict)
        mol.SetIntProp('leaf_nodes',len(frags))
        return frags
            
    def create_fragments(self,save_dump=False):
        self.mols_df['fragments'] = self.mols_df.molecule.apply(self.fragment)
        if save_dump:
            frag_dict = {}
            for f_list in self.mols_df.fragments:
                frag_dict[f_list[0].parent_smiles] = [frag.smiles for frag in f_list]
            print(frag_dict)
            self.dump_fragments(frag_dict)
                
    def dump_fragments(self,frag_dict):
        pickle_file = self.config['base_dir'] + self.config['frag_pickle']
        with open(pickle_file,'wb') as ph:
            pickle.dump(frag_dict,ph)
        
    def frags_metrics(self):
        totals_df = self.mols_df.molecule.apply(lambda x: x.GetIntProp('leaf_nodes'))
        mean = totals_df.mean()
        median = totals_df.median()
        single_core = totals_df[totals_df.isin([1])].count()
        metrics_dict = {'mean':mean,'median':median,'singles':single_core}
        return metrics_dict      
                
    def linear_sim_lambda_diff(self):
        lambdas = self.mols_df[['smiles','lambda']]
        lambdas = lambdas.set_index('smiles')
        lambdas = lambdas.rename(columns={'lambda':'lam1'})
        tmp = self.sims_df
        tmp = tmp.join(lambdas,on='mol1')
        lambdas = lambdas.rename(columns={'lam1':'lam2'})
        tmp = tmp.join(lambdas,on='mol2')
        tmp['difference'] = (tmp['lam1']-tmp['lam2']).abs()
        tmp = tmp.loc[tmp.difference <= 500]
        self.plotting_df = tmp
        y, X = dmatrices('difference ~ similarity', data=tmp, return_type='dataframe')
        mod = sm.OLS(y, X)
        self.reg_res = mod.fit()
        print(self.reg_res.summary())
        
        
class Fragment():

    def __init__(self,parent,smiles):
        self.parent_smiles = parent
        self.smiles = smiles
        
    @classmethod
    def fragment_mol(cls,mol,frag_dict=None,keepAllNodes=False):
        parent = Chem.MolToSmiles(mol)
        if frag_dict:
            frag_smiles = frag_dict[parent]
        else:
            frag_smiles = list(BRICS.BRICSDecompose(mol,keepNonLeafNodes=keepAllNodes))
        frags = []
        for fs in frag_smiles:
            fragment = cls(parent,fs)
            frags.append(fragment)
        return frags


if __name__ == '__main__':
    startTime = datetime.now()
    base = ChemBase()
    # base.create_fragments(save_dump=False)
    # print(datetime.now() - startTime)
