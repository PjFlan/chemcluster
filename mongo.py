# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import json
import os.path

import pandas as pd
import matplotlib.pyplot as plt

from pymongo import MongoClient
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Draw
from rdkit import DataStructs
from rdkit import RDLogger

import statsmodels.api as sm
from patsy import dmatrices

class Config():

    CONF_FILE = 'config.json'

    def __init__(self):
        self.pandas_config()
        with open(self.CONF_FILE) as config_file:
            self.params = json.load(config_file)
            
    def pandas_config(self):
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)

class Mongo():

    def __init__(self):
        self.client = MongoClient()
        self.config = Config().params
        self.db_name = self.config['load_db']
        self.coll_name = self.config['load_collection']
        self.connect()

    def connect(self):
        self.db = self.client[self.db_name]
        self.collection = self.db[self.coll_name]
        
    def resolve_query(self,record_type):
        if record_type == 'smiles':
            return self.get_smiles()
        elif record_type == 'comp_uvvis':
            return self.get_comp_uvvis()
        
    def get_smiles(self):
        smiles = []
        smiles_file = self.config['base_dir'] + self.config['smiles']
        if not os.path.isfile(smiles_file):
            smiles = self.collection.distinct("PRISTINE.SMI")
            self.save_to_file(smiles,smiles_file)
        else:
            with open(smiles_file) as infile:
                for smi in infile:
                    smiles.append(smi[:-2])
        smiles_df = pd.Series(smiles,name='smiles')
        return smiles_df
    
    def get_comp_uvvis(self):
        cursor = self.collection.find({},{
            '_id':0,
            'PRISTINE.SMI':1,
            'FILTERED.orca.excited_states.orbital_energy_list':{'$slice':1},
            'FILTERED.orca.excited_states.orbital_energy_list.amplitude':1,
            'FILTERED.orca.excited_states.orbital_energy_list.oscillator_strength':1
        })
        
        lambdas = []
        strengths = []
        smiles = []
        for record in cursor:
            try:
                smi = record['PRISTINE'][0]['SMI']
                lambda_comp = record['FILTERED'][0]['orca'][0]['excited_states']['orbital_energy_list'][0]['amplitude']
                osc_strength = record['FILTERED'][0]['orca'][0]['excited_states']['orbital_energy_list'][0]['oscillator_strength']
            except KeyError:
                continue
            smiles.append(smi)
            lambdas.append(lambda_comp)
            strengths.append(osc_strength)
        return pd.DataFrame({'smiles':smiles,'lambda':lambdas,'strength':strengths})
    
    def save_to_file(self,records,file):
        with open(file,'w') as outfile:
            for i in records:
                outfile.write(i + ',\n')

class ChemBase():

    def __init__(self):
        RDLogger.DisableLog('rdApp.*')
        self.config = Config().params
        self.fp_sim_map = {'tanimoto':'rdk','dice':'morgan'}
        self.get_molecules()

    def get_molecules(self):
        smiles_file = self.config['base_dir'] + self.config['smiles']
        smiles_df = self.extract_db('smiles')
        suppl = Chem.SmilesMolSupplier(smiles_file,delimiter=',',titleLine=0)
        rd_mols = [x for x in suppl]
        rd_mols_df = pd.Series(rd_mols,name='molecule')
        comp_df = self.extract_db('comp_uvvis')

        self.mols_df = pd.concat([smiles_df,rd_mols_df],axis=1)
        self.mols_df.dropna()
        self.mols_df = pd.merge(self.mols_df,comp_df,on='smiles')
        self.mols_df = self.mols_df[:600]
        
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
            
    def linear_sim_lambda_diff(self):
        lambdas = self.mols_df[['smiles','lambda']]
        lambdas = lambdas.set_index('smiles')
        lambdas = lambdas.rename(columns={'lambda':'lam1'})
        temp = self.sims_df
        temp = temp.join(lambdas,on='mol1')
        lambdas = lambdas.rename(columns={'lam1':'lam2'})
        temp = temp.join(lambdas,on='mol2')
        temp['difference'] = (temp['lam1']-temp['lam2']).abs()
        temp = temp.loc[temp.difference <= 500]
        self.plotting_df = temp
        y, X = dmatrices('difference ~ similarity', data=temp, return_type='dataframe')
        mod = sm.OLS(y, X)
        self.reg_res = mod.fit()
        print(self.reg_res.summary())
        
if __name__ == '__main__':
    base = ChemBase()
    base.similarities('tanimoto')
