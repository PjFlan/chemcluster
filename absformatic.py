#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 13:25:29 2020

@author: padraicflanagan
"""
import os
from copy import deepcopy
import random

from rdkit import DataStructs

from DAL import MoleculeData, FragmentData, GroupData, ChainData
from metrics import FragmentMetric, MoleculeMetric, GroupMetric
from fingerprint import AbsFingerprint

from helper import MyConfig, MyLogger, MyFileHandler
from helper import FingerprintNotSetError
from drawing import draw_mols_canvas, draw_entities

class Absformatic:
    
    def __init__(self):
        self.md = MoleculeData()
        self.fd = FragmentData(self.md)
        self.gd = GroupData(self.md, self.fd)
        self.cd = ChainData(self.md, self.fd, self.gd)
        self.af = AbsFingerprint(self.md, self.fd, self.gd, self.cd)
        self.mm = MoleculeMetric(self.md, self.fd, self.gd, self.cd, self.af)
        self.fm = FragmentMetric(self.md, self.fd, self.gd, self.cd, self.af)
        self.gm = GroupMetric(self.md, self.fd, self.gd, self.cd, self.af)
        self._configure()
        
    def _configure(self):
        self._logger = MyLogger().get_child(type(self).__name__)
        self._fh = MyFileHandler()
        self._config = MyConfig()
        
    def set_up(self):
        regen =  self._config.get_regen('grouping')
        self.DEV_FLAG = self._config.get_flag('dev')
        self.mols = self.md.get_molecules(regen)
        self.frags = self.fd.get_fragments(regen)
        self.groups = self.gd.get_groups(regen)
        self.subs = self.cd.get_substituents(regen)
        self.bridges = self.cd.get_bridges(regen)
        self.md.set_comp_data()
    
    def get_mol_fingerprint(self, mol_id, regen=False):
        
        mol = self.mols.loc[mol_id]
        if not regen:
            try:
                abs_fp = mol.get_abs_fp()
                print(abs_fp)
                return abs_fp
            except FingerprintNotSetError:  
                pass
        abs_fp = self.af.fingerprint_mol(mol_id)
        print(abs_fp)
        return abs_fp
    
    def get_group_cluster(self, group_id):
        return self.gd.get_group_cluster(group_id)
    
    def setup_query_draw(self, folder):
        outdir = os.path.join(self._config.get_directory('images'), 
                      f'{folder}')
        if self._config.use_tmp():
            outdir = self._config.get_directory('tmp')
        if os.path.exists(outdir):
            self._fh.clean_dir(outdir)
        return outdir
    
    def get_mol_fingerprints(self, regen=False):
        try:
            return self.fps
        except AttributeError:
            self.fps = self.mols.apply(
                lambda x: self._fingerprint_mol(x.get_mol_id()))  
        return self.fps
        
    def get_molecule(self, mol_id):

        return self.mols.loc[mol_id]

    def draw_entity_mols(self, ent, id_):
        
        self.mm.draw_entity_mols(ent_name=ent, ent_id=id_)
        
    def mols_same_group(self, occurrence=False, draw=False, query='', pattern=''):
    
        mols = self.md.clean_mols()
        sim_mols = self.af.mols_same_group(occurrence, pattern, exclude_benz=True)
        folder = f'mols_same_group_{occurrence}_{query}_{pattern}'
        outdir = self.setup_query_draw(folder)
        if draw:
            for key, mol_ids in sim_mols.items():
                
                mols_tmp = mols[mol_ids]
                legends = mols_tmp.apply(lambda x: x.get_legend(self.DEV_FLAG))
                draw_mols_canvas(mols_tmp, legends, outdir, suffix=key, clean_dir=False)
        if query:
            return mols[sim_mols[query]]
        return sim_mols

            
    def group_fp_query(self, g_query=[], complete=True):
        occurrence = False
        if isinstance(g_query[0], tuple):
            occurrence = True
        g_query = set(g_query)
        mols_with_fp = self.af.group_fp_query(g_query, complete, occurrence)
        return mols_with_fp
    
    def mols_same_except_one(self, except_group=None, draw=False, query=''):
        
        mol_sets = self.af.mols_same_except_one(except_group)
        mols = self.md.clean_mols()
        folder = f'mols_same_expt_one_{except_group}'
        if not draw:
            if query:
                return mols[list(mol_sets[query])]
            return mol_sets
        
        outdir = self.setup_query_draw(folder)
        for key, mol_ids in mol_sets.items():
            m_ids = list(mol_ids)
            mols_tmp = self.mols.loc[m_ids]
            legends = mols_tmp.apply(lambda x: x.get_legend(self.DEV_FLAG))
            if len(mols_tmp) <= 1:
                continue
            if key == '2':
                continue
            draw_mols_canvas(mols_tmp, legends, outdir, suffix=key, clean_dir=False)
        if query:
            return mols[mol_sets[query]]
        return mol_sets
            
    def similarity_report(self, mol_ids):
        report, table = self.mm.similarity(mol_ids)
        print(table)
        
    def draw_cluster_mols(self, cluster_id):
        group_idx = self.gd.get_cluster_groups(cluster_id)
        groups = self.groups[group_idx]

        mols = self.gd.get_groups_mols(groups.index)
        mg_dir = os.path.join(self._config.get_directory('images'),
                                     f'cluster_{cluster_id}_mols/')
        legends = mols.apply(lambda x: x.get_legend(self.DEV_FLAG))
        draw_mols_canvas(mols, legends, mg_dir, suffix='', start_idx=0)
            
    def draw_groups(self, from_idx=0, to_idx=200):
        
        groups = self.gd.get_groups()
        g_dir = os.path.join(self._config.get_directory('images'), f'groups')
        draw_entities(groups, g_dir, from_idx, to_idx)
        
    def draw_mols(self, mol_ids=[], img_type='png'):
        mols = self.mols[mol_ids]
        legends = mols.apply(lambda x: x.get_legend(self.DEV_FLAG))
        folder = os.path.join(self._config.get_directory('images'), 'mols_rpt')
        draw_mols_canvas(mols, legends, folder, img_type=img_type)
        
    def draw_fragments(self, from_idx, to_idx):

        frags = self.fd.get_fragments()
        frag_dir = os.path.join(self._config.get_directory('images'),'fragments')
        draw_entities(frags, frag_dir, from_idx, to_idx)        
    
        
    def draw_clusters(self, clust_nums=None, singletons=False, from_idx=0, to_idx=200):
        
        cgm = self.gd.get_group_clusters()
        if not clust_nums:
            clust_nums = range(0, cgm['cluster_id'].max() + 1)
        for clust_num in clust_nums[23:24]:
            group_indices = cgm[clust_num == cgm['cluster_id']]['group_id']
            groups_tmp = self.groups[group_indices]
            if (not singletons) and groups_tmp.size == 1:
                continue
            cluster_dir = os.path.join(self._config.get_directory('images'),f'clusters/cluster_{clust_num}/')
            draw_entities(groups_tmp, cluster_dir, from_idx, to_idx)
                
    def draw_group_mols(self, group_id, from_idx=0, to_idx=200):

        self.md.set_comp_data()
        group_mols = self.md.get_group_mols(group_id)
        group = str(group_id)

        legends = group_mols.apply(lambda x: x.get_legend(self.DEV_FLAG))
        folder = os.path.join(self._config.get_directory('images'), 
                          f'group_{group}_mols/')
        draw_mols_canvas(group_mols, legends, folder, start_idx=from_idx)

    def get_groups_with_pattern(self, pattern, draw=False):
        groups = self.gd.get_groups_with_pattern(pattern)
        folder = f'groups_with_{pattern}'
        outdir = self.setup_query_draw(folder)
        if draw:
            legends = groups.apply(lambda x: f'{x.get_id()}')
            draw_mols_canvas(groups, legends, outdir)
        return groups
        
    def get_subs_with_pattern(self, pattern, draw=False):
        subs = self.cd.get_subs_with_pattern(pattern)
        folder = f'subs_with_{pattern}'
        outdir = self.setup_query_draw(folder)
        if draw:
            legends = subs.apply(lambda x: f'{x.get_id()}')
            draw_mols_canvas(subs, legends, outdir)
        return subs
        
    def mols_with_fp(self, fp, draw=False):
        mols = self.af.mols_with_fp_general(fp)
        folder = f'mols_with_fp_{fp}'
        outdir = self.setup_query_draw(folder)
        if draw:
            legends = mols.apply(lambda x: x.get_legend(self.DEV_FLAG))
            draw_mols_canvas(mols, legends, outdir)
        return mols
    
    def mols_avg_sim(self, mols, fp_type='morgan', metric='tanimoto'):

        sims = 0
        cnt = 0
        for i, mol in enumerate(mols.tolist()):
            if i == len(mols) - 1:
                break
            for mol_c in mols[i+1:]:
                report, table = self.mm.similarity(mols=[mol, mol_c])
                sim = report[f'{fp_type}_{metric}']
                sims += sim
                cnt += 1
        return sims/cnt
    
    def large_comp_diff(self, mols, lam_thresh=75, osc_thresh=0.6):
        lambdas = mols.apply(lambda x: x.lambda_max)
        osc = mols.apply(lambda x: x.strength_max)
        lam_diff = lambdas.max() - lambdas.min()
        osc_diff = osc.max() - osc.min()

        if lam_diff > lam_thresh or osc_diff > osc_thresh:
            return True
        return False
    
    def same_fp_comp_diff(self, draw=False):
        mol_subsets = self.mols_same_group(occurrence=True)
        keep = []
        folder = f'same_fp_comp_diff'
        outdir = self.setup_query_draw(folder)
        for fp, mol_ids in mol_subsets.items():
            mols_tmp = self.mols[mol_ids]
            if self.large_comp_diff(mols_tmp):
                keep.append(fp)
                if draw:
                    legends = mols_tmp.apply(lambda x: x.get_legend(self.DEV_FLAG))
                    draw_mols_canvas(mols_tmp, legends, outdir, suffix=fp, clean_dir=False)
        return {key: mol_subsets[key] for key in keep}
        
    def diff_topology(self, draw=False, large_comp_diff=False):
        mol_subsets = self.af.mols_diff_topology()
        folder = f'diff_topology_{large_comp_diff}'
        outdir = self.setup_query_draw(folder)
        if draw:
            for fp, mol_ids in mol_subsets.items():
                mols_tmp = self.mols[mol_ids]
                if not large_comp_diff or self.large_comp_diff(mols_tmp):
                    legends = mols_tmp.apply(lambda x: x.get_legend(self.DEV_FLAG))
                    draw_mols_canvas(mols_tmp, legends, outdir, suffix=fp, clean_dir=False)
        return mol_subsets
    
    def group_fp_scatter(self, save_as=None):
        df = self.gm.group_fp_scatter(save_as=save_as)
        return df
    
    def same_except_sub(self, sub_id, draw=False):
        mol_subsets = self.af.same_fp_except_sub(sub_id)
        folder = f'same_except_{sub_id}'
        outdir = self.setup_query_draw(folder)
        if draw:
            for fp, mol_ids in mol_subsets.items():
                mols_tmp = self.mols[mol_ids]
                legends = mols_tmp.apply(lambda x: x.get_legend(self.DEV_FLAG))
                draw_mols_canvas(mols_tmp, legends, outdir, suffix=fp, clean_dir=False)               
        return mol_subsets
        
    def fp_avg_sim(self, fp_type='rdk', metric='dice'):
        
        COMPARISONS = 5
        random.seed(30)
        self.md.set_comp_data()
        clean_mols = self.md.clean_mols()
        fps = clean_mols.apply(lambda x: x.basic_fingerprint(fp_type))
        lambdas = clean_mols.apply(lambda x: x.lambda_max)
        num_mols = clean_mols.size
        total_fp = 0
        total_lam = 0
        if metric == 'dice':
            sim_func = DataStructs.DiceSimilarity
        else:
            sim_func = DataStructs.FingerprintSimilarity
            
        for mol in clean_mols:
            fp_sum = 0
            lam_sum = 0
            mol_fp = fps.loc[mol.get_id()]
            mol_lam = lambdas.loc[mol.get_id()]
            rand_indices = random.sample(range(0, num_mols), COMPARISONS)
            
            for idx in rand_indices:
                rand_fp = fps.iloc[idx]
                rand_lam = lambdas.iloc[idx]
                left = min(mol_lam, rand_lam)
                right = max(mol_lam, rand_lam)
                vals_btwn = lambdas[lambdas.between(left, right)].size
                lam_sim = (num_mols - vals_btwn)/num_mols
                fp_sim = sim_func(mol_fp, rand_fp)
                fp_sum += fp_sim
                lam_sum += abs(fp_sim - lam_sim)

                
            total_fp += (fp_sum/COMPARISONS)
            total_lam += (lam_sum/COMPARISONS)
        return total_fp/num_mols, total_lam/num_mols
                
if __name__ == '__main__':
    am = Absformatic()
    am.set_up()
    #am.mols_same_group()
