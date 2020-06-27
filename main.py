#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 14:40:36 2020
@author: padraicflanagan
"""
from helper import MyConfig,MyLogger,MyFileHandler
from metrics import FragmentMetric,MoleculeMetric
from data import MoleculeData,FragmentData
from rdkit.Chem import AllChem as Chem

my_config = MyConfig()
my_logger = MyLogger().get_root()
my_fh = MyFileHandler()

frag_metric = FragmentMetric()
mol_metric = MoleculeMetric()
mol_data = MoleculeData()
frag_data = FragmentData()

def draw_leaf_fragments():
    frag_metric.draw_top_frags(to_idx=800)
    
def draw_frag_groups():
    frag_metric.draw_top_frags(to_idx=800, groups=True)

def draw_frag_parents(id_):
    mol_metric.draw_parent_mols(id_=id_)

def draw_mols_with_group(group, to_idx=200):
    mol_metric.draw_parent_mols(group=group, to_idx=to_idx)

def mol_distribution(metric, paint={}, save_as=None):
    mol_metric.dist_plot(metric, paint, save_as)

def mult_dist_wrapper(shape, metrics=[], painters=[], save_as=None):
    mol_metric.mult_dist_plot(shape, metrics, painters, save_as)

def comp_hexbin(save_as=None):
    mol_metric.comp_hexbin(save_as=save_as)

def plot_group_subset(group='', id_=None, save_as=None):
    mol_metric.group_subset(group=group, id_=id_, save_as=save_as)

def multi_group_wrapper(groups, save_as=None):
    mol_metric.multi_group_subset(groups=groups, shape=(3,2), save_as=save_as)

def group_violin_plots(groups=[], save_as=None):
    mol_metric.group_lambda_dist(groups,save_as)

def most_freq_combos():
    mol_metric.groups_in_combo()

def fragment_similarities(ids=[]):
    frag_metric.similarity(ids)
    
def molecule_similarities(ids=[]):
    mol_metric.similarity(ids)
    
def assign_frag_groups():
    return frag_data.get_frag_groups()

#mol_distribution(metric='ac',paint={'title':'blah','xlabel':'blah'})
#comp_hexbin()
#draw_frag_parents(2348)
#draw_mols_with_group('heterocycle-5')
#plot_group_subset(smi_id=4047)
#multi_group_wrapper(groups=['azo','anthraquinone','coumarin','cyanoacrylate','benzothiazole','triarylmethane'],save_as="scatter_groups.eps")
#group_violin_plots(groups=['azo','anthraquinone','coumarin','cyanoacrylate','benzothiazole','triarylmethane'],save_as="group_violins.eps")
#most_freq_combos()
#draw_leaf_fragments()
#fragment_similarities(ids=[33503,908])
#molecule_similarities(ids=[3779,3823])
#groups = assign_frag_groups()
#draw_frag_groups()
# x = frag_data.get_frag_groups()
# m = frag_data.get_frags_with_group(386).iloc[0].get_rdk_mol()
# patt = '[C;R0;D1;!$(C-[!#6])]'
# patt = Chem.MolFromSmarts('[C;R0;D1;!$(C-[!#6])]')
