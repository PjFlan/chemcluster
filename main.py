#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 14:40:36 2020
@author: padraicflanagan
"""
from helper import MyConfig,MyLogger,MyFileHandler
from metrics import FragmentMetric,MoleculeMetric

my_config = MyConfig()
my_logger = MyLogger().get_root()
my_fh = MyFileHandler()

frag_metric = FragmentMetric()
mol_metric = MoleculeMetric()

def draw_leaf_fragments():
    frag_metric.draw_frags(to_idx=800)

def draw_frag_parents(smi_id):
    mol_metric.draw_parent_mols(smi_id=smi_id)

def draw_mols_with_group(group,to_idx=200):
    mol_metric.draw_parent_mols(group=group,to_idx=to_idx)

def mol_distribution(metric,paint={},save_as=None):
    mol_metric.dist_plot(metric,paint,save_as)

def mult_dist_wrapper(shape,metrics=[],painters=[],save_as=None):
    mol_metric.mult_dist_plot(shape,metrics,painters,save_as)

def comp_hexbin(save_as=None):
    mol_metric.comp_hexbin(save_as=save_as)

def plot_group_subset(group='',smi_id=None,save_as=None):
    mol_metric.group_subset(group=group,smi_id=smi_id,save_as=save_as)

def multi_group_wrapper(groups,save_as=None):
    mol_metric.multi_group_subset(groups=groups,shape=(3,2),save_as=save_as)

def group_violin_plots(groups=[],save_as=None):
    mol_metric.group_lambda_dist(groups,save_as)

def most_freq_combos():
    mol_metric.groups_in_combo()

def fragment_similarities(ids=[]):
    frag_metric.similarity(ids)

#mol_distribution(metric='ac',paint={'title':'blah','xlabel':'blah'})
#comp_hexbin()
#draw_frag_parents(830)
#draw_mols_with_group('heterocycle-5')
#plot_group_subset(smi_id=4047)
#multi_group_wrapper(groups=['azo','anthraquinone','coumarin','cyanoacrylate','benzothiazole','triarylmethane'],save_as="scatter_groups.eps")
#group_violin_plots(groups=['azo','anthraquinone','coumarin','cyanoacrylate','benzothiazole','triarylmethane'],save_as="group_violins.eps")
#most_freq_combos()
#draw_leaf_fragments()
fragment_similarities(ids=[23165,23981])
