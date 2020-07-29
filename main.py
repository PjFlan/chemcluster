#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 14:40:36 2020
@author: padraicflanagan
"""
from helper import MyConfig, MyLogger, MyFileHandler
from metrics import FragmentMetric, MoleculeMetric, FragmentGroupMetric
from data import MoleculeData, FragmentData, FragmentGroupData
from rdkit.Chem import AllChem as Chem

my_config = MyConfig()
my_logger = MyLogger().get_root()
my_fh = MyFileHandler()

fm = FragmentMetric()
mm = MoleculeMetric()
fgm = FragmentGroupMetric()
md = MoleculeData()
fd = FragmentData()
fgd = FragmentGroupData()

def draw_leaf_fragments():
    fm.draw_top_frags(to_idx=800)
    
def draw_frag_groups(tier=1):
    fgm.draw_frag_groups(tier, to_idx=800)

def draw_frag_parents(id_):
    mm.draw_parent_mols(id_=id_)

def draw_mols_with_group(group, to_idx=200):
    mm.draw_parent_mols(group=group, to_idx=to_idx)

def mol_distribution(metric, paint={}, save_as=None):
    mm.dist_plot(metric, paint, save_as)

def mult_dist_wrapper(shape, metrics=[], painters=[], save_as=None):
    mm.mult_dist_plot(shape, metrics, painters, save_as)

def comp_hexbin(save_as=None):
    mm.comp_hexbin(save_as=save_as)

def plot_group_subset(group='', id_=None, save_as=None):
    mm.group_subset(group=group, id_=id_, save_as=save_as)

def multi_group_wrapper(groups, save_as=None):
    mm.multi_group_subset(groups=groups, shape=(3,2), save_as=save_as)

def group_violin_plots(groups=[], save_as=None):
    mm.group_lambda_dist(groups, save_as)

def most_freq_combos():
    mm.groups_in_combo()

def fragment_similarities(ids=[]):
    fm.similarity(ids)
    
def molecule_similarities(ids=[]):
    mm.similarity(ids)


#mol_distribution(metric='ac',paint={'title':'blah','xlabel':'blah'})
#comp_hexbin()
#draw_frag_parents(2348)
#draw_mols_with_group('cyanoacrylate')
#plot_group_subset(smi_id=4047)
#multi_group_wrapper(groups=['azo','anthraquinone','coumarin','cyanoacrylate','benzothiazole','triarylmethane'],save_as="scatter_groups.eps")
#group_violin_plots(groups=['azo','anthraquinone','coumarin','cyanoacrylate','benzothiazole','triarylmethane'],save_as="group_violins.eps")
#most_freq_combos()
#draw_leaf_fragments()
#fragment_similarities(ids=[33503,908])
#molecule_similarities(ids=[3779,3823])
#groups = assign_frag_groups()
#draw_frag_groups(1)
#fg = fgd.get_frag_groups(tier=1)
#fgd.pickle_groups()
#clusters = fgd.cluster_fps()
#fgm.draw_cluster_groups(singletons=True)
#fgm.draw_group_parents(1898)
#fgd.pickle_groups([0,1])
#fgm.draw_frag_groups(tier=2)
#fps = fgd.fingerprint_mols()