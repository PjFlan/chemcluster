import os
from collections import defaultdict
import math
from itertools import combinations


from rdkit.Chem import rdMolDescriptors as Descriptors
from rdkit.Chem import Fragments
from rdkit import DataStructs

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from tabulate import tabulate
from matplotlib.colors import Normalize, LogNorm, SymLogNorm, PowerNorm

from helper import MyConfig, MyFileHandler, MyLogger

class Metric:
    
    def __init__(self, md, fd, gd, cd, af):
        self.md, self.fd, self.gd, self.cd, self.af = md, fd, gd, cd, af
        self._configure()
    
    def _configure(self):
        self._logger = MyLogger().get_child(type(self).__name__)
        self._fh = MyFileHandler()
        self._config = MyConfig()
        sns.set()
        
    def _set_fonts(self, large=False):
        if not large:
            plt.rc('font', size=20)
            plt.rc('legend', fontsize=15)
            plt.rc('axes', titlesize=25)     # fontsize of the axes title
            plt.rc('axes', labelsize=25)  
            plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
            plt.rc('ytick', labelsize=20)  
        else:
            plt.rc('font', size=32)
            plt.rc('legend', fontsize=25)
            plt.rc('axes', titlesize=32)     # fontsize of the axes title
            plt.rc('axes', labelsize=32)  
            plt.rc('xtick', labelsize=28)    # fontsize of the tick labels
            plt.rc('ytick', labelsize=28)  
            
    def _process_plot(self, file_name=None):
        if not file_name:
            plt.show()
            return
        data_dir = self._config.get_directory('data')
        filename = os.path.join(data_dir, 'plots', file_name)
        i = 1
        while os.path.isfile(filename):
            split = filename.split('.')
            name = split[0][:-1] + str(i)
            ext = split[1]
            filename = '.'.join([name, ext])
            i += 1
        if i > 1:
            print(f'Filename already exists. Saved to {filename} instead.')
        plt.savefig(filename,format='eps')
        
    def _paint_dist_axis(self, ax, paint):
        xlabel = paint.get('xlabel')
        title = paint.get('title')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Occurrence')
        ax.set_title(title)
        
    def _tidy_subplots(self,axes):
        cycle = axes.shape[1]
        for i, ax in enumerate(axes.flat):
            if i%cycle != 0:
                ax.yaxis.label.set_visible(False)
        plt.tight_layout()
        
    def basic_histogram(self, data, ax, norm=False, cut_off=0.9, bin_size=1):
        cut_off_val = math.ceil(data.quantile(cut_off))
        upper_bin = self._upper_bin(cut_off_val, bin_size)
        bins = list(range(0, int(upper_bin+bin_size*2), bin_size))

        data = data.apply(lambda x: x if x < upper_bin else upper_bin + bin_size/2)
        sns.distplot(data, bins, kde=False, norm_hist=norm, ax=ax)
        ax.set_xticks(bins[:-1])
        bins[-2] = str(bins[-2]) + '+'
        ax.set_xticklabels(bins[:-1])
        ax.set_ylabel('Frequency')
        ax.text(0.8, 0.9,f'N = {data.size:,}', transform = ax.transAxes)
        return ax
    
    def categ_histogram(self, data, ax, cut_off=0.95, bin_size=1):
        bins_order = None
        if data.dtype.name != 'category':
            categ_data, bins_order = self._int_to_cat(data, cut_off=cut_off, bin_size=bin_size)
        sns.countplot(categ_data, order=bins_order, palette=sns.color_palette("husl", 8), ax=ax)
        ax.set_ylabel('Occurrence')
        ax.text(0.8, 0.9, f'N = {data.size:,}', transform = ax.transAxes)
        return ax
        
    def similarity_report(self, entities):
        fps_1 = entities[0].basic_fingerprint()
        fps_2 = entities[1].basic_fingerprint()
        table = []
        sim_dict = {}
        fp_types = fps_1.keys()
        for fp in fp_types:
            dice = DataStructs.DiceSimilarity(fps_1[fp], fps_2[fp])
            tanimoto = DataStructs.FingerprintSimilarity(fps_1[fp], fps_2[fp])
            sim_dict[f'{fp}_dice'] = dice
            sim_dict[f'{fp}_tanimoto'] = tanimoto
            table.extend([[fp, 'dice', dice],[fp, 'tanimoto', tanimoto]])
        return sim_dict, tabulate(table)
        
    def _int_to_cat(self, data, cut_off, bin_size=1):
        cut_off_val = math.ceil(data.quantile(cut_off))
        upper_bin = self._upper_bin(cut_off_val,bin_size)
        bins = self._create_bins(bin_size,upper_bin)
        cat_data = data.apply(lambda x: self._assign_bin(x,bin_size,bins) 
                              if x < upper_bin else bins[upper_bin])
        bins_order = list(bins.values())
        return pd.Categorical(cat_data), bins_order
    
    def _upper_bin(self, upper, size):
        upper_bin = upper + (size-(upper%size))
        return upper_bin
    
    def _create_bins(self, size, upper_bin):
        bins = {i: ('-'.join([str(i),str(i+size)]) if size > 1 else str(i))
                for i in range(0,upper_bin,size)}
        bins[upper_bin] = str(upper_bin) + '+'
        return bins
        
    def _assign_bin(self, val, size, bins):
        mult = val//size
        left_edge = int(mult*size)
        return bins[left_edge]
    
    
class FragmentMetric(Metric):
    
    def __init__(self, md, fd, gd, cd, af):
        super().__init__(md, fd, gd, cd, af)
    
    def draw_top_frags(self, from_idx=0, to_idx=200):

        clean_frags = self.fd.clean_frags()
        clean_frags.set_occurrence()
        frag_dir = os.path.join(self._config.get_directory('images'),'fragments')
        self._draw_entities(clean_frags, frag_dir, from_idx, to_idx)        
    
    def similarity(self, ids):
        frags = self.fd.clean_frags()
        frags = frags[ids].tolist()
        super().similarity_report(entities=frags)
        
        
class GroupMetric(Metric):
        
    def __init__(self, md, fd, gd, cd, af):
        super().__init__(md, fd, gd, cd, af)
        
    def similarity(self, ids):
        groups = [self.gd.get_group(id_) for id_ in ids]
        super().similarity_report(entities=groups)
        
    def comp_data_report(self, id_):
        parents = self.gd.get_group_mols(id_)
        self.md.set_comp_data()
        parent_lam = parents.apply(lambda x: x.lambda_max)
        parent_f = parents.apply(lambda x: x.strength_max)
        ret_string = f'wavelength: {parent_lam.mean(): .5f} +- {parent_lam.std(): .3f}\n'
        ret_string = f'{ret_string}strength: {parent_f.mean(): .3f} +- {parent_f.std():.3f}'
        print(ret_string)
        
    def group_fp_scatter(self, min_size=15, occurrence=True, save_as=None):
        sim_mols = self.af.mols_same_group(occurrence=occurrence, exclude_benz=True)
        mols = self.md.clean_mols()
        sim_mols = {fp: ids for fp, ids in sim_mols.items() if len(ids) >= min_size}
        comb_df = []
        cat_num = 1
        for fp, mol_ids in sim_mols.items():
            tmp_mols = mols[mol_ids]
            lambdas = tmp_mols.apply(lambda x: x.lambda_max).tolist()
            strengths = tmp_mols.apply(lambda x: x.strength_max).tolist()
            categ = [cat_num]*len(lambdas)
            fp = [fp]*len(lambdas)
            comb_df.extend(list(zip(lambdas, strengths, categ, fp)))
            cat_num += 1
        comb_df = pd.DataFrame(comb_df)
        comb_df.columns = ['Wavelength', 'Strength', 'Category', 'Fingerprint']
        
        flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
        self._set_fonts(large=True)
        plots = len(sim_mols)//6
        shape = (plots//2, plots//2)
        figsize = (13*shape[0], 8*shape[1])
        fig, axs = plt.subplots(*shape, figsize=figsize)
        axes = axs.flat
        ord_ = 97
        for i in range(0, plots):
            ax = axes[i]
            from_idx = i*6
            to_idx = from_idx + 6
            tmp_df = comb_df[comb_df['Category'].gt(from_idx) & comb_df['Category'].le(to_idx)]
            #fig, ax = plt.subplots(1, 1, figsize=(1.5*len(sim_mols), 4))
            #ax = plt.gca()

            sns.scatterplot(x="Strength", y="Wavelength", 
                hue="Fingerprint", palette=sns.color_palette(flatui),
                data=tmp_df, ax=ax, s=70)
            ax.legend(loc=(0.8, 0.05))
            #sns.violinplot(x="Category", y="Strength", data=comb_df, ax=ax)
            ax.set_xlabel('Strength')
            ax.set_ylabel('Wavelength (nm)')
            ax.text(-0.1, 1.05, f'({chr(ord_ + i)})', transform = ax.transAxes)
        self._tidy_subplots(axs)
        self._process_plot(save_as)
           
class MoleculeMetric(Metric):
    
    def __init__(self, md, fd, gd, cd, af):
        super().__init__(md, fd, gd, cd, af)
        self.dist_func_map = {'ac': self.arom_cycles, 'ahc': self.arom_het_cycles,
                              'ha': self.heteroatom_count, 'hal': self.count_halogen, 
                              'cj': self.conjugation_count, 'lfc': self.largest_frag_count}
         
    def _get_largest_frag(self, mol, frags):
        mol_frags = self.fd.get_mol_frags(mol.get_id())
        max_size = frags[mol_frags.index].max()
        return max_size
        
    def similarity(self, ids=[], mols=[]):
        if not mols:
            mols = self.md.get_molecules()
            mols = mols[ids].tolist()
        report, table = super().similarity_report(entities=mols)
        return report, table
        
    def mult_dist_plot(self, shape, metrics=None, bin_sizes=[], painters=None, save_as=None):
        self.set_fonts()
        figsize = (10*shape[0], 8*shape[0])
        fig, axs = plt.subplots(*shape, sharey=True, figsize=figsize)
        if len(axs.flat) != len(metrics):
            return 'Shape does not match number of metrics.'
        axes = axs.flat
        for i, metric in enumerate(metrics):
            #axes[i].title.set_fontsize(30)
            self.dist_plot(metric, painters[i], bin_sizes[i], axes[i], is_mult=True)
        self._tidy_subplots(axs)
        self._process_plot(save_as)
        
    def dist_plot(self, metric, paint, bin_size=1, ax=None, is_mult=False, save_as=None):
        self.set_fonts()
        if not ax:
            ax = plt.gca()
        dist_func = self.dist_func_map[metric]
        count_df = dist_func()
        self.categ_histogram(data=count_df, ax=ax, bin_size=bin_size)
        self._paint_dist_axis(ax, paint)
        if not is_mult:
            self._process_plot(save_as)
                
    def arom_cycles(self):
        num_rings = self.md.get_molecules().apply(
            lambda x: Descriptors.CalcNumAromaticRings(x.get_rdk_mol()))
        return num_rings
            
    def arom_het_cycles(self):
        num_het_cyc = self.md.get_molecules().apply(
            lambda x: Descriptors.CalcNumAromaticHeterocycles(x.get_rdk_mol()))
        return num_het_cyc
    
    def heteroatom_count(self):
        num_het_atoms = self.md.get_molecules().apply(
            lambda x: Descriptors.CalcNumHeteroatoms(x.get_rdk_mol()))
        return num_het_atoms
    
    def count_thiophene(self):
        num_thiophenes = self.md.get_molecules().apply(
            lambda x: Fragments.fr_thiophene(x.get_rdk_mol()))
        return num_thiophenes
    
    def count_furan(self, molecules):
        num_furans = self.md.get_molecules().apply(
            lambda x: Fragments.fr_furan(x.get_rdk_mol()))
        return num_furans
    
    def count_halogen(self, molecules):
        num_halogens = self.md.get_molecules().apply(
            lambda x: Fragments.fr_halogen(x.get_rdk_mol()))
        return num_halogens
    
    def conjugation_count(self):
        self.md.set_conjugation()
        conj_bonds = self.md.get_molecules().apply(
            lambda x: x.get_conjugation())
        conj_bonds = conj_bonds.dropna()
        return conj_bonds
    
    def largest_frag_count(self):
        frags = self.fd.get_fragments()
        frags = frags.apply(lambda x: x.get_size())
        mols = self.md.get_molecules()
        largest_frag = mols.apply(self._get_largest_frag, args=(frags,))
        largest_frag.dropna(inplace=True)
        return largest_frag
            
    def comp_hexbin(self, save_as=None):
        self.md.set_comp_data()
        molecules = self.md.clean_mols()
        lambdas = molecules.apply(lambda x: x.lambda_max).dropna()
        strength = molecules.apply(lambda x: x.strength_max).dropna()
        
        ax = plt.gca()
        p = ax.hexbin(strength, lambdas, gridsize=25, cmap="summer", 
                      mincnt=1, norm=PowerNorm(.4))
        ax.set_xlabel('Oscillator Strength')
        ax.set_ylabel('Wavelength (nm)')
        ax.text(0.8, 0.9,f'N = {strength.size:,}', transform = ax.transAxes)
        plt.colorbar(p, ax=ax)
        self._process_plot(save_as)
        
    def multi_group_subset(self, shape, groups, save_as=None):
        
        sns.set_style("whitegrid")
        figsize = (6*shape[0], 6*shape[0])
        fig, axs = plt.subplots(*shape, sharey=True, figsize=figsize)
        if len(axs.flat) != len(groups):
            return 'Shape does not match number of metrics.'
        axes = axs.flat
        for i, group in enumerate(groups):
            self.group_subset(group, ax=axes[i], is_mult=True)
        self._tidy_subplots(axs)
        self._process_plot(save_as)
        
    def group_subset(self, group='', smi_id=None, ax=None, save_as=None, is_mult=False):
        if not ax:
            ax = plt.gca()
        all_molecules = self.md.clean_mols()
        if not smi_id:
            group_subset = self.md.find_mols_with_pattern(group)
        else:
            group_subset = self.md.get_parent_mols(smi_id)
        all_lambdas = all_molecules.apply(lambda x: x.lambda_max)
        all_strengths = all_molecules.apply(lambda x: x.strength_max)
        group_lambdas = group_subset.apply(lambda x: x.lambda_max)
        group_strengths = group_subset.apply(lambda x: x.strength_max)
        ax.grid(False)
        ax.set_title(group)
        ax.set_ylabel('Oscillator Strength')
        ax.set_xlabel('Wavelength (nm)')
        ax.scatter(all_lambdas, all_strengths, c='grey', s=8, alpha=0.3, marker='h')
        ax.scatter(group_lambdas, group_strengths, c='blue', s=8, marker='h')
        if not is_mult:
            self._process_plot(save_as)

    def group_lambda_dist(self, groups, save_as):
        self._set_fonts()
        self.md.clean_mols()
        plot_df = pd.DataFrame({'lambdas':[],'group':[]})
        if not groups:
            groups = ["coumarin","azo","anthraquinone","triarylmethane","thiophene","benzothiazole"]
        for group in groups:
            matches = self.md.find_mols_with_pattern(group)
            lambdas = matches.apply(lambda x: x.lambda_max)
            categ = matches.apply(lambda x: group)
            temp_df = pd.DataFrame({'Wavelength': lambdas, 'Group': categ})
            plot_df = pd.concat([plot_df, temp_df], ignore_index=True)
        fig, ax = plt.subplots(1, 1, figsize=(2.7*len(groups), 5))
        sns.violinplot(x="Group", y="Wavelength", data=plot_df)
        ax.set_xlabel('')
        ax.set_ylabel('Wavelength (nm)')
        self._process_plot(save_as)
        
    def groups_in_combo(self):
        combo_dict = defaultdict(int)
        mols = self.md.clean_mols()
        for mol in mols:
            groups = self.md.groups_in_combo(mol)
            combos = list(combinations(groups,2))
            for combo in combos:
                combo_dict[combo] += 1
        sorted_dict = {k: v for k, v in sorted(
            combo_dict.items(), reverse=True, key=lambda item: item[1])}
        for k,v in sorted_dict.items():
            print(k,v)