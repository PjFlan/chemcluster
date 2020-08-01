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
from tabulate import tabulate
from matplotlib.colors import Normalize, LogNorm, SymLogNorm, PowerNorm

from helper import MyConfig, MyFileHandler, MyLogger, draw_to_png_stream

class Metric:
    
    DRAWING_RES = 600
    DRAWING_FONT_SIZE = 30
    
    def __init__(self, md, fd, gd, cd):
        self.md, self.fd, self.gd, self.cd = md, fd, gd, cd
        self._configure()
    
    def _configure(self):
        self._logger = MyLogger().get_child(type(self).__name__)
        self._fh = MyFileHandler()
        self._config = MyConfig()
        sns.set()
        
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
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        
    def _tidy_subplots(self,axes):
        cycle = axes.shape[1]
        for i,ax in enumerate(axes.flat):
            if i%cycle != 0:
                ax.yaxis.label.set_visible(False)
        plt.tight_layout()
        
    def _draw_entities(self, entities, frag_dir, from_idx, to_idx):
        freqs = entities.apply(lambda x: x.occurrence)
        freq_df = pd.concat([entities, freqs], axis=1)
        freq_df.columns = ['entity','occurrence']
        freq_df = freq_df.sort_values(by='occurrence', ascending=False).iloc[from_idx:to_idx]
        
        entities = freq_df.apply(lambda x: x['entity'].get_rdk_mol(), axis=1)
        legends = freq_df.apply(lambda f: f"id: {f['entity'].id_}, freq: {f['occurrence']}", axis=1)

        self._draw_mols_canvas(entities, legends, outdir=frag_dir, suffix='', 
                              start_idx=from_idx, per_img=20, per_row=5)
        
    def basic_histogram(self, data, ax, norm=False, cut_off=0.9, bin_size=1):
        cut_off_val = math.ceil(data.quantile(cut_off))
        upper_bin = self._upper_bin(cut_off_val,bin_size)
        bins = list(range(0,int(upper_bin+bin_size*2),bin_size))

        data = data.apply(lambda x: x if x < upper_bin else upper_bin + bin_size/2)
        sns.distplot(data,bins,kde=False,norm_hist=norm,ax=ax)
        ax.set_xticks(bins[:-1])
        bins[-2] = str(bins[-2]) + '+'
        ax.set_xticklabels(bins[:-1])
        ax.set_ylabel('Frequency')
        ax.text(0.8,0.9,f'N = {data.size:,}',transform = ax.transAxes)
        return ax
    
    def categ_histogram(self, data, ax, cut_off=0.95, bin_size=1):
        bins_order = None
        if data.dtype.name != 'category':
            categ_data,bins_order = self._int_to_cat(data,cut_off=cut_off,bin_size=bin_size)
        sns.countplot(categ_data,order=bins_order,palette=sns.color_palette("husl", 8),ax=ax)
        ax.set_ylabel('Frequency')
        ax.text(0.8,0.9,f'N = {data.size:,}',transform = ax.transAxes)
        return ax
    
    def _draw_mols_canvas(self, mols, legends, outdir, suffix, 
                         start_idx=0, per_img=20, per_row=5):
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        num_mols = len(mols)
            
        if num_mols%per_img == 0:
            num_files = num_mols//per_img
        else:
            num_files = num_mols//per_img + 1
            
        res = self.DRAWING_RES
        sub_img_size= (res, res)
        n_rows = per_img//per_row
        full_size = (per_row * sub_img_size[0], n_rows * sub_img_size[1])
        font_size = self.DRAWING_FONT_SIZE*(res//600)
        file_num = 1
        file_begin = 0
        file_end = per_img
        while file_num<=num_files:
            file = f'{suffix}{file_begin+start_idx+1}-{file_end+start_idx}.png'
            file = os.path.join(outdir,file)
            curr_mols = mols.iloc[file_begin:file_end].tolist()
            lgnds = legends.iloc[file_begin:file_end].tolist()
            stream = draw_to_png_stream(curr_mols, full_size, sub_img_size, font_size, lgnds)
            
            with open(file,'wb+') as ih:
                ih.write(stream)
            
            file_num += 1
            file_begin += per_img
            if file_num == num_files:
                file_end = num_mols
            else:
                file_end += per_img
        
    def similarity_report(self, entities):
        fps_1 = entities[0].fingerprint()
        fps_2 = entities[1].fingerprint()
        table = []
        fp_types = fps_1.keys()
        for fp in fp_types:
            dice = DataStructs.DiceSimilarity(fps_1[fp],fps_2[fp])
            tanimoto = DataStructs.FingerprintSimilarity(fps_1[fp],fps_2[fp])
            table.extend([[fp,'dice',dice],[fp,'tanimoto',tanimoto]])
        print(tabulate(table)) 
        
    def _int_to_cat(self, data, cut_off, bin_size=1):
        cut_off_val = math.ceil(data.quantile(cut_off))
        upper_bin = self._upper_bin(cut_off_val,bin_size)
        bins = self._create_bins(bin_size,upper_bin)
        cat_data = data.apply(lambda x: self._assign_bin(x,bin_size,bins) 
                              if x < upper_bin else bins[upper_bin])
        bins_order = list(bins.values())
        return pd.Categorical(cat_data),bins_order
    
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
    
    def __init__(self, md, fd, gd, cd):
        super().__init__(md, fd, gd, cd)
    
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
        
    def __init__(self, md, fd, gd, cd):
        super().__init__(md, fd, gd, cd)
        
    def draw_groups(self, tier=0, from_idx=0, to_idx=200):
        
        groups = self.gd.get_groups()
        g_dir = os.path.join(self._config.get_directory('images'),f'fragment_groups_{tier}')
        self._draw_groups(groups, g_dir, from_idx, to_idx)
        
    def draw_clusters(self, clust_nums=None, singletons=False, from_idx=0, to_idx=200):
        
        cgm = self.gd.get_group_clusters()
        groups = self.gd.get_groups()
        if not clust_nums:
            clust_nums = range(0, cgm['cluster_id'].max() + 1)
        for clust_num in clust_nums:
            group_indices = cgm[clust_num == cgm['cluster_id']]['group_id']
            groups_tmp = groups[group_indices]
            if (not singletons) and groups_tmp.size == 1:
                continue
            cluster_dir = os.path.join(self._config.get_directory('images'),f'cluster_{clust_num}/')
            self._draw_entities(groups_tmp, cluster_dir, from_idx, to_idx)
            
    def draw_group_parents(self, id_, from_idx=0, to_idx=200):
        parents = self.gd.get_group_mols(id_)
        parent_mols = parents.apply(lambda x: x.get_rdk_mol())
        fg_parent_dir = os.path.join(self._config.get_directory('images'),f'group_{id_}_parents/')
        self.md.set_comp_data()
        legends = parents.apply(lambda p: f'{p.get_id()} ; {p.lambda_max}nm ; {p.strength_max:.4f}' 
            if p.lambda_max else '')
        self._draw_mols_canvas(parent_mols, legends, fg_parent_dir, suffix='', start_idx=from_idx)
        
    def draw_cluster_parents(self, cluster_id):

        group_idx = self.gd.get_cluster_groups(cluster_id)
        groups = self.gd.get_groups()[group_idx]
        mols = self.md.get_molecules()
        self.md.set_comp_data()
        parent_ids = []
        for group in groups:
            parent_ids.extend(group.get_parent_mols())
        parents = mols[parent_ids]
        parent_mols = parents.apply(lambda x: x.get_rdk_mol())
        fg_parent_dir = os.path.join(self._config.get_directory('images'),
                                     f'cluster_{cluster_id}_parents/')
        legends = parents.apply(lambda p: f'{p.get_id()} ; {p.lambda_max}nm ; {p.strength_max:.4f}' 
                                if p.lambda_max else '')
        self._draw_mols_canvas(parent_mols, legends, fg_parent_dir, suffix='', start_idx=0)
        
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
    
        
class MoleculeMetric(Metric):
    
    def __init__(self, md, fd, gd, cd):
        super().__init__(md, fd, gd, cd)
        self.dist_func_map = {'ac':self.arom_cycles,'ahc':self.arom_het_cycles,'ha':self.heteroatom_count,
                     'th':self.count_thiophene,'hal':self.count_halogen,'fu':self.count_furan,
                     'cj':self.conjugation_count,'lfc':self.largest_frag_count}
         
    def _get_largest_frag(self, mol):
        frag_sizes = [frag.get_size() for frag in mol.fragments]
        if frag_sizes:
            return max(frag_sizes)
        else:
            return None
        
    def similarity(self, ids):
        mols = self.md.get_molecules()
        mols = mols[ids].tolist()
        super().similarity_report(entities=mols)
        
    def draw_entity_mols(self, ent_name, group='', ent_id=None, from_idx=0, to_idx=200):
        mols = self.md.get_molecules()
        if not ent_id:
            parents = self.md.find_mols_with_pattern(group)
        else:
            parents = self.md.get_entity_mols(ent_id, mols, ent_name)
            group = str(ent_id)
        self.md.set_comp_data()
        mols = parents.apply(lambda p: p.get_rdk_mol())
        legends = parents.apply(lambda p: f'{p.get_id()} ; {p.lambda_max}nm ; {p.strength_max:.4f}' 
                    if p.lambda_max else '')
        folder = os.path.join(self._config.get_directory('images'), 
                              f'{ent_name}_{group}_mols/')
        
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        if to_idx and to_idx < len(mols):
            mols = mols.iloc[from_idx:to_idx]
            legends = legends.iloc[from_idx:to_idx]
        suffix = group + '_'
        self._draw_mols_canvas(mols=mols, legends=legends, outdir=folder, start_idx=from_idx, suffix=suffix,
                              per_img=12, per_row=3)
        
    def mult_dist_plot(self, shape, metrics=None, painters=None, save_as=None):
        figsize = (8*shape[0], 6*shape[0])
        fig, axs = plt.subplots(*shape, sharey=True, figsize=figsize)
        if len(axs.flat) != len(metrics):
            return 'Shape does not match number of metrics.'
        axes = axs.flat
        for i,metric in enumerate(metrics):
            self.dist_plot(metric, painters[i], axes[i], is_mult=True)
        self._tidy_subplots(axs)
        self._process_plot(save_as)
        
    def dist_plot(self, metric, paint, ax=None, is_mult=False, save_as=None):
        if not ax:
            ax = plt.gca()

        dist_func = self.dist_func_map[metric]
        count_df = dist_func()
        self.categ_histogram(data=count_df, ax=ax)
        self._paint_dist_axis(ax, paint)
        if not is_mult:
            self._process_plot(save_as)
                
    def arom_cycles(self):
        num_rings = self.md.molecules.apply(lambda x: Descriptors.CalcNumAromaticRings(x.get_rdk_mol()))
        return num_rings
            
    def arom_het_cycles(self):
        num_het_cyc = self.md.molecules.apply(lambda x: Descriptors.CalcNumAromaticHeterocycles(x.get_rdk_mol()))
        return num_het_cyc
    
    def heteroatom_count(self):
        num_het_atoms = self.md.molecules.apply(lambda x: Descriptors.CalcNumHeteroatoms(x.get_rdk_mol()))
        return num_het_atoms
    
    def count_thiophene(self):
        num_thiophenes = self.md.molecules.apply(lambda x: Fragments.fr_thiophene(x.get_rdk_mol()))
        return num_thiophenes
    
    def count_furan(self, molecules):
        num_furans = self.md.molecules.apply(lambda x: Fragments.fr_furan(x.get_rdk_mol()))
        return num_furans
    
    def count_halogen(self, molecules):
        num_halogens = self.md.molecules.apply(lambda x: Fragments.fr_halogen(x.get_rdk_mol()))
        return num_halogens
    
    def conjugation_count(self):
        self.md.get_conjugation(self.md.molecules.tolist())
        conj_bonds = self.md.molecules.apply(lambda x: x.conjugation)
        conj_bonds = conj_bonds.dropna()
        return conj_bonds
    
    def largest_frag_count(self):
        self.md.get_fragments()
        largest_frag = self.md.molecules.apply(lambda mol: self._get_largest_frag(mol))
        largest_frag.dropna()
        return largest_frag
            
    def comp_hexbin(self, save_as=None):
        self.md.set_comp_data()
        molecules = self.clean_mols()
        lambdas = molecules.apply(lambda x: x.lambda_max)
        strength = molecules.apply(lambda x: x.strength_max)
        
        ax = plt.gca()
        ax.grid(False)
        p = ax.hexbin(strength, lambdas, gridsize=25, cmap="summer", mincnt=1,norm=PowerNorm(.4))
        ax.set_xlabel('Oscillator Strength')
        ax.set_ylabel('Wavelength (nm)')
        plt.colorbar(p, ax=ax)
        self._process_plot(save_as)
        
    def multi_group_subset(self, shape, groups, save_as=None):
        sns.set_style("whitegrid")
        figsize = (6*shape[0],6*shape[0])
        fig, axs = plt.subplots(*shape, sharey=True,figsize=figsize)
        if len(axs.flat) != len(groups):
            return 'Shape does not match number of metrics.'
        axes = axs.flat
        for i,group in enumerate(groups):
            self.group_subset(group,ax=axes[i],is_mult=True)
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
        ax.scatter(all_lambdas,all_strengths, c='grey', s=8, alpha=0.3, marker='h')
        ax.scatter(group_lambdas, group_strengths, c='blue', s=8, marker='h')
        if not is_mult:
            self._process_plot(save_as)

    def group_lambda_dist(self, groups, save_as):
        self.md.clean_mols()
        plot_df = pd.DataFrame({'lambdas':[],'group':[]})
        if not groups:
            groups = ["coumarin","azo","anthraquinone","triarylmethane","thiophene","benzothiazole"]
        for group in groups:
            matches = self.md.find_mols_with_pattern(group)
            lambdas = matches.apply(lambda x: x.lambda_max)
            categ = matches.apply(lambda x: group)
            temp_df = pd.DataFrame({'Wavelength':lambdas,'Group':categ})
            plot_df = pd.concat([plot_df,temp_df],ignore_index = True)
        fig,ax = plt.subplots(1,1,figsize=(2*len(groups),5))
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
        sorted_dict = {k: v for k, v in sorted(combo_dict.items(), reverse=True, key=lambda item: item[1])}
        for k,v in sorted_dict.items():
            print(k,v)