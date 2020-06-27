import os
from collections import Counter, defaultdict
import math
from itertools import combinations

from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdMolDescriptors as Descriptors
from rdkit.Chem import Fragments
from rdkit import DataStructs

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
from matplotlib.colors import Normalize, LogNorm, SymLogNorm, PowerNorm

from data import MoleculeData, FragmentData
from helper import MyConfig, MyFileHandler, MyLogger

class Metric:
    DRAWING_RES = 600
    DRAWING_FONT_SIZE = 30
    
    def __init__(self):
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
    
    def draw_mols_canvas(self, mols, legends, outdir, suffix, 
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
            d2d = rdMolDraw2D.MolDraw2DCairo(full_size[0], full_size[1], sub_img_size[0], sub_img_size[1])
            d2d.drawOptions().legendFontSize = font_size
            d2d.DrawMolecules(curr_mols,legends=legends.iloc[file_begin:file_end].tolist())
            d2d.FinishDrawing()
            
            with open(file,'wb+') as ih:
                ih.write(d2d.GetDrawingText())
            
            file_num += 1
            file_begin += per_img
            if file_num == num_files:
                file_end = num_mols
            else:
                file_end += per_img

    def draw_to_svg_stream(self, mol):
        d2svg = rdMolDraw2D.MolDraw2DSVG(300,300)
        d2svg.DrawMolecule(mol)
        d2svg.FinishDrawing()
        return d2svg.GetDrawingText()
    
    def similarity(self, entities):
        fps_1 = entities[0].fingerprint()
        fps_2 = entities[1].fingerprint()
        self.similarity_report(fps_1, fps_2)
        
    def similarity_report(self, fps_1, fps_2):
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
    
    def __init__(self):
        super().__init__()
        self.frag_data = FragmentData()
        self.mol_data = MoleculeData()
    
    def draw_top_frags(self, from_idx=0, to_idx=200, groups=False):
        if not groups:
            clean_frags = self.frag_data.clean_frags()
        else:
            clean_frags = self.frag_data.get_frag_groups()
        freqs = clean_frags.apply(lambda f: f.occurrence)
        freq_df = pd.concat([clean_frags, freqs], axis=1)
        freq_df.columns = ['fragment','frequency']
        freq_df = freq_df.sort_values(by='frequency', ascending=False).iloc[from_idx:to_idx]
        
        frag_mols = freq_df.apply(lambda f: f['fragment'].get_rdk_mol(), axis=1)
        legends = freq_df.apply(lambda f: f'id: {f["fragment"].id_}, freq: {f["frequency"]}', axis=1)
    
        frag_dir = os.path.join(self._config.get_directory('images'),'fragments')
        suffix = ''
        if groups:
            suffix = 'group_'
        self.draw_mols_canvas(frag_mols, legends, outdir=frag_dir, suffix=suffix, 
                              start_idx=from_idx, per_img=20, per_row=5)
    
    def similarity(self, ids):
        frags = self.frag_data.clean_frags()
        frags = frags[ids].tolist()
        super().similarity(entities=frags)
        
        
class MoleculeMetric(Metric):
    
    def __init__(self):
        super().__init__()
        self.mol_data = MoleculeData()
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
        mols = self.mol_data.get_molecules()
        mols = mols[ids].tolist()
        super().similarity(entities=mols)
        
    def draw_parent_mols(self, group='', id_=None, from_idx=0, to_idx=200):
        self.mol_data.set_comp_data()
        self.mol_data.get_fragments()
        if not id_:
            parents = self.mol_data.find_mols_with_pattern(group)
        else:
            parents = self.mol_data.get_parent_mols(id_)
            group = str(id_)
        mols = parents.apply(lambda p: p.get_rdk_mol())
        legends = parents.apply(lambda p: f'{p.get_id()} ; {p.lambda_max}nm ; {p.strength_max:.4f}' 
                    if p.lambda_max else '')
        parents_dir = os.path.join(self._config.get_directory('images'),'parents')
        folder = os.path.join(parents_dir,group)
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        if to_idx and to_idx < len(mols):
            mols = mols.iloc[from_idx:to_idx]
            legends = legends.iloc[from_idx:to_idx]
        suffix = group + '_'
        self.draw_mols_canvas(mols=mols, legends=legends, outdir=folder, start_idx=from_idx, suffix=suffix,
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
        num_rings = self.mol_data.molecules.apply(lambda x: Descriptors.CalcNumAromaticRings(x.get_rdk_mol()))
        return num_rings
            
    def arom_het_cycles(self):
        num_het_cyc = self.mol_data.molecules.apply(lambda x: Descriptors.CalcNumAromaticHeterocycles(x.get_rdk_mol()))
        return num_het_cyc
    
    def heteroatom_count(self):
        num_het_atoms = self.mol_data.molecules.apply(lambda x: Descriptors.CalcNumHeteroatoms(x.get_rdk_mol()))
        return num_het_atoms
    
    def count_thiophene(self):
        num_thiophenes = self.mol_data.molecules.apply(lambda x: Fragments.fr_thiophene(x.get_rdk_mol()))
        return num_thiophenes
    
    def count_furan(self, molecules):
        num_furans = self.mol_data.molecules.apply(lambda x: Fragments.fr_furan(x.get_rdk_mol()))
        return num_furans
    
    def count_halogen(self, molecules):
        num_halogens = self.mol_data.molecules.apply(lambda x: Fragments.fr_halogen(x.get_rdk_mol()))
        return num_halogens
    
    def conjugation_count(self):
        self.mol_data.get_conjugation(self.mol_data.molecules.tolist())
        conj_bonds = self.mol_data.molecules.apply(lambda x: x.conjugation)
        conj_bonds = conj_bonds.dropna()
        return conj_bonds
    
    def largest_frag_count(self):
        self.mol_data.get_fragments()
        largest_frag = self.mol_data.molecules.apply(lambda mol: self._get_largest_frag(mol))
        largest_frag.dropna()
        return largest_frag
            
    def comp_hexbin(self, save_as=None):
        self.mol_data.set_comp_data()
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
        all_molecules = self.mol_data.clean_mols()
        if not smi_id:
            group_subset = self.mol_data.find_mols_with_pattern(group)
        else:
            group_subset = self.mol_data.get_parent_mols(smi_id)
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
        self.mol_data.clean_mols()
        plot_df = pd.DataFrame({'lambdas':[],'group':[]})
        if not groups:
            groups = ["coumarin","azo","anthraquinone","triarylmethane","thiophene","benzothiazole"]
        for group in groups:
            matches = self.mol_data.find_mols_with_pattern(group)
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
        mols = self.mol_data.clean_mols()
        for mol in mols:
            groups = self.mol_data.groups_in_combo(mol)
            combos = list(combinations(groups,2))
            for combo in combos:
                combo_dict[combo] += 1
        sorted_dict = {k: v for k, v in sorted(combo_dict.items(), reverse=True, key=lambda item: item[1])}
        for k,v in sorted_dict.items():
            print(k,v)