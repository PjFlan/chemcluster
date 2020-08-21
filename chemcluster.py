"""
This is the main entry-point module for
the application. There is very little in terms
of logic or algorithms in here - it is principally
just a wrapper for the functionality exposed by
other modules. The majority of the application logic 
is found in DAL.py.
"""
import os

from DAL import MoleculeData, FragmentData, GroupData, ChainData
from metrics import FragmentMetric, MoleculeMetric, GroupMetric
from fingerprint import NovelFingerprintData

from helper import MyConfig, MyLogger, MyFileHandler
from helper import FingerprintNotSetError
from drawing import draw_mols_canvas, draw_entities

class ChemCluster:
    
    def __init__(self):
        self.md = MoleculeData()
        self.fd = FragmentData(self.md)
        self.gd = GroupData(self.md, self.fd)
        self.cd = ChainData(self.md, self.fd, self.gd)
        self.nfp = NovelFingerprintData(self.md, self.fd, self.gd, self.cd)
        self.mm = MoleculeMetric(self.md, self.fd, self.gd, self.cd, self.nfp)
        self.fm = FragmentMetric(self.md, self.fd, self.gd, self.cd, self.nfp)
        self.gm = GroupMetric(self.md, self.fd, self.gd, self.cd, self.nfp)
        self._configure()
        
    def _configure(self):
        self._logger = MyLogger().get_child(type(self).__name__)
        self._fh = MyFileHandler()
        self._config = MyConfig()
        
    def _setup_query_folder(self, folder):
        
        outdir = os.path.join(self._config.get_directory('images'), 
                      f'{folder}')
        if self._config.use_tmp():
            outdir = self._config.get_directory('tmp')
        if os.path.exists(outdir):
            self._fh.clean_dir(outdir)
        return outdir   
    
    def _large_comp_diff(self, mols, lam_thresh=75, osc_thresh=0.5):
        lambdas = mols.apply(lambda x: x.get_lambda_max())
        osc = mols.apply(lambda x: x.get_strength_max())
        lam_diff = lambdas.max() - lambdas.min()
        osc_diff = osc.max() - osc.min()

        if lam_diff > lam_thresh or osc_diff > osc_thresh:
            return True
        return False    
    
    def generate_fragments(self):
        regen =  self._config.get_regen('grouping')
        self.DEV_FLAG = self._config.get_flag('dev')
        self.mols = self.md.get_molecules(regen)
        self.frags = self.fd.get_fragments(regen)
        self.groups = self.gd.get_groups(regen)
        self.subs = self.cd.get_substituents(regen)
        self.bridges = self.cd.get_bridges(regen)
        self.md.set_comp_data()
        self.nfp.set_up()

    def generate_novel_fps(self):
        
        self._fps = self.mols.apply(
            lambda x: self.get_novel_fp(x.get_id()))  
    
    def get_novel_fp(self, mol_id, regen=False):
        
        mol = self.mols.loc[mol_id]
        if not regen:
            try:
                novel_fp = mol.get_novel_fp()
                return novel_fp
            except FingerprintNotSetError:  
                pass
        novel_fp = self.nfp.fingerprint_mol(mol_id)
        return novel_fp
    
    def get_novel_fps(self):
        try:
            return self._fps
        except AttributeError:
            self.generate_novel_fps()
            return self._fps
               
    def get_molecule(self, mol_id):
        return self.mols.loc[mol_id]
    
    def get_fragment(self, frag_id):
        return self.frags.loc[frag_id]
    
    def get_group(self, group_id):
        return self.groups.loc[group_id]
    
    def get_substituent(self, sub_id):
        return self.subs.loc[sub_id]
    
    def get_bridge(self, bridge_id):
        return self.bridges.loc[bridge_id]
    
    def draw_entity_mols(self, ent_type, ent_id, from_idx=0):
        ent_func_dict = {'fragment': self.fd.get_frag_mols, 
                         'group': self.gd.get_group_mols,
                         'substituent': self.cd.get_sub_mols,
                         'bridge': self.cd.get_bridge_mols}
        ent_func = ent_func_dict[ent_type]
        mols = ent_func(ent_id)
        legends = mols.apply(lambda x: x.get_legend(self.DEV_FLAG))
        folder = os.path.join(self._config.get_directory('images'), 
                          f'{ent_type}_{ent_id}_mols/')
        draw_mols_canvas(mols, legends, folder, start_idx=from_idx)

    #all the functions starting with q_ are queries
        
    def q_same_groups(self, counts=True, query='', pattern='', 
                     large_comp_diff=False, draw=True):
    
        mols = self.md.clean_mols()
        sim_mols = self.nfp.mols_same_group(counts, pattern, exclude_benz=True)
        folder = f'mols_same_group_{counts}_{query}_{pattern}'
        outdir = self._setup_query_folder(folder)
        if draw:
            for key, mol_ids in sim_mols.items():
                mols_tmp = mols[mol_ids]
                if not large_comp_diff or self._large_comp_diff(mols_tmp):
                    legends = mols_tmp.apply(lambda x: 
                                             x.get_legend(self.DEV_FLAG))
                    draw_mols_canvas(mols_tmp, legends, 
                                     outdir, suffix=key, clean_dir=False)
            return None
        if query:
            return mols[sim_mols[query]]
        return sim_mols
    
    def q_diff_counts(self, group_id=None, draw=True, query=''):
        
        mol_sets = self.nfp.mols_same_except_one(group_id)
        mols = self.md.clean_mols()
        folder = f'mols_same_expt_one_{group_id}'
        if not draw:
            if query:
                return mols[list(mol_sets[query])]
            return mol_sets
        if query:
            return mols[mol_sets[query]]      
        outdir = self._setup_query_folder(folder)
        for key, mol_ids in mol_sets.items():
            m_ids = list(mol_ids)
            mols_tmp = self.mols.loc[m_ids]
            legends = mols_tmp.apply(lambda x: x.get_legend(self.DEV_FLAG))
            if len(mols_tmp) <= 1:
                continue
            if key == '2':
                continue
            draw_mols_canvas(mols_tmp, legends, outdir, suffix=key, clean_dir=False)
        
    def q_diff_topology(self, draw=True, large_comp_diff=False):
        mol_subsets = self.nfp.mols_diff_topology()
        folder = f'diff_topology_{large_comp_diff}'
        outdir = self._setup_query_folder(folder)
        if not draw:
            return mol_subsets

        for fp, mol_ids in mol_subsets.items():
            mols_tmp = self.mols[mol_ids]
            if not large_comp_diff or self._large_comp_diff(mols_tmp):
                legends = mols_tmp.apply(lambda x: 
                                         x.get_legend(self.DEV_FLAG))
                draw_mols_canvas(mols_tmp, legends, 
                                 outdir, suffix=fp, clean_dir=False)

    def q_diff_substituents(self, sub_id, draw=True, large_comp_diff=False):
        mol_subsets = self.nfp.same_fp_except_sub(sub_id)
        folder = f'same_except_{sub_id}'
        outdir = self._setup_query_folder(folder)
        if not draw:
            return mol_subsets

        for fp, mol_ids in mol_subsets.items():
            mols_tmp = self.mols[mol_ids]
            if not large_comp_diff or self._large_comp_diff(mols_tmp):
                legends = mols_tmp.apply(lambda x: 
                                         x.get_legend(self.DEV_FLAG))
                draw_mols_canvas(mols_tmp, legends, 
                                 outdir, suffix=fp, clean_dir=False)               

            
    def similarity_report(self, mol_ids):
        report, table = self.mm.similarity(mol_ids)
        print(table)
    
    def average_cluster_sim(self, mols, fp_type='morgan', metric='tanimoto'):

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
            
    def average_fp_sim(self, fp_type='rdk', metric='dice'):
        return self.mm.average_fp_sim(fp_type, metric)
    
    def get_mols_with_fp(self, fp, counts=True, draw=False):
        mols = self.nfp.mols_with_fp(fp, counts=True)
        folder = f'mols_with_fp_{fp}'
        outdir = self._setup_query_folder(folder)
        if draw:
            legends = mols.apply(lambda x: x.get_legend(self.DEV_FLAG))
            draw_mols_canvas(mols, legends, outdir)
        return mols
        
    def get_groups_with_pattern(self, pattern, draw=False):
        groups = self.gd.find_groups_with_pattern(pattern)
        folder = f'groups_with_{pattern}'
        outdir = self._setup_query_folder(folder)
        if draw:
            legends = groups.apply(lambda x: f'{x.get_id()}')
            draw_mols_canvas(groups, legends, outdir)
        return groups
    
    def get_subs_with_pattern(self, pattern, draw=False):
        subs = self.cd.find_subs_with_pattern(pattern)
        folder = f'subs_with_{pattern}'
        outdir = self._setup_query_folder(folder)
        if draw:
            legends = subs.apply(lambda x: f'{x.get_id()}')
            draw_mols_canvas(subs, legends, outdir)
        return subs
        
    def draw_mols(self, mol_ids=[], img_type='png'):
        mols = self.mols[mol_ids]
        legends = mols.apply(lambda x: x.get_legend(self.DEV_FLAG))
        folder = os.path.join(self._config.get_directory('images'), 'mols_rpt')
        draw_mols_canvas(mols, legends, folder, img_type=img_type)        
            
    def draw_groups(self, from_idx=0, to_idx=200):
        
        groups = self.gd.get_groups()
        g_dir = os.path.join(self._config.get_directory('images'), f'groups')
        draw_entities(groups, g_dir, from_idx, to_idx)
        
    def draw_substituents(self, from_idx=0, to_idx=200):
        
        subs = self.cd.get_substituents()
        s_dir = os.path.join(self._config.get_directory('images'), f'subs')
        draw_entities(subs, s_dir, from_idx, to_idx)
        
    def draw_bridges(self, from_idx=0, to_idx=200):
        
        bridges = self.cd.get_bridges()
        b_dir = os.path.join(self._config.get_directory('images'), f'bridges')
        draw_entities(bridges, b_dir, from_idx, to_idx)
        
    def draw_fragments(self, from_idx, to_idx):

        frags = self.fd.get_fragments()
        frag_dir = os.path.join(self._config.get_directory('images'),'fragments')
        draw_entities(frags, frag_dir, from_idx, to_idx)        
       
    def draw_group_clusters(self,  cutoff=0.2, fp_type='MACCS', similarity='dice', 
                            singletons=False, from_idx=0, to_idx=200, clust_nums=None):
        
        cgm = self.gd.get_group_clusters()
        if not clust_nums:
            clust_nums = range(0, cgm['cluster_id'].max() + 1)
        for clust_num in clust_nums:
            group_indices = cgm[clust_num == cgm['cluster_id']]['group_id']
            groups_tmp = self.groups[group_indices]
            if (not singletons) and groups_tmp.size == 1:
                continue
            cluster_dir = os.path.join(self._config.get_directory('images'),
                                       f'clusters/cluster_{clust_num}/')
            draw_entities(groups_tmp, cluster_dir, from_idx, to_idx)
           
    def novel_fp_scatter(self, min_size, counts=True, save_as=None):
        self.gm.group_fp_scatter(min_size, counts, save_as)
        
    def comp_dist_plot(self, save_as=None):
        self.mm.comp_hexbin(save_as)
        
    def group_violin_plots(self, groups, save_as=None):
        self.mm.group_lambda_dist(groups, save_as)
        
    def get_group_cluster(self, group_id):
        return self.gd.get_group_cluster(group_id)
    
    def draw_group_cluster_mols(self, cluster_id):

        group_idx = self.gd.get_cluster_groups(cluster_id)
        groups = self.groups[group_idx]

        mols = self.gd.get_groups_mols(group_ids=groups.index)
        mg_dir = os.path.join(self._config.get_directory('images'),
                                     f'cluster_{cluster_id}_mols/')
        legends = mols.apply(lambda x: x.get_legend(self.DEV_FLAG))
        draw_mols_canvas(mols, legends, mg_dir, suffix='', start_idx=0)


if __name__ == '__main__':
    cc = ChemCluster()
    cc.generate_fragments()
    cc.generate_novel_fps()