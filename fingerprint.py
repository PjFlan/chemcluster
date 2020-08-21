"""
This module defines the novel fingerprint class.
The implementations for the fingerprint queries are
defined in this module. This module also exposes a class for 
accessing the traditional fingerprints.
"""
import re
from collections import defaultdict

from rdkit.Chem import AllChem as Chem
from rdkit.Chem import MACCSkeys

from drawing import draw_to_png_stream


class NovelFingerprintData:
    """
    Determine and set the novel fingerprint for each molecule.
    
    This class uses the pools of fragment entities extracted in
    DAL.py and the associated link tables between entities to 
    determine the entities contained by each molecule. The logic
    for using these novel fingerprints to construct and execute
    queries is also found in this class.
    """
    def __init__(self, mol_data, frag_data, group_data,
                 chain_data):
        self.md, self.fd, self.gd, self.cd =\
        mol_data, frag_data, group_data, chain_data
    
    def set_up(self):
        self._mols = self.md.get_molecules()
        self._groups = self.gd.get_groups()
        self._subs = self.cd.get_substituents()
        self._bridges = self.cd.get_bridges()
        
    def _prep_fp_for_grouping(self, fp, full=False):
        """
        Convert a fingerprint set object to a str.
        
        The fingerprints encoded as tuples of sets cannot be passed
        to the Pandas groupby function. This function takes these
        tuples and produces a order-deterministic string
        representation.
        """
        if full:
            ret_str = ''
            for ent, ent_fp in sorted(fp.items()):
                if isinstance(ent_fp, dict):
                    ent_fp = ent_fp.items()
                tmp_fp = sorted(list(set(ent_fp)))
                ret_str += '_'.join(map(str, tmp_fp))
        else:
            fp = sorted(list(fp))
            ret_str = '_'.join(map(str, fp))
        return ret_str
    
    def _check_sets_equal(self, set1, set2, complete):
        """
        Check if two sets interect.

        Parameters
        ----------
        set1 : set
        set2 : set
        complete : boolean
            If True, the sets must match exactly. If not,
            set2 only need be a subset of set1.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if complete:
            return set1 == set2
        return set2.issubset(set1)
    
    def _remove_sub(self, fp, sub_id):
        subs = fp['subs']
        subs.pop(sub_id, None)
        fp['subs'] = subs
        return fp
        
    def fingerprint_mol(self, mol_id):
        """
        Generate the novel fingerprint for a molecule
        """
        mol = self.md.get_molecule(mol_id)
        groups, bridges, subs = set(), set(), set()
        
        b_dict = defaultdict(int)
        mb = self.cd.get_link_table('mol_bridge')
        mb = mb[mb['mol_id'] == mol_id]
        for idx, row in mb.iterrows():
            b_id = row['bridge_id']
            b_dict[b_id] += 1
            bridge = self._bridges.loc[b_id]
            bridges.add(bridge)
          
        mg = self.gd.get_link_table('mol_group').drop_duplicates()
        mg = mg[mg['mol_id'] == mol_id]
        mgs = self.cd.get_link_table('mol_group_sub')
        mgs = mgs[mgs['mol_id'] == mol_id]

        g_dict = defaultdict(int)
        s_dict = defaultdict(list)

        for idx, row in mgs.iterrows():
            g_id = row['group_id']
            g_inst = row['instance_id']
            s_id = row['group_sub_id']
            s_dict[s_id].append((g_id, g_inst))
            sub = self._subs.loc[s_id]
            subs.add(sub)
            
        for idx, row in mg.iterrows():
            g_id = row['group_id']
            group = self._groups.loc[g_id]
            g_dict[g_id] += mol.pattern_count(group.get_rdk_mol())
            mol.pattern_count(group.get_rdk_mol())
            groups.add(group)
        mfp = NovelFingerprint(g_dict, s_dict, b_dict,
                             groups, bridges, subs)
        mol.set_novel_fp(mfp)
        return mfp
        
    def get_mol_fingerprints(self):
        """ 
        Generate the novel fingerprint for every molecule.
        """
        try:
            return self._fps
        except AttributeError:
            pass
        mols = self.md.clean_mols()
        self._fps = mols.apply(lambda x: self.fingerprint_mol(x.get_id()))
        return self._fps
    
    def get_group_fps(self, counts=False, exclude_benz=False):
        """ 
        A query for finding all molecules with the same groups.
        """
        self.get_mol_fingerprints()
        mols = self.md.clean_mols()
        g_fps = mols.apply(lambda x: x.get_novel_fp().get_group_fp(counts))
        if exclude_benz:
            if counts:
                is_only_benzene = g_fps.apply(
                    lambda x: len(x) == 1 and list(x)[0][0] == 2)
            else:
                is_only_benzene = g_fps.apply(
                    lambda x: len(x) == 1 and list(x)[0] == 2)
            g_fps = g_fps[~is_only_benzene]
        return g_fps
    
    def get_full_fps(self, as_set=True):
        """
        Return a series of novel fingerprints for every molecule
        """
        self.get_mol_fingerprints()
        mols = self.md.clean_mols()
        fps = mols.apply(lambda x: 
                         x.get_novel_fp().get_full_fp(as_set))
        return fps
    
    def get_sub_fps(self, counts=False):
        """
        Return a series of novel fingerprints (subs only).
        """
        self.get_mol_fingerprints()
        mols = self.md.clean_mols()
        s_fps = mols.apply(lambda x: x.get_novel_fp().get_sub_fp(counts))
        return s_fps
  
    def get_bridge_fps(self, counts=False):
        """
        Return a series of novel fingerprints (bridges only).
        """        
        self.get_mol_fingerprints()
        mols = self.md.clean_mols()
        b_fps = mols.apply(lambda x: x.get_novel_fp().get_bridge_fp(counts))
        return b_fps
    
    def group_fp_query(self, g_query, complete, counts):
        """
        Return a series of novel fingerprints (groups only).
        """       
        if counts:
            g_fps = self.get_group_fps(counts=True)
        else:
            g_fps = self.get_group_fps()
        isin = g_fps.apply(self._check_sets_equal, args=(g_query, complete))
        isin = isin[isin]
        ret_mols = self._mols[isin.index]
        return ret_mols
    
    def sub_fp_query(self, s_query, complete, counts):
        
        if counts:
            s_fps = self.get_sub_fps(counts=True)
        else:
            s_fps = self.get_sub_fps()
        isin = s_fps.apply(self._check_sets_equal, args=(s_query, complete))
        isin = isin[isin]
        ret_mols = self._mols[isin.index]
        return ret_mols  
 
    def bridge_fp_query(self, b_query, complete, counts):
        
        if counts:
            b_fps = self.get_bridge_fps(counts=True)
        else:
            b_fps = self.get_sub_fps()
        isin = b_fps.apply(self._check_sets_equal, args=(b_query, complete))
        isin = isin[isin]
        ret_mols = self._mols[isin.index]
        return ret_mols  
    
    def mols_same_group(self, counts=False, pattern='', exclude_benz=False):
        if pattern:
            mol_patt_ids = self.md.find_mols_with_pattern(
                pattern, clean_only=True).index
            
            g_fps = self.get_group_fps(
                counts, exclude_benz=exclude_benz)
            mol_ids = list(set.intersection(set(mol_patt_ids), 
                                       set(g_fps.index)))
            g_fps = g_fps[mol_ids]
        else:
            g_fps = self.get_group_fps(counts, exclude_benz=exclude_benz)
        g_fps = g_fps.apply(self._prep_fp_for_grouping)
        tmp_dict = g_fps.groupby(g_fps).groups
        groups_dict = {key: value for key, value in tmp_dict.items()
                       if len(value) > 1}
        return groups_dict

    def mols_with_fp(self, fp, counts=True):
        
        groups = fp.get('g')
        bridges = fp.get('b')
        subs = fp.get('s')
        sets = []
        if groups:
            s1 = set(self.group_fp_query(set(groups), 
                                         complete=False, counts=counts).index)
            sets.append(s1)
        if bridges:  
            s2 = set(self.bridge_fp_query(set(bridges), 
                                          complete=False, counts=counts).index)
            sets.append(s2)
        if subs:
            s3 = set(self.sub_fp_query(set(subs), 
                                       complete=False, counts=counts).index)
            sets.append(s3)
        mol_ids = list(set.intersection(*tuple(sets)))
        return self._mols[mol_ids]
    
    def mols_same_except_one(self, except_group):
        sim_mols = self.mols_same_group()
        total_dict = defaultdict(set)
        for key, mol_ids in sim_mols.items():
            
            mols = self._mols[mol_ids]

            for mol in mols:
                set_1 = mol.get_novel_fp().get_group_fp(counts=True)
                for mol_c in mols:
                    set_2 = mol_c.get_novel_fp().get_group_fp(counts=True)
                    diff = list(set.difference(set_1, set_2))
                    sim = list(set.intersection(set_1, set_2))
                    if len(diff) == 1:
                        if diff[0][0] == 2:
                            continue
                        if except_group and diff[0][0] != except_group:
                            continue
                        label = f'{str(sim)}_{str(diff[0][0])}'
                        total_dict[label].update(
                            [mol.get_id(), mol_c.get_id()])
        return total_dict
    
    def mols_diff_topology(self):
        fps = self.get_full_fps()
        fps = fps.apply(self._prep_fp_for_grouping, args=(True,))
        tmp_dict = fps.groupby(fps).groups
        top_dict = {key: value for key, value in tmp_dict.items()
           if len(value) > 1}
        return top_dict
    
    def same_fp_except_sub(self, sub_id):
        fps = self.get_full_fps(as_set=False)
        fps = fps.apply(self._remove_sub, args=(sub_id,))
        fps = fps.apply(self._prep_fp_for_grouping, args=(True,))
        tmp_dict = fps.groupby(fps).groups
        mols_with_sub = self.cd.get_sub_mols(sub_id)
        ret_dict = {fp: mol_ids for fp, mol_ids in tmp_dict.items()
           if any(item in mols_with_sub for item in mol_ids) and len(mol_ids) > 1}
        return ret_dict
        
                      
class NovelFingerprint:
    """
    The class representing the novel fingerprint object
    """
    
    def __init__(self, groups, subs, bridges, *args):
        self._g_dict = groups
        self._s_dict = {key: tuple(subs[key]) for key, val in subs.items()}
        self._b_dict = bridges
        self._groups, self._bridges, self._subs = args
        
    def get_group_fp(self, counts=False, as_set=True):
        if not as_set:
            return self._g_dict
        if counts:
            groups = set(self._g_dict.items())
        else:
            groups = set(self._g_dict)
        return groups

        
    def get_sub_fp(self, counts=False, as_set=True):
        if not as_set:
            return self._s_dict
        if counts:
            subs = set(self._s_dict.items())
        else:
            subs = set(self._s_dict)
        return subs
    
    def get_bridge_fp(self, counts=False, as_set=True):
        if not as_set:
            return self._b_dict
        if counts:
            bridges = set(self._b_dict.items())
        else:
            bridges = set(self._b_dict)
        return bridges
    
    def get_full_fp(self, as_set=True):
        fp_dict = {}
        fp_dict['bridges'] = self.get_bridge_fp(counts=True, as_set=as_set)
        fp_dict['subs'] = self.get_sub_fp(counts=True, as_set=as_set)
        fp_dict['groups'] = self.get_group_fp(counts=True, as_set=as_set)
        return fp_dict
        
    def _prepare_output(self):
        try:
            return self._str_rep
        except AttributeError:
            pass
        ret_str = 'bridges: '
        for id_, qty in self._b_dict.items():
            ret_str += f'({id_}, {qty}), '
        
        ret_str += '\ngroups: '
        for id_, qty in self._g_dict.items():
            ret_str += f'({id_}, {qty}), '
            
        ret_str += '\nsubs: '
        for id_, tup in self._s_dict.items():
            ret_str += f'({id_}, {list(tup)}), '
        ret_str = re.sub(',\s\n|,\s$', '\n', ret_str)
        self._str_rep = ret_str
        return ret_str
    
    def __str__(self):
        ret_str = self._prepare_output()
        return ret_str
    
    def _repr_png_(self):
        print(self.__str__())
        mols = []
        lgnds = []
        
        for g in self._groups:
            mols.append(g.get_rdk_mol())
            lgnds.append(f'group {g.get_id()}')    
           
        for s in self._subs:
            mols.append(s.get_rdk_mol())
            lgnds.append(f'substituent {s.get_id()}')    
           
        for b in self._bridges:
            mols.append(b.get_rdk_mol())
            lgnds.append(f'bridge {b.get_id()}')
            
        stream = draw_to_png_stream(mols, lgnds, to_disc=False)

        return stream
    
class BasicFingerprint:
    """
    A class that exposes common fingerprint schemes.
    
    If each molecule has its own BasicFingerprint object
    then it can easily access a number of common fingerprint
    encodings.
    """
    def __init__(self, mol=None):
        self._fp_dict = {}
        self.fp_map = {"MACCS": self._calc_MACCS, "morgan": self._calc_morgan,
                  "rdk": self._calc_rdk, "morgan_feat": self._calc_morgan_feat,
                  "topology": self._calc_topology}
        self._mol = mol
    
    def _calc_all(self):
        for method, fp_func in self.fp_map.items():
            if not method in self._fp_dict:
                fp_func()

    def _calc_MACCS(self):
        self.MACCS = MACCSkeys.GenMACCSKeys(self._mol)
        self._fp_dict['MACCS'] = self.MACCS
        
    def _calc_morgan(self):
        self.morgan = Chem.GetMorganFingerprintAsBitVect(self._mol, 
                                                         2, nBits=1024)
        self._fp_dict['morgan'] = self.morgan
        
    def _calc_rdk(self):
        self.rdk = Chem.RDKFingerprint(self._mol)
        self._fp_dict['rdk'] = self.rdk
        
    def _calc_morgan_feat(self):
        self.morgan_feat = Chem.GetMorganFingerprintAsBitVect(
            self._mol ,2, nBits=1024, useFeatures=True)
        self._fp_dict['morgan_feat'] = self.morgan_feat
        
    def _calc_topology(self):
        self.topology = Chem.GetMorganFingerprintAsBitVect(
            self._mol, 2, nBits=1024, 
            invariants=[1]*self._mol.GetNumAtoms())
        self._fp_dict['topology'] = self.topology
        
    def get_fp(self, fp_type):
        if fp_type == "all":
            self._calc_all()
            return self._fp_dict
        try:
            return self._fp_dict[fp_type]
        except KeyError:
            fp_func = self.fp_map[fp_type]
            fp_func()
        return self._fp_dict[fp_type]
    
           