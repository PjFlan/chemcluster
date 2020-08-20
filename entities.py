"""
The classes for each entity are defined here.
The different entities are: molecules, fragments,
groups, substituents and bridges. Each entity object
must have a valid SMILES representation, even if dummy
atoms are included.
"""
import re

from rdkit.Chem import AllChem as Chem

from rdkit.Chem import Descriptors as Descriptors_
from rdkit.Chem import rdMolDescriptors as Descriptors
from rdkit.Chem import BRICS

from fingerprint import BasicFingerprint
from helper import FingerprintNotSetError, CompDataNotSetError
from drawing import draw_to_png_stream

        
class Entity:
    """
    Superclass for the Entity child classes.
    
    An entity is any chemical compound that has a 
    valid SMILES representation. This includes molecules,
    fragments, substituents etc.
    
    Attributes
    ----------
    SMARTS_mols : dict
        {SMARTS string : RDKit.Mol representation}
        Updated dynamically as new patterns are defined. 
        Saves having to create the Mol object again.
    occurrence: int
        the number of molecules in the database containing
        this entity substructure
    
    Methods
    -------
    basic_fingerprint(fp_type='all')
        calculates the common fingerprint scheme
        encondings of this entity
    get_rdk_mol()
        return the RDKit.Mol representation
    pattern_count(pattern):
        number of times 'pattern' occurs in entity
    has_pattern(pattern):
        check if entity contains 'pattern'
    get_size():
        the number of heavy atoms contained in the entity
    get_id():
        unique ID of the entity. Each entity child class
        has its own set of IDs starting from 0
    set_id()
    """
    
    SMARTS_mols = {}
    occurrence = None
    
    def __init__(self, smiles, id_):
        """
        
        Parameters
        ----------
        smiles : str
            SMILES representation of the entity.
        id_ : int
            unique identifier for the entity

        Returns
        -------
        None.

        """
        self.smiles = smiles
        self._id = id_
        self._size = 0
        self._mol = None
        
    def _remove_stereo(self, smiles):
        """
        Strip stereoatom notation from a SMILES string
        
        """
        new = re.sub('\[(\w+)@+H*\]',r'\1',smiles)
        return new
    
    def _remove_link_atoms(self, smiles):
        """
        Strip BRICS link atom isotopes from a SMILES string
        """
        new = re.sub('\[[0-9]+\*\]','C',smiles)
        return new
    
    def _shrink_alkyl_chains(self, mol):
        """
        Reduce lonk alkyl chains to a single hydrogen

        Parameters
        ----------
        mol : RDKit.Mol
            molecule object whose alkyl chains need shrinking

        Returns
        -------
        mol : RDKit.Mol
            molecule object with no long alkyl chains

        """
        alkyl_SMARTS = '[C;R0;D1;$(C-[#6,#7,#8;-0])]'
        try:
            patt = self.SMARTS_mols[alkyl_SMARTS]
        except KeyError:
            patt = Chem.MolFromSmarts(alkyl_SMARTS)
            self.SMARTS_mols[alkyl_SMARTS] = patt
        #check if has any alkyl carbons
        matches = mol.GetSubstructMatches(patt)
        if not matches:
            return mol
        mol = Chem.RWMol(mol) #Same as RDKit.Mol but added features
        for atom in matches:
            #replace carbon with hydrogen
            mol.ReplaceAtom(atom[0], Chem.Atom(1))
        smi = Chem.MolToSmiles(mol)
        mol = Chem.MolFromSmiles(smi)
        #recursively keep going until all carbons replaced
        mol = self._shrink_alkyl_chains(mol)
        return mol  
    
        
    def basic_fingerprint(self, fp_type='all'):
        """
        

        Parameters
        ----------
        fp_type : str, optional
            Which encoding scheme to use. The default is 'all'.

        Returns
        -------
        fps : BasicFingerprint
            object providing access to the entity fingerprint
            using scheme requested by fp_type

        """
        try:
            return self.fp.get_fp(fp_type)
        except AttributeError:
            self.base_fp = BasicFingerprint(self.get_rdk_mol())
            fps = self.base_fp.get_fp(fp_type)
        return fps
    
    def get_rdk_mol(self):
        """
        RDKit.Mol representation of the entity
        """
        if self._mol:
            return self._mol
        self._mol = Chem.MolFromSmiles(self.smiles)
        return self._mol
    
    def pattern_count(self, pattern):
        """
        number of occurrences of pattern in the entity

        Parameters
        ----------
        pattern : RDKit.Mol
            RDKit.Mol representation of the SMARTS pattern
            to be searched for

        Returns
        -------
        tuple:
            a tuple of matches with each match itself a tuple
            of the atom indices in the pattern

        """
        matches = self.get_rdk_mol().GetSubstructMatches(pattern)
        return len(matches)
    
    def has_pattern(self, pattern):
        """
        check if entity contains the pattern

        Parameters
        ----------
        pattern : RDKit.Mol
            RDKit.Mol representation of the SMARTS pattern
            to be searched for

        Returns
        -------
        boolean
            has the pattern or not

        """
        return self.get_rdk_mol().HasSubstructMatch(pattern)
    
    def get_size(self):
        """
        number of heavy atoms in entity
        """
        if self._size:
            return self._size
        self._size = Descriptors_.HeavyAtomCount(self.get_rdk_mol())
        return self._size
    
    def get_id(self):
        """
        unique identifier for this entity
        """
        return self._id
    
    def _repr_png_(self):

        legends = []
        return draw_to_png_stream([self.get_rdk_mol()], legends)
    
        
class Fragment(Entity):
    """
    A class used to represent a BRICS fragment.
    
    A BRICS fragment is a valid molecule with 
    a valid SMILES. Link atoms are represented using
    isotope notation.

    Methods
    -------
    get_direct_subs()
        find direct substituents if fragment is
        isolated benzene ring
    convert_to_group()
        transform fragment into a core group
    set_group()
        set ID of the Group object to which this fragment
        reduces
    get_group()
        return ID of Group object to which this fragment
        reduces
    """
    
    def __init__(self, smiles, id_):
        super().__init__(smiles, id_)
        self._group = None #ID of core group it reduces to
        
    def _remove_redundant(self):
        """
        strip down to its core so can determine its group
        
        This involves removing anything that only complicates
        the molecule. Removes BRICS link atoms, removes stereo
        information and shrinks alkyl chains. This makes it easier
        to determine which group the fragment belongs to.

        Returns
        -------
        new_smiles : str
            The reduced version of the SMILES without redundant
            information

        """
        new_smiles = self._remove_link_atoms(self.smiles)
        new_smiles = self._remove_stereo(new_smiles)
        new_mol = Chem.MolFromSmiles(new_smiles)
        new_mol = self._shrink_alkyl_chains(new_mol)
        new_smiles = Chem.MolToSmiles(new_mol)
        return new_smiles
    
    def _get_core(self, as_mol=False):
        """
        remove the redundant atoms and SMILES information

        Parameters
        ----------
        as_mol : boolean, optional
            return a RDKit.Mol (true) or a SMILES string (false)

        Returns
        -------
        core : str/RDKit.Mol
            the fragment with unimportant information removed

        """
        try:
            core = self._core
        except AttributeError:
            core = self._remove_redundant()
            self._core = core
        if as_mol:
            return Chem.MolFromSmiles(core)
        return core
    
    def get_direct_subs(self):
        """
        determines the direct substituents attached to fragment
        
        Direct substituents are small substituents like halogens
        and alcohols. These are not cleaved by BRICS. This method only
        looks for substituents if the fragment is a single benzene ring.
        
        Returns
        -------
        list
            a list of SMILES strings representing the
            direct substituents on this fragment

        """
        benzene_SMARTS = 'c1ccccc1'
        try:
            patt = self.SMARTS_mols[benzene_SMARTS]
        except KeyError:
            patt = Chem.MolFromSmarts(benzene_SMARTS)
            self.SMARTS_mols[benzene_SMARTS] = patt
        core_mol = self._get_core(as_mol=True)
        if not core_mol:
            return []
        if Descriptors.CalcNumRings(core_mol) != 1:
            return []
        if not core_mol.HasSubstructMatch(patt):
            return []
        subs = Chem.ReplaceCore(core_mol, patt)
        smiles = Chem.MolToSmiles(subs)
        smiles = re.sub('\[[0-9]+\*\]', '', smiles)
        res = smiles.split('.')
        try:
            res.remove('')
        except ValueError:
            pass
        return res
    
    def convert_to_group(self, direct_subs):
        """
        convert BRICS fragment into its group
        
        A fragment SMILES has lots of redundant info such as
        stereochem, link atoms and alkyl chains. This info is not
        needed to determine the  core group e.g. two fragments with
        different link atoms can still have the same group.

        Parameters
        ----------
        direct_subs : list
            a list of all the common direct substituents in
            the database, expressed in SMILES notation.

        Returns
        -------
        can_smi : str
            canonical SMILES representation of the core group.

        """
        core_mol = self._get_core(as_mol=True)
        for sub, patt in direct_subs.items():
            #remove the direct subsituent if present
            rm = Chem.DeleteSubstructs(core_mol, patt)
            rm_smi = Chem.MolToSmiles(rm)
            if '.' in rm_smi:
                #if after removing the sub the rm_smi has a '.'
                #it must not correspond to a direct substituent on this 
                #molecule since when removed it shouldn't fragment the molecule.
                #Ignore this sub and move on.
                continue
            core_mol = Chem.MolFromSmiles(rm_smi)
        can_smi = Chem.MolToSmiles(core_mol)
        return can_smi
    
    def set_group(self, group_id):
        self._group = group_id
        
    def get_group(self):
        return self._group


class Group(Entity):
    """
    A class used to represent a core group.
    
    Multiple BRICS fragments all reduce to the same core group.
    A core group must have at least one ring and at least one
    conjugated carbon.
    
    Attributes
    ----------
    is_taa : boolean
        flags if group is a triarylamine substructure since
        this structure isnt preserved by BRICS - needs manual
        treatment
    
    Methods
    -------
    set_cluster()
        store ID of cluster to which group belongs when groups
        are clustered
    get_cluster()
        return ID of the cluster this group is in
    """
    
    def __init__(self, smiles, id_):
        super().__init__(smiles, id_)
        self._cluster = None
        self.is_taa = False

    def set_cluster(self, cluster_id):
        self._cluster = cluster_id
        
    def get_cluster(self):
        return self._cluster
        
    
class Molecule(Entity):
    """
    A class used to represent a molecule from the database.
    
    The molecules in the database are stored in SMILES notation.
    This class provides a number of helper methods and attributes
    to make manipulation of these molecules easier.
    
    
    Methods
    -------
    fragment()
        apply BRICS fragmentation algorithm to the molecule
    set_comp_data()
        store the sTDA computational data of this molecule
    get_comp_data()
        get the sTDA computational data of this molecule
    set_conjugation()
        set the size of the largest conjugated chain
    get_conjugation()
        get the size of the largest conjugated chain
    get_lambda_max()
        get the wavelength of the strongest excitation
    get_strength_max()
        get the strength of the strongest excitation
    get_legend()
        prepare legend for drawing the molecule
    set_novel_fp(nfp)
        store the NovelFingeprint object of this class
    get_novel_fp()
        get the NovelFingerprint object of this class
    """
    
    def __init__(self, smiles, id_, mol_rdk=None):
        super().__init__(smiles, id_)
        if mol_rdk:
            self._mol = mol_rdk
        self._lam_max = self._osc_max = None

    def fragment(self):
        """
        apply the BRICS algorithm to this molecule.

        Returns
        -------
        leaf_frags : list
            list of the SMILES representations of
            the leaf fragments of this molecule

        """
        leaf_frags = []
        if self.get_rdk_mol():
            leaf_frags = BRICS.BRICSDecompose(self.get_rdk_mol(),
                                              keepNonLeafNodes=False)
        return leaf_frags
    
    def set_comp_data(self, comp):
        """
        store wavelengths and strengths from sTDA calculations.
        
        the first three excitations are retrieved from the database
        and the amplitude and oscillator strengths are stored. The 
        strongest excitation is determined and stored separately.

        Parameters
        ----------
        comp : dict
            a dictionary with details of the first 3 excitations
            (amplitudes and strengths)

        Returns
        -------
        None.

        """
        strengths = comp['strength']
        lambdas = comp['lambda']
        max_idx = strengths.index(max(strengths))
        lam_max = lambdas[max_idx]
        osc_max = strengths[max_idx]
        self._lambdas = lambdas
        self._strengths = strengths
        self._lam_max, self._osc_max = lam_max, osc_max
        
    def get_comp_data(self):
        try:
            comp_dict = {'lambdas': self._lambdas, 
                         'strengths': self._strengths}
            return comp_dict
        except AttributeError:
            raise CompDataNotSetError()
        
    def set_conjugation(self, conj_bonds):
        """
        store number of bonds in longest conjugated region.

        Parameters
        ----------
        conj_bonds : int
            the number of bonds in the longest conjugated
            region

        Returns
        -------
        None.

        """
        self._conjugation = conj_bonds
        
    def get_conjugation(self):
        return self._conjugation
    
    def get_lambda_max(self):
        return self._lam_max
    
    def get_strength_max(self):
        return self._osc_max
    
    def get_legend(self, dev_flag=True):
        """
        produce the legend for when drawing the molecule.

        Parameters
        ----------
        dev_flag : boolean, optional
            if in development mode, print the molecule ID as well.
            The default is True.

        Returns
        -------
        str
            the molecule legend containing its comp data.
        """
        if dev_flag:
            return f'{self.get_id()} ; {self._lam_max} nm ; {self._osc_max:.4f}'\
                if self._lam_max else ''
        
        return f'{self.lambda_max} nm ; {self.strength_max:.4f}'\
            if self.lambda_max else ''
    
    def set_novel_fp(self, nfp):
        self._nfp = nfp
        
    def get_novel_fp(self):
        try:
            return self._nfp
        except AttributeError:
            raise FingerprintNotSetError()
            
    def _repr_png_(self):

        legends = [f'{self.get_lambda_max()} nm ;  {self.get_strength_max():.3f}']
        return draw_to_png_stream([self.get_rdk_mol()], legends)
    

class Bridge(Entity):
    """
    A class used to represent a bridge entity.
    
    A bridge entity connects two Group entities within a 
    Molecule entity. This class has no methods of its own 
    but provides a wrapper around the parent Entity class so 
    that bridge objects can be distinguished and have their 
    own ID system.
    """
    def __init__(self, smiles, id_):
        smiles = self._remove_stereo(smiles)
        super().__init__(smiles, id_)
        

class Substituent(Entity):
    """
    A class used to represent a substituent entity.
    
    A substituent entity branches off a Group entity in
    a molecule entity. This class has no methods of its own 
    but provides a wrapper around the parent Entity class so 
    that substituent objects can be distinguished and have their 
    own ID system.
    """
    def __init__(self, smiles, id_):
        smiles = self._remove_stereo(smiles)
        super().__init__(smiles, id_)