# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.abspath('..'))

import unittest
from unittest.mock import patch,mock_open,call

import pandas as pd
from rdkit.Chem import AllChem as Chem

from entities import Molecule,Fragment
from set_up import helper_set_up

class TestMolecule(unittest.TestCase):

    def setUp(self):
        helper_set_up(self,'entities')
        
        rdk_patcher = patch('entities.Chem',autospec=True)
        self.mock_rdk = rdk_patcher.start()
        self.addCleanup(rdk_patcher.stop)
        
    @patch('os.path.isfile')
    def test_get_molecules(self,mock_isfile):
        mock_isfile.side_effect = [True,False]
        self.mock_config.get_reload.return_value = 0
        
        smiles = ['one','two']
        mols_dict = {'one':'test1','two':'test2'}
        self.mock_fh.load_from_pickle.return_value = mols_dict
        
        molecules = Molecule.get_molecules(smiles)
        self.mock_fh.load_from_pickle.assert_called()
        self.assertEqual(len(molecules),2)
        
        molecules = Molecule.get_molecules(smiles)
        self.mock_rdk.MolFromSmiles.assert_has_calls([call('one'),call('two')])
        
    def test_get_comp_data(self):
        smiles = ['c1ccccc1','CC(=O)O']
        lambdas = [400,450]
        osc = [1,1.2]
        mol1 = Molecule(smiles[0])
        df = pd.DataFrame({'smiles':smiles,'lambda':lambdas,'strength':osc})
        df = df.set_index('smiles')
        
        mol1.set_comp_data(df)
        self.assertEqual(mol1.comp_lambda,400)
        mol2 = Molecule('NotASmiles')
        mol2.set_comp_data(df)
        self.assertIsNone(mol2.comp_lambda)
        
class TestFragment(unittest.TestCase):
    
    def setUp(self):
        helper_set_up(self,'entities')
        
    @patch('entities.Molecule')
    def test_get_fragments(self,mock_mol):
        test_smiles = 'CCOc1nc2nc(c3nc4nc(OCC)c(c(c4c4c3C4)c3ccccc3)C#N)c3c(c2c(c1C#N)c1ccccc1)C3'
        rdk_mol = Chem.MolFromSmiles(test_smiles)
        mock_mol.get_rdk_mol.return_value = rdk_mol
        
        fragments = Fragment.get_fragments(mock_mol)
        self.assertEqual(len(fragments),31)
        num_leafs = 0
        for frag in fragments:
            if frag.is_leaf:
                num_leafs += 1
        self.assertEqual(num_leafs,4)
        
if __name__ == '__main__':    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMolecule)
    unittest.TextTestRunner(verbosity=2).run(suite)
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFragment)
    unittest.TextTestRunner(verbosity=2).run(suite)