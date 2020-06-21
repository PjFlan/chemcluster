# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.abspath('..'))

import unittest
from unittest.mock import patch,mock_open,call

from data import MongoLoad
from helper import MyConfigParamError
from set_up import helper_set_up

class TestMongoLoad(unittest.TestCase):
    
    def setUp(self):
        helper_set_up(self,'data')
        
        client_patcher = patch('data.MongoClient')
        self.mock_client = client_patcher.start().return_value
        self.addCleanup(client_patcher.stop)
        
    @patch('data.MyLogger')
    def test_mongo_connect(self,mock_parent_logger):
        self.mock_config.get_db_source.return_value = {"database":"testdb","collection":"testcol"}
        MongoLoad()
        self.mock_client.__getitem__.assert_called_with("testdb")
        mock_parent_logger.return_value.get_child.assert_called_with("MongoLoad")
        
    @patch.object(MongoLoad,'get_smiles')
    def test_resolve_query(self,mock_smiles):
        mongo = MongoLoad()
        mongo.resolve_query('smiles')
        mock_smiles.assert_called_once()

    @patch('os.path.isfile')
    def test_get_smiles(self,mock_isfile):
        smiles = ['smiles1','smiles2']
        self.mock_fh.load_from_text.return_value = smiles
        collection = self.mock_client.__getitem__.return_value.__getitem__.return_value
        collection.distinct.return_value = smiles
        
        self.mock_config.get_reload.side_effect = [0,0,1]
        self.mock_config.get_directory.side_effect = [MyConfigParamError(),'path/to/smiles','path/to/smiles']
        mock_isfile.side_effect = [True,False,True]
        mongo = MongoLoad()
        
        #Test 1 - Config SMILES param doesn't exist
        mongo.get_smiles()
        self.mock_logger.warning.assert_called()
        self.mock_fh.load_from_text.assert_called()
    
        #Test 2 - Test file does not exists so loads from DB
        mongo.get_smiles()
        collection.distinct.assert_called_with("PRISTINE.SMI")
        self.mock_fh.output_to_text.assert_called_with(smiles,'path/to/smiles')
        
        #Test 3 - Test reload flag set to True so reloads from DB
        self.mock_fh.load_from_text.reset_mock()
        mongo.get_smiles()
        self.mock_fh.load_from_text.assert_not_called()
        
if __name__ == '__main__':    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMongoLoad)
    unittest.TextTestRunner(verbosity=2).run(suite)