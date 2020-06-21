# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.abspath('..'))
from io import StringIO

import unittest
from unittest.mock import patch,mock_open,call

from helper import MyConfig,MyLogger,MyFileHandler,MyConfigFileError,MyConfigParamError

class TestConfig(unittest.TestCase):
        
    def setUp(self):
        config_patcher = patch('helper.MyConfig._MyConfig__MyConfig.configure')
        self.mock_config = config_patcher.start()
        self.addCleanup(config_patcher.stop)
        
    def test_singleton_creation(self):
        obj1 = MyConfig()
        self.mock_config.assert_called()
        obj2 = MyConfig()
        self.assertIs(obj1,obj2)

    @patch('builtins.open', new_callable=mock_open, read_data='{"dirs":{"root":"path/to/root/"}}')
    @patch('os.path.isfile')
    def test_params_config(self,mock_isfile,mo):
        mock_isfile.side_effect = [False,True,True]
        config = MyConfig._MyConfig__MyConfig()
        self.assertRaises(MyConfigFileError,config.params_config)
            
        handlers = (mo.return_value,mock_open(read_data="{}").return_value,)
        mo.side_effect = handlers
        config.params_config()
        self.assertEqual(config.params,{"dirs":{"root":"path/to/root/"}})
        self.assertEqual(config._ROOT,'path/to/root/')
        config.params_config()
        self.assertEqual(config._ROOT,'./')
        
    def test_get_directory(self):
         config = MyConfig._MyConfig__MyConfig()
         config._ROOT = 'path/to/root/'
         config.params = {"dirs":{"log":"logs/"}}
         path = config.get_directory('log')
         self.assertEqual(path,'path/to/root/logs/')
         self.assertRaises(MyConfigParamError,config.get_directory,"pickle")
            
class TestLogger(unittest.TestCase):

    def setUp(self):
        config_patcher = patch('helper.MyLogger._MyLogger__MyLogger.configure')
        logging_patcher = patch('helper.logging')
        self.mock_config = config_patcher.start()
        self.mock_logging = logging_patcher.start()
        self.addCleanup(config_patcher.stop)
        self.addCleanup(logging_patcher.stop)
    
    def test_singleton(self):
        MyLogger()
        self.mock_config.assert_called()
        
    @patch('os.path.isdir')
    def test_create_root(self,mock_isdir):
        mock_isdir.return_value = True
        logger = MyLogger._MyLogger__MyLogger()
        logger._DIR = 'test/dir/'
        logger.create_root()
        self.mock_logging.getLogger.assert_called()
        self.mock_logging.Formatter.assert_called()
        self.mock_logging.FileHandler.assert_called_with('test/dir/main.log')
        
    def test_add_child(self):
        logger = MyLogger._MyLogger__MyLogger()
        logger._root = self.mock_logging
        logger.get_child('child')
        self.mock_logging.getChild.assert_called_with('child')
        
class TestFileHandler(unittest.TestCase):

    def setUp(self):
        self.fh = MyFileHandler()
        
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_output_to_text(self,mo,mock_exists):
        mock_exists.return_value = True
        records = [['abc','def'],['ghi','jkl']]
        self.fh.output_to_text(records,'foo')
        handle = mo()
        calls = [call('abc,def\n'),call('ghi,jkl\n')]
        handle.write.assert_has_calls(calls)
        records = ['test','of','write']
        self.fh.output_to_text(records,'foo')
        calls = [call('test\n'),call('of\n'),call('write\n')]
        handle.write.assert_has_calls(calls)
        
    @patch('builtins.open', new_callable=mock_open)
    def test_load_from_text(self,mo):
        mo.side_effect = [StringIO("some,text,data\nto,be,read"),StringIO("test\none\nrow")]
        result = self.fh.load_from_text('foo')
        self.assertEqual(result,[['some','text','data'],['to','be','read']])
        result = self.fh.load_from_text('foo')
        self.assertEqual(result,['test','one','row'])
        
if __name__ == '__main__':    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestConfig)
    unittest.TextTestRunner(verbosity=2).run(suite)
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLogger)
    unittest.TextTestRunner(verbosity=2).run(suite)
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFileHandler)
    unittest.TextTestRunner(verbosity=2).run(suite)