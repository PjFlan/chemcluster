# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.abspath('..'))

from unittest.mock import patch

def helper_set_up(object_,source):
        logger_patcher = patch(source+'.MyLogger')
        fh_patcher = patch(source+'.MyFileHandler')
        config_patcher = patch(source+'.MyConfig')
        

        object_.mock_logger = logger_patcher.start().return_value.get_child.return_value
        object_.mock_fh = fh_patcher.start().return_value
        object_.mock_config = config_patcher.start().return_value
        
        object_.addCleanup(logger_patcher.stop)
        object_.addCleanup(fh_patcher.stop)
        object_.addCleanup(config_patcher.stop)
