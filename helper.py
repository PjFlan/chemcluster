#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 14:40:36 2020
@author: padraicflanagan
"""

import os
try:
   import cPickle as pickle
except:
   import pickle
from datetime import datetime
import json
import logging

import pandas as pd
import rdkit.RDLogger as RDLogger
from rdkit.Chem.Draw import rdMolDraw2D


def draw_to_svg_stream(rdk_mol):
    d2svg = rdMolDraw2D.MolDraw2DSVG(300,300)
    d2svg.DrawMolecule(rdk_mol)
    d2svg.FinishDrawing()
    return d2svg.GetDrawingText()

def draw_to_png_stream(mols, full_size, sub_img_size, font_size, legends=[]):
    d2d = rdMolDraw2D.MolDraw2DCairo(full_size[0], full_size[1], sub_img_size[0], sub_img_size[1])
    d2d.drawOptions().legendFontSize = font_size
    d2d.DrawMolecules(mols,legends=legends)
    d2d.FinishDrawing()
    return d2d.GetDrawingText()

class MyLogger():
    
    class __MyLogger:
        
        _DIR = './'
        _LEVEL = 10
        
        def __init__(self):
            self.configure()

        def configure(self):
            self.config = MyConfig()
            self._DIR = self.config.get_directory("log")
            self._LEVEL = self.config.get_logging("level")
            self.create_root()
            
        def create_root(self):
            logger = logging.getLogger()
            ch = self.create_file_handler()
            ch.setLevel(self._LEVEL)
            formatter = self.create_formatter()
            ch.setFormatter(formatter)
            logger.addHandler(ch)
            self._root = logger
        
        def get_root(self):
            return self._root
        
        def get_child(self, name):
            child = self._root.getChild(name)
            child.setLevel(logging.DEBUG)
            return child
            
        def create_formatter(self):
            format_ = '%(asctime)s - %(name)s - %(message)s'
            datefmt='%Y-%m-%d %H:%M'
            return logging.Formatter(format_, datefmt)
        
        def create_file_handler(self):
            if not os.path.isdir(self._DIR):
                os.mkdir(self._DIR)
            log_file = self._DIR + 'main.log'
            return logging.FileHandler(log_file)
        
    _instance = None
        
    def __new__(cls):
        if not MyLogger._instance:
            MyLogger._instance = MyLogger.__MyLogger()
        return MyLogger._instance 
    
    def __getattr__(self, name):
        return getattr(self._instance, name)
    

class MyFileHandler:
    
    def output_to_text(self, records, file, delim=','):
        directory = os.path.dirname(file)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(file,'w') as fh:
            for row in records:
                if isinstance(row,list):
                    row = delim.join(str(x) for x in row)
                row += '\n'
                fh.write(row)
            
    def load_from_text(self, file, delim=','):
        records = []
        with open(file,'r') as fh:
            for line in fh:
                row = line.rstrip().split(delim)
                if len(row) == 1:
                    row = row[0]
                records.append(row)
        return records
    
    def load_from_json(self, file):
        with open(file) as json_file:
            dict_ = json.load(json_file)
        return dict_
    
    def output_to_json(self, dict_, file):
        with open(file,'w') as json_file:
            json.dump(dict_, json_file)
            
    def load_from_pickle(self, file):
        with open(file,'rb') as ph:
            depickled = pickle.load(ph)
        return depickled
            
    def dump_to_pickle(self, item, file):
        with open(file,'wb') as ph:
            pickle.dump(item, ph)        
            
            
class MyConfig:

    class __MyConfig:
        _CONFIG = 'config.json'

        def __init__(self):
            self.configure()
        
        def configure(self):
            self.params_config()
            self.pandas_config()
            self.rdkit_config()
            
        def params_config(self):
            if not os.path.isfile(self._CONFIG):
                raise MyConfigFileError(self._CONFIG)
            with open(self._CONFIG) as config_file:
                self.params = json.load(config_file)
            try:
                self._ROOT = self.params["dirs"]["root"]
            except KeyError:
                self._ROOT = './'
            
        def pandas_config(self):
            pd.set_option('display.max_rows', 510)
            pd.set_option('display.max_columns', 500)
            pd.set_option('display.width', 1000)

        def rdkit_config(self):
            RDLogger.DisableLog('rdApp.*') 
            
        def get_directory(self, dir_):
            try:
                path = self._ROOT + self.params["dirs"][dir_]
            except KeyError:
                raise MyConfigParamError(dir_)
            return path
        
        def get_logging(self, param):
            try:
                value = self.params["logging"][param]
            except KeyError:
                raise MyConfigParamError(param)
            return value
        
        def get_db_source(self):
            try:
                info = self.params["database"]["source"]
            except KeyError:
                raise MyConfigParamError("database.source")
            return info
        
        def get_regen(self, file_type):
            try:
                regen_flag = self.params["regenerate"][file_type]
            except KeyError:
                regen_flag = 0
            return regen_flag
        
        def get_comp_thresholds(self, entity):
            try:
                cleaning = self.params["cleaning"][entity]
            except KeyError:
                raise MyConfigParamError("cleaning.{entity}")
                cleaning = {}
            return cleaning
        
    _instance = None
        
    def __new__(cls):
        if not MyConfig._instance:
            MyConfig._instance = MyConfig.__MyConfig()
        return MyConfig._instance
    
    def __getattr__(self, name):
        return getattr(self._instance, name)
    

class MyConfigError(Exception):
    
    def __init__(self, msg=None):
        super().__init__(msg)
        
        
class MyConfigParamError(MyConfigError):
    
    def __init__(self, missing=''):
        msg = f"The parameter '{missing}' is not provided in config.json."
        super().__init__(msg)
        
        
class MyConfigFileError(MyConfigError):
    
    def __init__(self, file=''):
        msg = f"File not found. Please ensure the path '{file}' exists."
        super().__init__(msg)
        
        
class NoDataError(Exception):
    
    def __init__(self):
        msg = "Link tables not populated. Set 'grouping' flag in config.json to 1 and run AbsorboMatic.set_up()."
        super().__init__(msg)
      

class FingerprintNotSetError(Exception):
    
    def __init__(self):
        msg = "The absorbo-specific fingerprint has not been created for this molecule.\
        Run AbsorboFingerprint.fingerprint_mol to create."
        super().__init__(msg)