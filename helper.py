"""
This module defines a number of helper utilites
that handle tasks such as writing to and loading
from disk, getting config options, logging functionality
and and bespoke error classes.
"""
import os, shutil

try:
    import cPickle as pickle
except:
    import pickle

import json
import logging

import pandas as pd
import rdkit.RDLogger as RDLogger


class MyLogger:
    class __MyLogger:

        _DIR = "./"
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
            format_ = "%(asctime)s - %(name)s - %(message)s"
            datefmt = "%Y-%m-%d %H:%M"
            return logging.Formatter(format_, datefmt)

        def create_file_handler(self):
            if not os.path.isdir(self._DIR):
                os.mkdir(self._DIR)
            log_file = self._DIR + "main.log"
            return logging.FileHandler(log_file)

    _instance = None

    def __new__(cls):
        if not MyLogger._instance:
            MyLogger._instance = MyLogger.__MyLogger()
        return MyLogger._instance

    def __getattr__(self, name):
        return getattr(self._instance, name)


class MyFileHandler:
    def output_to_text(self, records, file, delim=","):
        directory = os.path.dirname(file)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(file, "w") as fh:
            for row in records:
                if isinstance(row, list):
                    row = delim.join(str(x) for x in row)
                row += "\n"
                fh.write(row)

    def load_from_text(self, file, delim=","):
        records = []
        with open(file, "r") as fh:
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
        with open(file, "w") as json_file:
            json.dump(dict_, json_file)

    def load_from_pickle(self, file):
        with open(file, "rb") as ph:
            depickled = pickle.load(ph)
        return depickled

    def dump_to_pickle(self, item, file):
        directory = os.path.dirname(file)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(file, "wb") as ph:
            pickle.dump(item, ph)

    def clean_dir(self, folder):

        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print("Failed to delete %s. Reason: %s" % (file_path, e))


class MyConfig:
    class __MyConfig:
        _CONFIG = "config.json"

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
                self._ROOT = "./"

        def pandas_config(self):
            pd.set_option("display.max_rows", 510)
            pd.set_option("display.max_columns", 500)
            pd.set_option("display.width", 1000)

        def rdkit_config(self):
            RDLogger.DisableLog("rdApp.*")

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

        def get_flag(self, flag):
            return self.params["flags"][flag]

        def get_comp_thresholds(self, entity):
            try:
                cleaning = self.params["cleaning"][entity]
            except KeyError:
                raise MyConfigParamError("cleaning.{entity}")
                cleaning = {}
            return cleaning

        def use_tmp(self):
            use_tmp = self.params["tmp"]
            return use_tmp

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
    def __init__(self, missing=""):
        msg = f"The parameter '{missing}' is not provided in config.json."
        super().__init__(msg)


class MyConfigFileError(MyConfigError):
    def __init__(self, file=""):
        msg = f"File not found. Please ensure the path '{file}' exists."
        super().__init__(msg)


class NoLinkTableError(Exception):
    def __init__(self):
        msg = "Link tables not populated. "
        msg += "Set 'grouping' flag in config.json to 1 and run Chemformatic.set_up()."
        super().__init__(msg)


class FingerprintNotSetError(Exception):
    def __init__(self):
        msg = "The novel fingerprint has not been created for this molecule. "
        msg += "Please generate novel finerprints first."
        super().__init__(msg)


class CompDataNotSetError(Exception):
    def __init__(self):
        msg = "This molecule either has no computational data "
        msg += "or the computational data has not yet been fetched from the database."
        super().__init__(msg)
