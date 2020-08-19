#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 11:59:48 2020

@author: padraicflanagan
"""
import math
import os
import pandas as pd

from rdkit.Chem.Draw import rdMolDraw2D
from helper import MyConfig, MyFileHandler

config = MyConfig()
fh = MyFileHandler()

DRAWING_RES_DISC = 600
DRAWING_RES_PUB = 900
FONT_SIZE_DISC = 30
FONT_SIZE_PUB = 40
DRAWING_RES_CONSOLE = 200
FONT_SIZE_CONSOLE = 20
USE_TMP = config.use_tmp()
TMP_DIR = config.get_directory('tmp')
DEV_FLAG = config.get_flag('dev')

def draw_mols_canvas(mols, legends, outdir, suffix='img', img_type='png',
                     clean_dir=True, start_idx=0, per_img=20, per_row=5):
    if USE_TMP:
        outdir = TMP_DIR
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    num_mols = len(mols)
    num_files = math.ceil(num_mols/per_img)

    file_num = 1
    file_begin = 0
    file_end = per_img
    mols = mols.apply(lambda x: x.get_rdk_mol())
    if clean_dir:
        fh.clean_dir(outdir)

    while file_num<=num_files:
        file = f'{suffix}.{file_begin+start_idx+1}-{file_end+start_idx}'
        file = os.path.join(outdir, file)
        curr_mols = mols.iloc[file_begin:file_end].tolist()

        lgnds = legends.iloc[file_begin:file_end].tolist()
        if img_type == 'png':
            file += '.png'
            stream = draw_to_png_stream(curr_mols, lgnds, to_disc=True)
            with open(file, 'wb+') as ih:
                ih.write(stream)
        else:
            file += '.svg'
            stream = draw_to_svg_stream(curr_mols, lgnds, to_disc=True)
            with open(file, 'w+') as ih:
                ih.write(stream)
        
        file_num += 1
        file_begin += per_img
        if file_num == num_files:
            file_end = num_mols
        else:
            file_end += per_img

def draw_to_png_stream(mols, legends=[], to_disc=False):
    if to_disc:
        if DEV_FLAG:
            res, font_size = DRAWING_RES_DISC, FONT_SIZE_DISC
            mpr = 5
        else:
            res, font_size = DRAWING_RES_PUB, FONT_SIZE_PUB
            mpr = 4
    else:
        res, font_size = DRAWING_RES_CONSOLE, FONT_SIZE_CONSOLE
        mpr = 3
    sub_img_size = (res, res)
    cols = math.ceil(len(mols)/mpr)
    rows = mpr if len(mols) >= mpr else len(mols)
    full_size = (rows*res, cols*res)
    d2d = rdMolDraw2D.MolDraw2DCairo(full_size[0], full_size[1], sub_img_size[0], sub_img_size[1])
    d2d.drawOptions().legendFontSize = font_size
    d2d.DrawMolecules(mols, legends=legends)
    d2d.FinishDrawing()
    return d2d.GetDrawingText()  
     
def draw_to_svg_stream(mols, legends=[], to_disc=False):
    if to_disc:
        if DEV_FLAG:
            res, font_size = DRAWING_RES_DISC, FONT_SIZE_DISC
            mpr = 5
        else:
            res, font_size = DRAWING_RES_PUB, FONT_SIZE_PUB
            mpr = 4
    else:
        res, font_size = DRAWING_RES_CONSOLE, FONT_SIZE_CONSOLE
        mpr = 3
    sub_img_size = (res, res)
    cols = math.ceil(len(mols)/mpr)
    rows = mpr if len(mols) >= mpr else len(mols)
    full_size = (rows*res, cols*res)
    d2d = rdMolDraw2D.MolDraw2DSVG(full_size[0], full_size[1], sub_img_size[0], sub_img_size[1])
    d2d.drawOptions().legendFontSize = font_size
    d2d.DrawMolecules(mols, legends=legends)
    d2d.FinishDrawing()
    return d2d.GetDrawingText()  
           

def draw_entities(entities, outdir, from_idx, to_idx):
    freqs = entities.apply(lambda x: x.occurrence)
    freq_df = pd.concat([entities, freqs], axis=1)
    freq_df.columns = ['entity','occurrence']
    freq_df = freq_df.sort_values(by='occurrence', ascending=False).iloc[from_idx:to_idx]
    entities = freq_df['entity']
    legends = freq_df.apply(lambda f: f"id: {f['entity'].get_id()}, freq: {f['occurrence']}", axis=1)
    draw_mols_canvas(entities, legends, outdir=outdir, 
                     start_idx=from_idx, per_row=5)     