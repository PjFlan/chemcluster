#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 14:40:36 2020

@author: padraicflanagan
"""
# class ListNode:
#   def __init__(self, val=0, next=None):
#       self.val = val
#       self.next = next
      
# class Solution:
#     def swapPairs(self, head: ListNode) -> ListNode:       
#         first = head
#         second = head.next
#         third = second.next
#         head = second
#         while third:
#             second.next = first
#             first.next = third
#             first = third
#             second = third.next
#             third = second.next
#         second.next = first
#         first.next = None
#         return head
    
# string = '[#6]-[#8]-[#15](=[#8])(-[#8]-[#6])'
# i = len(string) - 1
# for s in reversed(string):
#     if s == ']':
#         break
#     i -= 1
# string = list(string)
# string[i] = ';D1' + string[i]
# print(''.join(string))
import rdkit.Chem as Chem
from rdkit.Chem import BRICS

mol = Chem.MolFromSmiles('Cc1ccc2cc3c4ccc(-c5ccc(-c6ccc(C=C(C#N)C(=O)O)s6)s5)cc4c4ccccc4n3c2c1')
frags = list(BRICS.BRICSDecompose(mol, returnMols=True))
for i,f in enumerate(frags):
    for atom in f.GetAtoms():
        atom.SetIsotope(0)
    params = Chem.AdjustQueryParameters()
    params.makeBondsGeneric = True
    params.makeDummiesQueries = True
    rogue_frag = Chem.AdjustQueryProperties(f, params)
   
    print("(%d)"%i, Chem.MolToSmiles(rogue_frag), mol.HasSubstructMatch(rogue_frag))
