#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 14:40:36 2020

@author: padraicflanagan
"""
class ListNode:
  def __init__(self, val=0, next=None):
      self.val = val
      self.next = next
      
class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:       
        first = head
        second = head.next
        third = second.next
        head = second
        while third:
            second.next = first
            first.next = third
            first = third
            second = third.next
            third = second.next
        second.next = first
        first.next = None
        return head