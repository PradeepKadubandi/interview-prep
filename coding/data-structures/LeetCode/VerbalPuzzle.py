# https://leetcode.com/problems/verbal-arithmetic-puzzle/

import copy
import itertools

class Solution(object):
    def word_to_int(self, word, char_to_value_map):
        return int(str.join('', [str(char_to_value_map[c]) for c in word]))
    
    def is_valid_mapping(self, words, result, char_to_value_map):
        s = 0
        for w in words:
            s += self.word_to_int(w, char_to_value_map)
        if s == self.word_to_int(result, char_to_value_map):
            return True
        return False
    
    def find_valid_mapping_rec(self, vmap_partial, rem_chars, non_zero_chars):
        if len(vmap_partial) == 10:
            yield copy.deepcopy(vmap_partial)
        else:
            for i, c in enumerate(rem_chars):
                if len(vmap_partial) > 0 or c not in non_zero_chars:
                    vmap_partial.append(c)
                    for m in self.find_valid_mapping_rec(vmap_partial, rem_chars[:i] + rem_chars[i+1:], non_zero_chars):
                        yield m
                    vmap_partial = vmap_partial[:-1]
        
    def isSolvable(self, words, result):
        """
        :type words: List[str]
        :type result: str
        :rtype: bool
        """
        chars = set()
        non_zero_chars = set()
        for w in words + [result]:
            for i, c in enumerate(w):
                chars.add(c)
                if i == 0:
                    non_zero_chars.add(c)
                            
        rem_chars = [c for c in chars]
        assert len(rem_chars) <= 10
        
        if len(rem_chars) < 10:
            padding = 10 - len(rem_chars)
            rem_chars = rem_chars + ([' '] * padding)
            
        vmap_partial = []
        # for mapping in self.find_valid_mapping_rec(vmap_partial, rem_chars, non_zero_chars):
        for mapping in itertools.permutations(rem_chars, len(rem_chars)):
            if mapping[0] in non_zero_chars:
                continue
            char_to_value_map = {c: i for i, c in enumerate(mapping)}
            if self.is_valid_mapping(words, result, char_to_value_map):
                return True
        return False
        

print (Solution().isSolvable(["SEND","MORE"], "MONEY"))