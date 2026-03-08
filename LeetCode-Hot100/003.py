"""
给定一个字符串 s ，请你找出其中不含有重复字符的 最长 子串 的长度。

示例 1:

输入: s = "abcabcbb"
输出: 3
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。注意 "bca" 和 "cab" 也是正确答案。
示例 2:

输入: s = "bbbbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。
示例 3:

输入: s = "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
"""


class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        #设置哈希表，key=字符，value=该字符最后一次出现的索引
        char_index = dict()
        left = 0 #左指针
        max_len = 0 #最大长度

        for right in range(len(s)):
            current = s[right]
            #核心判断：当前字符是否在「当前窗口内」重复
            # 条件1：current_char 在哈希表中（出现过）
            # 条件2：该字符最后一次出现的索引 >= left（说明在当前窗口内）
            if current  in char_index and char_index[current] >= left:
                left = char_index[current] + 1
            #更新哈希表 ：记录当前字符的最新索引（覆盖旧值）
            char_index[current] = right
            #计算当前窗口长度，更新最大值
            max_len = max(max_len, right - left + 1)

        return max_len


