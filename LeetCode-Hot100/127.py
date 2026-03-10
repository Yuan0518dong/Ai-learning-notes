"""
字典 wordList 中从单词 beginWord 到 endWord 的 转换序列 是一个按下述规格形成的序列 beginWord -> s1 -> s2 -> ... -> sk：

每一对相邻的单词只差一个字母。
 对于 1 <= i <= k 时，每个 si 都在 wordList 中。注意， beginWord 不需要在 wordList 中。
sk == endWord
给你两个单词 beginWord 和 endWord 和一个字典 wordList ，返回 从 beginWord 到 endWord 的 最短转换序列 中的 单词数目 。如果不存在这样的转换序列，返回 0 。


示例 1：

输入：beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
输出：5
解释：一个最短转换序列是 "hit" -> "hot" -> "dot" -> "dog" -> "cog", 返回它的长度 5。
示例 2：

输入：beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log"]
输出：0
解释：endWord "cog" 不在字典中，所以无法进行转换。

"""
from typing import List
from collections import defaultdict, deque

"""
核心逻辑：把单词转换看成 “图的最短路径”
每个单词是图中的一个节点；
两个单词如果能通过 “改变一个字符” 互相转换，就是图中的一条边；
问题转化为：求从 beginWord 到 endWord 的最短路径长度（BFS 是求无权图最短路径的最优算法）
"""


class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        # 1. 边界判断：endWord不在wordList中，直接返回0
        if endWord not in wordList:
            return 0

        l = len(beginWord)
        #为什么用 defaultdict (list)：普通字典如果访问不存在的 key 会报错，而 defaultdict(list) 会自动为不存在的 key 创建一个空列表，方便后续 append 操作
        all_com_dict = defaultdict(list) #初始化默认字典：key=通用状态，value=该状态对应的单词列表
        #遍历单词列表中的每个单词
        for word in wordList:
            #遍历单词的每个字符位置（0到L-1）
            for i in range(l):
                #word[:i]：取单词中第 i 位之前的所有字符;word[i+1:]：取单词中第 i 位之后的所有字符
                genetic_state = word[:i] + "*" + word[i+1:]
                all_com_dict[genetic_state].append(word)


        queue = deque()
        queue.append((beginWord,1))#把「单词」和「路径长度」两个值作为一个整体存入队列
        #初始化访问集合：避免重复访问单词（防止环）
        vist = set()
        vist.add(beginWord)


        while queue:
            current , level = queue.popleft()
            for i in range(l):
                #变例该单词所有通用状态
                genetic_state = current[:i] + "*" + current[i+1:]

                for candidate in all_com_dict[genetic_state]:
                    #找到endWord，返回当前层级+1（因为candidate是下一个（层）单词）
                    if candidate == endWord:
                        return level + 1
                    #如果候选词未被访问，则加入访问集合，并将该单词加入队列
                    if candidate not in vist:
                        vist.add(candidate)
                        queue.append((candidate,level+1))

                # 优化：清空该通用状态的候选单词（避免重复处理）
                all_com_dict[genetic_state]=[]

        #遍历完所有可能仍未找到endWord，返回0
        return 0














