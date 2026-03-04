"""
你这个学期必须选修 numCourses 门课程，记为 0 到 numCourses - 1 。

在选修某些课程之前需要一些先修课程。 先修课程按数组 prerequisites 给出，其中 prerequisites[i] = [ai, bi] ，表示如果要学习课程 ai 则 必须 先学习课程  bi 。

例如，先修课程对 [0, 1] 表示：想要学习课程 0 ，你需要先完成课程 1 。
请你判断是否可能完成所有课程的学习？如果可以，返回 true ；否则，返回 false 。

示例 1：
输入：numCourses = 2, prerequisites = [[1,0]]
输出：true
解释：总共有 2 门课程。学习课程 1 之前，你需要完成课程 0 。这是可能的。

示例 2：
输入：numCourses = 2, prerequisites = [[1,0],[0,1]]
输出：false
解释：总共有 2 门课程。学习课程 1 之前，你需要先完成课程 0 ；并且学习课程 0 之前，你还应先完成课程 1 。这是不可能的。
"""
# deque: 双端队列
from collections import deque
from typing import List

from scipy.cluster.hierarchy import complete

# 拓扑排序 Kahn 算法（入度表 + 队列）
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        # 构建邻接表
        graph = [[] for _ in range(numCourses)] # 邻接表，索引为课程，值为后续课程列表
        in_degree = [0] * numCourses #构造入度数组，记录每个课程的先修课数量

        # 2. 构建邻接表和入度数组
        for course, pre_course in prerequisites:
            graph[pre_course].append(course)    # pre_course -> course（学course先学pre_course）
            in_degree[course] += 1

        # 3. 初始化队列：入度为0的课程（无先修课）
        queue = deque()
        for i in range(numCourses):
            if in_degree[i] == 0:
                queue.append(i)         #将入度为零的课程入队

        # 4. 拓扑排序，统计能完成的课程数
        completed = 0
        while queue:
            current  = queue.popleft()  #取出当前可完成的课程
            completed += 1
            # 遍历当前课程的所有后续课程
            for next in graph[current]:
                in_degree[next] -= 1  # 后续课程的先修课少了一门
                # 如果后续课程的入度为0，则可以学习 即入队
                if in_degree[next] == 0:
                    queue.append(next)

        #5.判断是否所有课程都能完成
        return completed == numCourses






