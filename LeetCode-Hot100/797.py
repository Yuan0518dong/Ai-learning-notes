"""
给你一个有 n 个节点的 有向无环图（DAG），请你找出从节点 0 到节点 n-1 的所有路径并输出（不要求按特定顺序）

 graph[i] 是一个从节点 i 可以访问的所有节点的列表（即从节点 i 到节点 graph[i][j]存在一条有向边）。



示例 1：

输入：graph = [[1,2],[3],[3],[]]
输出：[[0,1,3],[0,2,3]]
解释：有两条路径 0 -> 1 -> 3 和 0 -> 2 -> 3

示例 2：

输入：graph = [[4,3,1],[3,2,4],[3],[4],[]]
输出：[[0,4],[0,3,4],[0,1,3,4],[0,1,2,3,4],[0,1,4]]
"""
from typing import List


class Solution:
    def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
        #结果数组
        result = []
        #终点节点编号
        target = len(graph)-1

        #定义DFS函数，current是当前节点，path是当前路径
        def dfs(current,path):
            #将当前节点加入路径
            path.append(current)

            # 终止条件：到达终点，记录路径
            if current == target:
                #必须传path的副本（path.copy()），否则后续修改会改变已存入的路径
                result.append(path.copy())
                #回溯：移除当前节点，继续探索其他分支
                path.pop()
                return


            #遍历当前节点的所有邻接节点，递归搜索
            for next in graph[current]:
                dfs(next, path)

            # 回溯：当前节点的所有分支都遍历完了，移除当前节点
            path.pop()

        dfs(0,[])
        return result








