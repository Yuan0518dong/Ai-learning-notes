"""
给你二叉树的根节点 root ，返回其节点值的 层序遍历 （即逐层地，从左到右访问所有节点）。
示例 1：

输入：root = [3,9,20,null,null,15,7]
输出：[[3],[9,20],[15,7]]
示例 2：

输入：root = [1]
输出：[[1]]
示例 3：

输入：root = []
输出：[]
"""

from typing import List, Optional
from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []

        result = []
        queue = deque([root])#初始化队列

        while queue:
            level_size = len(queue) #记录当前队列中的节点个数，在这一层循环时作为边界
            current_level = []

            for _ in range(level_size):
                node = queue.popleft()  #从队列左边取出一个节点
                current_level.append(node.val)

                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

            result.append(current_level)#遍历完该层后，将该层的节点值存入结果数组中，其中current_level也是一个数组

        return result


# 辅助函数：根据列表创建二叉树
def build_tree(values: List[Optional[int]]) -> Optional[TreeNode]:
    if not values:
        return None

    root = TreeNode(values[0])
    queue = deque([root])
    i = 1

    while queue and i < len(values):
        node = queue.popleft()

        # 左子节点
        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i])
            queue.append(node.left)
        i += 1

        # 右子节点
        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])
            queue.append(node.right)
        i += 1

    return root


def main():
    solution = Solution()

    # 测试用例
    test_cases = [
        ([3, 9, 20, None, None, 15, 7], [[3], [9, 20], [15, 7]]),
        ([1], [[1]]),
        ([], []),
        ([1, 2, 3, 4, None, None, 5], [[1], [2, 3], [4, 5]]),
    ]

    for i, (input_list, expected) in enumerate(test_cases, 1):
        print(f"Test case {i}:")
        print(f"Input: {input_list}")

        root = build_tree(input_list)
        result = solution.levelOrder(root)

        print(f"Output: {result}")
        print(f"Expected: {expected}")
        print(f"Pass: {result == expected}")
        print("-" * 40)


if __name__ == "__main__":
    main()
