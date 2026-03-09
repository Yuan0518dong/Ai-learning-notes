"""
给你一个链表数组，每个链表都已经按升序排列。

请你将所有链表合并到一个升序链表中，返回合并后的链表。

示例 1：

输入：lists = [[1,4,5],[1,3,4],[2,6]]
输出：[1,1,2,3,4,4,5,6]
解释：链表数组如下：
[
  1->4->5,
  1->3->4,
  2->6
]
将它们合并到一个有序链表中得到。
1->1->2->3->4->4->5->6
示例 2：

输入：lists = []
输出：[]
示例 3：

输入：lists = [[]]
输出：[]
"""
import heapq
from typing import List, Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        # 1. 初始化最小堆
        heap = []
        # 唯一索引，避免节点值相同时报错
        idx = 0

        # 2. 把每个非空链表的头节点加入堆
        for l in lists:
            if l:
                heapq.heappush(heap, (l.val, idx, l))
                idx += 1
        #虚拟头结点（简化链表拼接）
        dummy = ListNode()
        #结果链表中的当前结点，用于存储新节点
        current = dummy

        while heap:
            # 弹出堆顶的最小节点（val最小）
            val, idx, node = heapq.heappop(heap)
            #更新结果链表
            current.next = node
            current = current.next

            #如果弹出的节点后还有节点，则让其进入堆
            if node.next:
                heapq.heappush(heap,(node.next.val,idx,node.next))
                idx += 1


        return dummy.next
