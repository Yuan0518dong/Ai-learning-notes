# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
from typing import Optional



# 1. 定义链表节点 (LeetCode 后台隐藏的代码，本地必须写)
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # 双指针法 (迭代)
        prev = None
        curr = head
        while curr is not None:
            temp = curr.next

            # 反转指针
            curr.next = prev
            prev = curr

            curr = temp

        return prev


def create_linked_list(arr):
    """把 Python 列表 [1,2,3] 变成 链表 1->2->3"""
    if not arr:
        return None
    head = ListNode(arr[0])
    curr = head
    for val in arr[1:]:
        curr.next = ListNode(val)
        curr = curr.next
    return head

def print_linked_list(head):
    """把链表打印出来看"""
    vals = []
    curr = head
    while curr:
        vals.append(str(curr.val))
        curr = curr.next
    print(" -> ".join(vals))

if __name__ == "__main__":
    # 准备数据
    input_list = [1, 2, 3, 4, 5]
    head = create_linked_list(input_list)

    print("原始链表:")
    print_linked_list(head)

    # 运行算法
    solution = Solution()
    new_head = solution.reverseList(head)

    print("\n反转后链表:")
    print_linked_list(new_head)

    # 验证是否反转成功
    # 应该输出: 5 -> 4 -> 3 -> 2 -> 1


