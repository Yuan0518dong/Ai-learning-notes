"""
请你设计并实现一个满足  LRU (最近最少使用) 缓存 约束的数据结构。
实现 LRUCache 类：
LRUCache(int capacity) 以 正整数 作为容量 capacity 初始化 LRU 缓存
int get(int key) 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1 。
void put(int key, int value) 如果关键字 key 已经存在，则变更其数据值 value ；如果不存在，则向缓存中插入该组 key-value 。如果插入操作导致关键字数量超过 capacity ，则应该 逐出 最久未使用的关键字。
函数 get 和 put 必须以 O(1) 的平均时间复杂度运行。

示例：
输入
["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
输出
[null, null, null, 1, null, -1, null, -1, 3, 4]

解释
LRUCache lRUCache = new LRUCache(2);
lRUCache.put(1, 1); // 缓存是 {1=1}
lRUCache.put(2, 2); // 缓存是 {1=1, 2=2}
lRUCache.get(1);    // 返回 1
lRUCache.put(3, 3); // 该操作会使得关键字 2 作废，缓存是 {1=1, 3=3}
lRUCache.get(2);    // 返回 -1 (未找到)
lRUCache.put(4, 4); // 该操作会使得关键字 1 作废，缓存是 {4=4, 3=3}
lRUCache.get(1);    // 返回 -1 (未找到)
lRUCache.get(3);    // 返回 3
lRUCache.get(4);    // 返回 4
"""

class Node:
    """定义双向链表节点"""
    def __init__(self, key=0,value=0):
        self.key = key
        self.value = value
        self.next = None
        self.prev = None

"""
    哈希表 + 双向链表:
 LRU 缓存要求 get 和 put 操作的时间复杂度是 O (1)，而双向链表虽然能高效完成 “移动节点、删除节点”，但无法快速查找某个 key 对应的节点（遍历链表是 O (n)）
"""
class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = dict() #dict() 是 Python 中创建空字典的内置方法，字典（Dictionary）是一种键值对（key-value） 结构的无序集合 核心特性是：通过 key 查找 value 的时间复杂度是 O(1)（哈希表实现）；每个 key 唯一，不会重复。
        self.size = 0
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1

        node = self.cache[key]
        self.moveToHead(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        if key not in self.cache:
            node = Node(key, value)
            #添加到哈希表
            self.cache[key] = node
            #添加到双链表头部
            self.addToHead(node)
            self.size += 1
            if self.size > self.capacity:
                remove = self.removeTail()
                #删除哈希表中对应的项
                self.cache.pop(remove.key)
                self.size -= 1

        elif key in self.cache:
            #如果 key 存在，先通过哈希表定位，再修改 value，并移到头部
            node = self.cache[key]
            node.value = value
            self.moveToHead(node)

    def addToHead(self, node):
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def removeNode(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def moveToHead(self, node):
        self.removeNode(node)
        self.addToHead(node)

    def removeTail(self):
        node = self.tail.prev
        self.removeNode(node)
        return node















