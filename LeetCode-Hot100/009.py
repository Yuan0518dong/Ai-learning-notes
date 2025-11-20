
"""
回文数

给你一个整数 x ，如果 x 是一个回文整数，返回 true ；否则，返回 false 。
回文数是指正序（从左向右）和倒序（从右向左）读都是一样的整数。
例如，121 是回文，而 123 不是
"""

"""
class Solution:
    def isPalindrome(self, x: int) -> bool:
        if x < 0 or (x != 0 and x % 10 == 0):
            return False
        else:
            res = 0
            temp = x
            while x > 0:
                a =x % 10
                res = a  + res * 10
                x = x // 10 #单斜杠 / 执行的是浮点除法（真除法），而双斜杠 // 才是整除
            return res == temp
"""
#最佳解决方案：翻转数字的后半部分
class Solution:
    def isPalindrome(self, x: int) -> bool:
        if x < 0 or (x != 0 and x % 10 == 0):
            return False
        else:
            res = 0
        #核心逻辑：当原始数字 x 大于 翻转数字 reversed_half 时，说明还没翻转到一半
            while x > res :
                res = res * 10 + x % 10
                x = x // 10
            return res == x or res//10 == x
# 避免溢出（针对 C++/Java）：在 Python 中整数无限大，不会溢出。但在 C++ 或 Java 中，如果输入是 2147447412，反转后会超过 int 的最大范围导致报错。
# 只翻转一半，数字绝对不会超过原数字，彻底杜绝了溢出风险。效率更高：只处理一半的数字，循环次数减半，虽然时间复杂度在量级上还是 $O(\log_{10} n)$，但实际操作指令减少了一半，执行速度更快.
# 逻辑优雅：不需要额外的变量来存储原始值（比如你之前代码里的 temp），空间复杂度是纯粹的 $O(1)$。

if __name__ == "__main__":
    # 1. 实例化解题对象
    solver = Solution()
    # 2. 准备测试数据 (模拟 C 语言的 main 函数输入)
    test_nums = 0
    # 3. 调用函数并打印结果
    result = solver.isPalindrome(test_nums)
    print(f"输入: nums={test_nums}")
    print(f"输出: {result}")