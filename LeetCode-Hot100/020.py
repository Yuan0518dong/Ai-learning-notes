"""
    判断括号字符串是否有效
"""
class Solution:
    def isValid(self, s: str) -> bool:
        """
            当你发现“某种输入总对应某种输出”时，就该考虑用 mapping（字典）了！
        """
        if len(s) % 2 != 0:
            return False
        stack = []
        # 创建映射：右括号 -> 对应的左括号
        mapping = {')':'(',']':'[','}':'{'}#{key: value},实现了“键（key）到值（value）”的映射
        for c in s:
            if c not in mapping:
                stack.append(c)
            elif c in mapping:
                if not stack or mapping[c] != stack.pop():
                    return False
        return not stack


# ======================
# 测试部分
# ======================
if __name__ == "__main__":
    # 定义测试用例：(输入, 期望输出)
    test_cases = [
        ("()", True),
        ("()[]{}", True),
        ("([{}])", True),
        ("(]", False),
        ("([)]", False),
        ("{[()]}", True),
        ("(((", False),
        (")))", False),
        ("", True),  # 空字符串有效
        ("{[}]", False),
        ("{[]}", True),
    ]

    print("开始测试 isValid 函数：\n")
    # 先创建 Solution 的实例
    # 用这个实例去调用 isValid
    res = Solution()
    for i, (input_str, expected) in enumerate(test_cases, 1):
        result = res.isValid(input_str)
        status = "✅ 通过" if result == expected else "❌ 失败"
        print(f"测试 {i}: 输入 = '{input_str}' | 期望 = {expected} | 实际 = {result} | {status}")

    print("\n测试完成！")
