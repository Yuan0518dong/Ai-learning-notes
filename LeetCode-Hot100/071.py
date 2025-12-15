"""
给你一个字符串 path ，表示指向某一文件或目录的 Unix 风格 绝对路径 （以 '/' 开头），请你将其转化为 更加简洁的规范路径。

在 Unix 风格的文件系统中规则如下：

一个点 '.' 表示当前目录本身。
此外，两个点 '..' 表示将目录切换到上一级（指向父目录）。
任意多个连续的斜杠（即，'//' 或 '///'）都被视为单个斜杠 '/'。
任何其他格式的点（例如，'...' 或 '....'）均被视为有效的文件/目录名称。
返回的 简化路径 必须遵循下述格式：

始终以斜杠 '/' 开头。
两个目录名之间必须只有一个斜杠 '/' 。
最后一个目录名（如果存在）不能 以 '/' 结尾。
此外，路径仅包含从根目录到目标文件或目录的路径上的目录（即，不含 '.' 或 '..'）。
返回简化后得到的 规范路径 。
"""

class Solution:
    def simplifyPath(self, path: str) -> str:
        parts = path.split('/')
        stack = []

        for part in parts:
            if part == '..':
                if stack:
                    stack.pop()
            elif part == '' or part == '.':
                continue
            else:
                stack.append(part)

        return '/'+'/'.join(stack)

# 测试
if __name__ == '__main__':
    # LeetCode 官方公开测试用例（来自题目描述和常见测试集）
    leetcode_tests = [
        ("/home/", "/home"),
        ("/../", "/"),
        ("/home//foo/", "/home/foo"),
        ("/a/./b/../../c/", "/c"),
        ("/", "/"),
        ("/.//", "/"),
        ("/a/../../b/../c//.//", "/c"),
        ("/.../", "/..."),      # 注意：... 是普通目录名
        ("/..//", "/"),
        ("/./", "/"),
        ("/a//b////c/d//././/..", "/a/b/c"),
    ]

    # 运行测试
    print("Running LeetCode-style test cases:\n")
    # enumerate() 是 Python 内置函数，用于在遍历时同时获取索引和元素
    # 默认从 0 开始计数，但这里传了第二个参数 1，表示从 1 开始计数
    sol = Solution()
    for i, (inp, expected) in enumerate(leetcode_tests, 1):
        output = sol.simplifyPath(inp)
        if output == expected:
            print(f"✅ Test {i}: PASS")
        else:
            print(f"❌ Test {i}: FAIL")
            print(f"   Input:    {repr(inp)}")
            print(f"   Output:   {repr(output)}")
            print(f"   Expected: {repr(expected)}")