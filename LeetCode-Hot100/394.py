"""
LeetCode 394. 字符串解码（Decode String）

给定一个经过编码的字符串，返回解码后的字符串。
编码规则为：k[encoded_string]，表示其中方括号内的字符串需要重复 k 次。

示例：
输入：s = "3[a]2[bc]"
输出："aaabcbc"

输入：s = "3[a2[c]]"
输出："accaccacc"

输入：s = "2[abc]3[cd]ef"
输出："abcabccdcdcdef"

规则说明：
- 数字 k 可以是多位数，例如 12[a]
- 可以多层嵌套，例如 3[a2[c]]
- 目标是把字符串完全展开还原
"""

class Solution:
    def decodeString(self, s: str) -> str:
        # 数字栈：用来存每一次遇到 '[' 之前的重复次数
        numStack = []
        # 字符串栈：用来存每一次遇到 '[' 之前已经构造好的字符串
        strStack = []

        # 当前正在解析的数字（可能是多位数）
        currentNum = 0
        # 当前正在构造的字符串
        currentString = ""

        # 遍历输入字符串的每一个字符
        for ch in s :
            # 如果是数字：构建重复次数（可能是多位数）
            if ch.isdigit():
                # 举例：如果看到 '2' '3' 应该变成 23
                currentNum = currentNum * 10 + int(ch)

            # 如果遇到 '[' ：把当前状态入栈
            elif ch == "[":
                # 将当前的数字（重复次数）压入数字栈
                numStack.append(currentNum)
                # 将当前已构造的字符串压入字符串栈
                strStack.append(currentString)

                # 重置当前数字与当前字符串
                # 因为 '[' 后面会出现一个新的子结构需要重新构建
                currentString = ""
                currentNum = 0

            #  如果遇到 ']' ：处理一个完整的编码段
            elif ch == "]":
                # 从数字栈弹出这段字符串应该重复的次数
                repeat_times = numStack.pop()
                # 从字符串栈中取出这一段编码之前的字符串
                prev_str = strStack.pop()

                # 如 prev_str = "a", currentStr = "bc", repeat_times = 3
                # 则 newStr = "a" + "bc" * 3 = "abcbcbc"
                currentString = prev_str + currentString * repeat_times

            else:
                # 普通字符，直接加入当前字符串
                currentString = currentString + ch

        return currentString