"""
给你一个整数数组 coins ，表示不同面额的硬币；以及一个整数 amount ，表示总金额。

计算并返回可以凑成总金额所需的 最少的硬币个数 。如果没有任何一种硬币组合能组成总金额，返回 -1 。

你可以认为每种硬币的数量是无限的。



示例 1：

输入：coins = [1, 2, 5], amount = 11
输出：3
解释：11 = 5 + 5 + 1
示例 2：

输入：coins = [2], amount = 3
输出：-1
示例 3：

输入：coins = [1], amount = 0
输出：0
"""
from typing import List


class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        # 定义dp数组：dp[i]表示凑出金额i的最少硬币数
        dp = [amount+1]*(amount+1)#[amount + 1] * 长度 表示数组中每一个位置的初始值都是 amount + 1
        dp[0] = 0
        #遍历所有金额（从1到amount）
        for i in range(1, amount+1):
            #遍历所有硬币，尝试用当前硬币凑金额i
            for coin in coins:
                if i - coin >= 0:
                    # 状态转移：dp[i] = min(当前dp[i], dp[i-coin]+1)
                    dp[i] = min(dp[i], dp[i-coin] + 1)
        # 结果判断：如果dp[amount]还是amount+1，说明无法凑出，返回-1
        return dp[amount] if dp[amount] <= amount else -1













