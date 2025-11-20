from typing import List  # 必须导入这个，否则 List[int] 会报错
"""
两数之和

给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 target  的那 两个 整数，并返回它们的数组下标。
你可以假设每种输入只会对应一个答案，并且你不能使用两次相同的元素。
你可以按任意顺序返回答案

"""
# --- 这里是 LeetCode 网页让你写的代码 ---
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        # 你的逻辑 (Python 不需要像 C 那样手动管理内存)
            hash_map = {}
            for i, x in enumerate(nums):#x=num[i]
                for j in range(i + 1, len(nums)):
                    if x + nums[j] == target:
                        return [i, j]
            return []
# ---------------------------------------

# --- 下面是你在本地 VS Code 调试用的代码 (网页上不需要提交) ---
if __name__ == "__main__":
    # 1. 实例化解题对象
    solver = Solution()
    # 2. 准备测试数据 (模拟 C 语言的 main 函数输入)
    test_nums = [2, 7, 11, 15]
    test_target = 9
    
    # 3. 调用函数并打印结果
    result = solver.twoSum(test_nums, test_target)
    print(f"输入: nums={test_nums}, target={test_target}")
    print(f"输出: {result}")