"""
给你一个整数数组 nums ，判断是否存在三元组 [nums[i], nums[j], nums[k]] 满足 i != j、i != k 且 j != k ，同时还满足 nums[i] + nums[j] + nums[k] == 0 。请你返回所有和为 0 且不重复的三元组。

注意：答案中不可以包含重复的三元组。





示例 1：

输入：nums = [-1,0,1,2,-1,-4]
输出：[[-1,-1,2],[-1,0,1]]
解释：
nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0 。
nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0 。
nums[0] + nums[3] + nums[4] = (-1) + 2 + (-1) = 0 。
不同的三元组是 [-1,0,1] 和 [-1,-1,2] 。
注意，输出的顺序和三元组的顺序并不重要。
示例 2：

输入：nums = [0,1,1]
输出：[]
解释：唯一可能的三元组和不为 0 。
示例 3：

输入：nums = [0,0,0]
输出：[[0,0,0]]
解释：唯一可能的三元组和为 0 。
"""
class Solution:
    def threeSum(self, nums: list[int]) -> list[list[int]]:
        # 初始化结果列表
        result = []
        #排序数组
        nums.sort()
        n = len(nums)

        for i in range(n):
            # 剪枝1：排序后第一个数>0，三数之和不可能为0（直接终止）
            if nums[i] > 0:
                break
            # 去重1：跳过重复的第一个数（避免重复三元组）
            if i > 0 and nums[i] == nums[i-1]:
                continue
            #双指针初始化：left在i右侧起点，right在数组末尾
            left = i + 1
            right = n - 1
            #目标：找 nums[left] + nums[right] = -nums[i]
            target = -nums[i]

            #双指针缩范围
            while left < right:
                # 情况1：和等于目标值 → 找到有效三元组
                if nums[left] + nums[right] == target:
                    result.append([nums[i],nums[left], nums[right]])
                    # 去重2：跳过left侧重复元素
                    while left < right and nums[left]== nums[left+1]:
                        left += 1
                    # 去重3：跳过right侧重复元素
                    while left < right and nums[right]== nums[right-1]:
                        right -= 1

                    # 指针同时移动（找下一组可能的数）
                    left += 1
                    right -= 1

                # 情况2：和小于目标值 → 左指针右移（增大和）
                elif nums[left] + nums[right] < target:
                    left += 1
                # 情况3：和大于目标值 → 右指针左移（减小和）
                elif nums[left] + nums[right] > target:
                    right -= 1

        return result





















