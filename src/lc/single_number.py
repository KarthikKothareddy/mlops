

from typing import List

inputs = [
    [2, 2, 1],
    [4, 1, 2, 1, 2],
    [1]
]

outputs = [
    1, 4, 1
]


class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        for i in range(1, len(nums)):
            nums[0] ^= nums[i]
        return nums[0]


sol = Solution()
for input_ in inputs:
    pass
    # print(sol.singleNumber(nums=input_))

print(1 ^ 1)
