
from typing import List

inputs = [
    [0, 0, 1, 1, 1, 2, 2, 3, 3, 4],
    [1, 1, 2]
]

outputs = [
    5, 2
]


def remove_dups(nums: List) -> int:
    len_ = 1
    for i in range(1, len(nums)):
        if nums[i] != nums[i-1]:
            nums[len_] = nums[i]
            len_ += 1
    return len_, nums


for input_ in inputs:
    print(remove_dups(nums=input_))


class Solution:
    def remove_duplicates(self, nums: List[int]) -> int:
        len_ = 1
        if len(nums) == 0:
            return 0
        for i in range(1, len(nums)):
            if nums[i] != nums[i-1]:
                nums[len_] = nums[i]
                len_ += 1
        return len_


sol = Solution()
for input_ in inputs:
    pass
    # print(sol.remove_duplicates(nums=input_))


