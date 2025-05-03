from typing import List

# Odd even merge sort based on https://hwlang.de/algorithmen/sortieren/networks/oemen.htm
# Verified on https://leetcode.com/problems/sort-an-array/

class Solution:

    def compare(self,i, j, arr):
        if j < len(arr):
            if arr[i] > arr[j]:
                arr[i], arr[j] = arr[j], arr[i]


    def operate(self,tid, A, S, arr, num_blocks, comparisons_per_block):
        block_id = tid // comparisons_per_block
        comparison_id = tid % comparisons_per_block
        block_start = block_id * 2 * A
        if A == S:
            self.compare(block_start + comparison_id, block_start + comparison_id + A, arr)
        else:
            start = block_start + S
            comparison_chunk_width_id = comparison_id // S
            start += 2 * S * comparison_chunk_width_id
            comparison_id_in_chunk = comparison_id % S
            self.compare(start + comparison_id_in_chunk, start + comparison_id_in_chunk + S, arr)


    def odd_even_merge_sort(self,arr):
        Npower2 = 1
        while Npower2 < len(arr):
            Npower2 *= 2
        A = 1  # already sorted width
        while A < Npower2:
            S = A
            while S >= 1:
                num_blocks = Npower2 // (2 * A)
                comparisons_per_block = A if A == S else A - S
                # this section parallelizes on a GPU well, can launch each thread independently 
                for tid in range(num_blocks * comparisons_per_block):
                    self.operate(tid, A, S, arr, num_blocks, comparisons_per_block)
                S //= 2
            A *= 2

    def sortArray(self, nums: List[int]) -> List[int]:
        self.odd_even_merge_sort(nums)
        return nums

if __name__ == "__main__":
    sol = Solution()
    print(sol.sortArray([5,2,3,1,6,1,4,3,2,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]))
