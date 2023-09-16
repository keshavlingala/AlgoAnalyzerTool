# Performing unit testing for sorting algorithms

import random
import unittest

from main import BubbleSort, InsertionSort, SelectionSort, MergeSort, QuickSort, HeapSort


class TestSortingAlgorithms(unittest.TestCase):

    def setUp(self):
        self.test_list = [random.randint(0, 100) for i in range(10)]
        self.sorted_list = sorted(self.test_list)
        self.reverse_list = sorted(self.test_list, reverse=True)
        self.empty_list = []
        self.one_element_list = [1]
        self.two_element_list = [1, 2]

    def test_bubble_sort(self):
        self.assertEqual(BubbleSort().sort(self.test_list)[0], self.sorted_list)
        self.assertEqual(BubbleSort().sort(self.reverse_list)[0], self.sorted_list)
        self.assertEqual(BubbleSort().sort(self.empty_list)[0], self.empty_list)
        self.assertEqual(BubbleSort().sort(self.one_element_list)[0], self.one_element_list)
        self.assertEqual(BubbleSort().sort(self.two_element_list)[0], self.two_element_list)

    def test_insertion_sort(self):
        self.assertEqual(InsertionSort().sort(self.test_list)[0], self.sorted_list)
        self.assertEqual(InsertionSort().sort(self.reverse_list)[0], self.sorted_list)
        self.assertEqual(InsertionSort().sort(self.empty_list)[0], self.empty_list)
        self.assertEqual(InsertionSort().sort(self.one_element_list)[0], self.one_element_list)
        self.assertEqual(InsertionSort().sort(self.two_element_list)[0], self.two_element_list)

    def test_selection_sort(self):
        self.assertEqual(SelectionSort().sort(self.test_list)[0], self.sorted_list)
        self.assertEqual(SelectionSort().sort(self.reverse_list)[0], self.sorted_list)
        self.assertEqual(SelectionSort().sort(self.empty_list)[0], self.empty_list)
        self.assertEqual(SelectionSort().sort(self.one_element_list)[0], self.one_element_list)
        self.assertEqual(SelectionSort().sort(self.two_element_list)[0], self.two_element_list)

    def test_merge_sort(self):
        self.assertEqual(MergeSort().sort(self.test_list)[0], self.sorted_list)
        self.assertEqual(MergeSort().sort(self.reverse_list)[0], self.sorted_list)
        self.assertEqual(MergeSort().sort(self.empty_list)[0], self.empty_list)
        self.assertEqual(MergeSort().sort(self.one_element_list)[0], self.one_element_list)
        self.assertEqual(MergeSort().sort(self.two_element_list)[0], self.two_element_list)

    def test_quick_sort(self):
        self.assertEqual(QuickSort().sort(self.test_list)[0], self.sorted_list)
        self.assertEqual(QuickSort().sort(self.reverse_list)[0], self.sorted_list)
        self.assertEqual(QuickSort().sort(self.empty_list)[0], self.empty_list)
        self.assertEqual(QuickSort().sort(self.one_element_list)[0], self.one_element_list)
        self.assertEqual(QuickSort().sort(self.two_element_list)[0], self.two_element_list)

    def test_heap_sort(self):
        self.assertEqual(HeapSort().sort(self.test_list)[0], self.sorted_list)
        self.assertEqual(HeapSort().sort(self.reverse_list)[0], self.sorted_list)
        self.assertEqual(HeapSort().sort(self.empty_list)[0], self.empty_list)
        self.assertEqual(HeapSort().sort(self.one_element_list)[0], self.one_element_list)
        self.assertEqual(HeapSort().sort(self.two_element_list)[0], self.two_element_list)


if __name__ == '__main__':
    unittest.main()
