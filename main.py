import sys
import time

from PyQt5.QtWidgets import QApplication

from gui import SortingApp


#  utils functions
def time_it(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        logger(f"Time taken by {func.__name__}: {time.time() - start}")
        return result

    return wrapper


# static Functions
def swap(i, j, array):
    array[i], array[j] = array[j], array[i]


def merge(array, l, m, r):
    n1 = m - l + 1
    n2 = r - m

    L = [0] * n1
    R = [0] * n2

    for i in range(n1):
        L[i] = array[l + i]
    for j in range(n2):
        R[j] = array[m + 1 + j]

    i = j = 0
    k = l

    while i < n1 and j < n2:
        if L[i] <= R[j]:
            array[k] = L[i]
            i += 1
        else:
            array[k] = R[j]
            j += 1
        k += 1

    while i < n1:
        array[k] = L[i]
        i += 1
        k += 1

    while j < n2:
        array[k] = R[j]
        j += 1
        k += 1


def partition(array, low, high):
    i = low - 1
    pivot = array[high]

    for j in range(low, high):
        if array[j] < pivot:
            i += 1
            swap(i, j, array)
    swap(i + 1, high, array)
    return i + 1


# Algorithms

class SortingAlgorithm:
    def sort(self, array):
        pass

    def has_limitation(self):
        return False


class BubbleSort(SortingAlgorithm):
    @time_it
    def sort(self, array):
        for i in range(len(array)):
            for j in range(len(array) - 1):
                if array[j] > array[j + 1]:
                    swap(j, j + 1, array)
        return array


class SelectionSort(SortingAlgorithm):
    @time_it
    def sort(self, array):
        for i in range(len(array)):
            min_index = i
            for j in range(i + 1, len(array)):
                if array[j] < array[min_index]:
                    min_index = j
            swap(i, min_index, array)
        return array


class InsertionSort(SortingAlgorithm):
    @time_it
    def sort(self, array):
        for i in range(1, len(array)):
            key = array[i]
            j = i - 1
            while j >= 0 and key < array[j]:
                array[j + 1] = array[j]
                j -= 1
            array[j + 1] = key
        return array


class MergeSort(SortingAlgorithm):

    @time_it
    def sort(self, array):
        self.merge_sort(array, 0, len(array) - 1)
        return array

    def merge_sort(self, array, l, r):
        if l < r:
            m = (l + r) // 2
            self.merge_sort(array, l, m)
            self.merge_sort(array, m + 1, r)
            merge(array, l, m, r)


class QuickSort(SortingAlgorithm):
    @time_it
    def sort(self, array):
        self.quick_sort(array, 0, len(array) - 1)
        return array

    def quick_sort(self, array, low, high):
        if low < high:
            pi = partition(array, low, high)
            self.quick_sort(array, low, pi - 1)
            self.quick_sort(array, pi + 1, high)


class HeapSort(SortingAlgorithm):
    @time_it
    def sort(self, array):
        n = len(array)
        for i in range(n // 2 - 1, -1, -1):
            self.heapify(array, n, i)
        for i in range(n - 1, 0, -1):
            swap(i, 0, array)
            self.heapify(array, i, 0)
        return array

    def heapify(self, array, n, i):
        largest = i
        l = 2 * i + 1
        r = 2 * i + 2

        if l < n and array[largest] < array[l]:
            largest = l
        if r < n and array[largest] < array[r]:
            largest = r
        if largest != i:
            swap(i, largest, array)
            self.heapify(array, n, largest)


class CountingSort(SortingAlgorithm):
    @time_it
    def sort(self, array):
        max_element = max(array)
        min_element = min(array)
        range_of_elements = max_element - min_element + 1
        count_arr = [0 for _ in range(range_of_elements)]
        output_arr = [0 for _ in range(len(array))]

        for i in range(0, len(array)):
            count_arr[array[i] - min_element] += 1

        for i in range(1, len(count_arr)):
            count_arr[i] += count_arr[i - 1]

        for i in range(len(array) - 1, -1, -1):
            output_arr[count_arr[array[i] - min_element] - 1] = array[i]
            count_arr[array[i] - min_element] -= 1

        for i in range(0, len(array)):
            array[i] = output_arr[i]
        return array


class RadixSort(SortingAlgorithm):
    def sort(self, array):
        max_element = max(array)
        exp = 1
        while max_element // exp > 0:
            self.count_sort(array, exp)
            exp *= 10
        return array

    def count_sort(self, array, exp):
        n = len(array)
        output = [0] * n
        count = [0] * 10

        for i in range(0, n):
            index = array[i] // exp
            count[index % 10] += 1

        for i in range(1, 10):
            count[i] += count[i - 1]

        i = n - 1
        while i >= 0:
            index = array[i] // exp
            output[count[index % 10] - 1] = array[i]
            count[index % 10] -= 1
            i -= 1

        i = 0
        for i in range(0, len(array)):
            array[i] = output[i]


class BucketSort(SortingAlgorithm):
    def sort(self, array):
        max_element = max(array)
        min_element = min(array)
        range_of_elements = max_element - min_element + 1
        bucket_arr = [[] for _ in range(range_of_elements)]
        output_arr = []

        for i in range(0, len(array)):
            bucket_arr[array[i] - min_element].append(array[i])

        for i in range(0, len(bucket_arr)):
            bucket_arr[i] = sorted(bucket_arr[i])

        for i in range(0, len(bucket_arr)):
            for j in range(0, len(bucket_arr[i])):
                output_arr.append(bucket_arr[i][j])
        return output_arr


all_algorithms = [
    {
        "id": 0,
        "name": "Bubble Sort",
        "instance": BubbleSort(),
        "description": "Bubble Sort is the simplest sorting algorithm that works by repeatedly swapping the adjacent elements if they are in wrong order.",
        "time_complexity": "O(n^2)",
        "space_complexity": "O(1)"
    },
    {
        "id": 1,
        "name": "Selection Sort",
        "instance": SelectionSort(),
        "description": "The selection sort algorithm sorts an array by repeatedly finding the minimum element (considering ascending order) from unsorted part and putting it at the beginning.",
        "time_complexity": "O(n^2)",
        "space_complexity": "O(1)"
    },
    {
        "id": 2,
        "name": "Insertion Sort",
        "instance": InsertionSort(),
        "description": "Insertion sort is a simple sorting algorithm that works similar to the way you sort playing cards in your hands.",
        "time_complexity": "O(n^2)",
        "space_complexity": "O(1)"
    },
    {
        "id": 3,
        "name": "Merge Sort",
        "instance": MergeSort(),
        "description": "Merge Sort is a Divide and Conquer algorithm. It divides the input array into two halves, calls itself for the two halves, and then merges the two sorted halves.",
        "time_complexity": "O(nlogn)",
        "space_complexity": "O(n)"
    },
    {
        "id": 4,
        "name": "Quick Sort",
        "instance": QuickSort(),
        "description": "QuickSort is a Divide and Conquer algorithm. It picks an element as pivot and partitions the given array around the picked pivot.",
        "time_complexity": "O(nlogn)",
        "space_complexity": "O(1)"
    },
    {
        "id": 5,
        "name": "Heap Sort",
        "instance": HeapSort(),
        "description": "Heap sort is a comparison based sorting technique based on Binary Heap data structure. It is similar to selection sort where we first find the maximum element and place the maximum element at the end.",
        "time_complexity": "O(nlogn)",
        "space_complexity": "O(1)"
    },
    {
        "id": 6,
        "name": "Radix Sort",
        "instance": RadixSort(),
        "description": "Radix sort is a sorting technique that sorts the elements by first grouping the individual digits of the same place value. Then, sort the elements according to their increasing/decreasing order.",
        "time_complexity": "O(nk)",
        "space_complexity": "O(n+k)"
    },
    {
        "id": 7,
        "name": "Counting Sort",
        "instance": CountingSort(),
        "description": "Counting sort is a sorting technique based on keys between a specific range. It works by counting the number of objects having distinct key values.",
        "time_complexity": "O(n+k)",
        "space_complexity": "O(n+k)"
    },
    {
        "id": 8,
        "name": "Bucket Sort",
        "instance": BucketSort(),
        "description": "Bucket sort is mainly useful when input is uniformly distributed over a range. For example, consider the following problem. Sort a large set of floating point numbers which are in range from 0.0 to 1.0 and are uniformly distributed across the range.",
        "time_complexity": "O(n+k)",
        "space_complexity": "O(n+k)"
    }
]


def run_callback(args):
    if not gui.get_input_text():
        logger("Please enter some numbers")
        return
    # if invalid input

    nums = list(map(int, gui.get_input_text().split()))

    for id, checkbox in gui.get_checkboxes().items():
        if checkbox.isChecked():
            # find the algorithm from id
            logger("Id is " + str(id))
            logger(all_algorithms[id]['name'])


def analyze_callback(args):
    logger(f"Analyze button clicked from {args}")


def show_stats_callback(args):
    logger(f"Show Stats button clicked from {args}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = SortingApp()
    logger = gui.log_message
    gui.on_run_button_clicked(run_callback)
    gui.on_analyze_button_clicked(analyze_callback)
    gui.on_show_stats_button_clicked(show_stats_callback)
    sys.exit(app.exec_())
    # App End
