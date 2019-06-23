---
layout: post
title: 归并排序 - MergeSort
date: 2017-12-22 21:10 +0800
categories: LeetCode
tags:
- 技术
mathjax: true
copyright: true
---

## <center> 归并排序 - MergeSort </center>

------

### 合并函数

```
void merge(int *arr, int left, int mid, int right)
{
	int *tmparr = new int[right - left + 1];
	int leftindex = left, rightindex = mid + 1;
	int tmpindex = 0;
	while (leftindex <= mid && rightindex <= right)
	{
		if (arr[leftindex] < arr[rightindex])
			tmparr[tmpindex++] = arr[leftindex++];
		else
			tmparr[tmpindex++] = arr[rightindex++];
	}
	while (leftindex <= mid)
		tmparr[tmpindex++] = arr[leftindex++];
	while (rightindex <= right)
		tmparr[tmpindex++] = arr[rightindex++];
	tmpindex = 0;
	while (left <= right)
		arr[left++] = tmparr[tmpindex++];
}
```

--------

### 递归

```cpp
void mergesort_recursive(int *arr, int left, int right)
{
	if (left >= right)
		return;
	int mid = (left + right) / 2;
	mergesort_recursive(arr, left, mid);
	mergesort_recursive(arr, mid + 1, right);
	merge(arr, left, mid, right);
}
```

--------

### 非递归

```
void mergesort_nonerecursive(int *arr, int left, int right)
{
	int step = 2, i = 0;
	while (step <= right + 1)
	{
		i = 0;
		while (i + step <= right)
		{
			merge(arr, i, i + step / 2 - 1, i + step - 1);
			i += step;
		}
		merge(arr, i, i + step / 2 - 1, right);
		step *= 2;
	}
	merge(arr, 0, step / 2 - 1, right);
}
```
