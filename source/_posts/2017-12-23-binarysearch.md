---
layout: post
title: 二分查找 - BinarySearch
date: 2017-12-23 21:10 +0800
categories: LeetCode
tags:
- 技术
mathjax: true
copyright: true
---


### 递归

```cpp
int binarysearch_recursion(vector<int> arr, int start, int end, int key)
{
	int mid = (end - start) / 2 + start;
	if (arr[mid] == key)
		return mid;
	if (start >= end)
		return -1;
	else if (key > arr[mid])
		return binarysearch_recursion(arr, mid + 1, end, key);
	else if (key < arr[mid])
		return binarysearch_recursion(arr, start, mid - 1, key);
	return -1;
}
```


-----------

### 非递归

```cpp
int binarysearch_nonrecursion(vector<int> arr, int key)
{
	int mid, start = 0, end = arr.size() - 1;
	while (start <= end)
	{
		mid = (end - start) / 2 + start;
		if (key < arr[mid])
			end = mid - 1;
		else if (key > arr[mid])
			start = mid + 1;
		else
			return mid;
	}
	return -1;
}
```


----------

### 二分排序

```cpp
//O(nlogn)
void binarysort(vector<int> &arr)
{
	int start, end, tmp = 0, mid, j;
	for (int i = 1; i < arr.size(); i++)
	{
		start = 0; 
		end = i - 1; 
		tmp = arr[i];
		while (start <= end)
		{
			mid = (start + end) / 2;
			if (tmp < arr[mid])
				end = mid - 1;
			else
				start = mid + 1;
		}
		for (j = i - 1; j >= start; j--)
		{
			arr[j + 1] = arr[j];
		}
		arr[start] = tmp;
	}
}
```
