package median

import (
	"container/heap"
)

type MinIntHeap []int

func (this MinIntHeap) Swap(i, j int) {
	this[i], this[j] = this[j], this[i]
}

func (this MinIntHeap) Len() int {
	return len(this)
}

func (this MinIntHeap) Less(i, j int) bool {
	return this[i] < this[j]
}

func (this *MinIntHeap) Push(v any) {
	*this = append(*this, v.(int))
}

func (this *MinIntHeap) Pop() any {
	s := *this
	if len(s) == 0 {
		panic("empty")
	}
	sLen := len(s)
	v := s[sLen-1]
	*this = s[:sLen-1]
	return v
}

type MaxIntHeap []int

func (this MaxIntHeap) Swap(i, j int) {
	this[i], this[j] = this[j], this[i]
}

func (this MaxIntHeap) Len() int {
	return len(this)
}

func (this MaxIntHeap) Less(i, j int) bool {
	return this[i] > this[j]
}

func (this *MaxIntHeap) Push(v any) {
	*this = append(*this, v.(int))
}

func (this *MaxIntHeap) Pop() any {
	s := *this
	if len(s) == 0 {
		panic("empty")
	}
	sLen := len(s)
	v := s[sLen-1]
	*this = s[:sLen-1]
	return v
}

type HeapMedian struct {
	left  MaxIntHeap
	right MinIntHeap
}

func (hm *HeapMedian) AddNum(v int) {
	lLen := len(hm.left)
	rLen := len(hm.right)

	if lLen > 0 && v < hm.left[0] {
		heap.Push(&hm.left, v)
	} else if rLen > 0 && v > hm.right[0] {
		heap.Push(&hm.right, v)
	} else if lLen < rLen {
		heap.Push(&hm.left, v)
	} else if lLen > rLen {
		heap.Push(&hm.right, v)
	} else {
		heap.Push(&hm.left, v)
	}

	for len(hm.left)-len(hm.right) > 1 {
		v := heap.Pop(&hm.left)
		heap.Push(&hm.right, v)
	}

	for len(hm.right)-len(hm.left) > 1 {
		v := heap.Pop(&hm.right)
		heap.Push(&hm.left, v)
	}
}

func (hm HeapMedian) GetMedian() int {
	lLen := len(hm.left)
	rLen := len(hm.right)

	if lLen < rLen {
		return hm.right[0]
	} else if lLen > rLen {
		return hm.left[0]
	} else {
		return (hm.left[0] + hm.right[0]) / 2
	}
}
