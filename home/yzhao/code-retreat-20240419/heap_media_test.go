package median

import (
	"container/heap"
	"fmt"
	"reflect"
	"testing"
)

func TestMinIntHeap(t *testing.T) {
	var h MinIntHeap
	h.Push(3)
	h.Push(2)
	h.Push(1)
	heap.Init(&h)
	if !reflect.DeepEqual([]int(h), []int{1, 2, 3}) {
		t.Error("got", h)
	}
}

func TestHeapMedian(t *testing.T) {
	var hm HeapMedian
	hm.AddNum(2)
	fmt.Println(hm)
	if hm.GetMedian() != 2 {
		t.Error("got:", hm)
	}
	hm.AddNum(4)
	if hm.GetMedian() != 3 {
		t.Error("got:", hm)
	}
	hm.AddNum(6)
	if hm.GetMedian() != 4 {
		t.Error("got:", hm)
	}
	hm.AddNum(8)
	if hm.GetMedian() != 5 {
		t.Error("got:", hm)
	}
}
