package bubblesort

import (
	"testing"
	"reflect"
)

func TestBubbleSort(t *testing.T) {
	ints := []int{5, 4, 3, 2, 1}
	BubbleSort(ints)
	expInts := []int{1, 2, 3, 4, 5}
	if !reflect.DeepEqual(ints, expInts) {
		t.Fatalf("expected %v got %v", expInts, ints)
	}
}
