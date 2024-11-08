package donggang

import (
	"reflect"
	"testing"
)

func TestBubbleSort(t *testing.T) {
	type args struct {
		arr      []int
		expected []int
	}
	tests := []struct {
		name string
		args args
	}{
		// TODO: Add test cases.
		{
			name: "case-1-ok",
			args: args{
				arr:      []int{},
				expected: []int{},
			},
		},
		{
			name: "case-2-ok",
			args: args{
				arr:      []int{2, 1, 3},
				expected: []int{1, 2, 3},
			},
		},
		{
			name: "case-3-ok",
			args: args{
				arr:      []int{64, 34, 25, 12, 22, 11, 90},
				expected: []int{11, 12, 22, 25, 34, 64, 90},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			//nolint
			BubbleSort(tt.args.arr)
			if !reflect.DeepEqual(tt.args.arr, tt.args.expected) {
				t.Fatalf("expected %v got %v", tt.args.expected, tt.args.arr)
			}
		})
	}
}
