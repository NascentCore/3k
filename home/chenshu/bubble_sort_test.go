package chenshu

import (
	"reflect"
	"testing"
)

func Test_bubbleSort(t *testing.T) {
	type args struct {
		nums []int
	}
	tests := []struct {
		name string
		args args
		want []int
	}{
		{
			args: args{nums: []int{3, 2, 1, 4}},
			want: []int{1, 2, 3, 4},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if BubbleSort(tt.args.nums); !reflect.DeepEqual(tt.args.nums, tt.want) {
				t.Errorf("BubbleSort() = %v, want %v", tt.args.nums, tt.want)
			}
		})
	}
}
