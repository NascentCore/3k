package math

import (
	"testing"
)

func TestRound(t *testing.T) {
	type args struct {
		val    float64
		places int
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		{
			name: "round to 2 decimal places",
			args: args{val: 1.23456789, places: 2},
			want: 1.23,
		},
		{
			name: "round to 3 decimal places",
			args: args{val: 1.23456789, places: 3},
			want: 1.235,
		},
		{
			name: "round to 0 decimal places",
			args: args{val: 1.23456789, places: 0},
			want: 1,
		},
		{
			name: "round to 1 decimal place",
			args: args{val: 1.25, places: 1},
			want: 1.3,
		},
		{
			name: "round negative number",
			args: args{val: -1.23456789, places: 2},
			want: -1.23,
		},
		{
			name: "round to 2 decimal places with halfway value",
			args: args{val: 1.235, places: 2},
			want: 1.24,
		},
		{
			name: "round large number",
			args: args{val: 12345.6789, places: 2},
			want: 12345.68,
		},
		{
			name: "round small number",
			args: args{val: 0.000123456789, places: 8},
			want: 0.00012346,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Round(tt.args.val, tt.args.places); got != tt.want {
				t.Errorf("round() = %v, want %v", got, tt.want)
			}
		})
	}
}
