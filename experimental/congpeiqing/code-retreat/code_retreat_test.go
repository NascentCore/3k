package main

import (
	"reflect"
	"testing"
)

func Test_countAlive(t *testing.T) {
	type args struct {
		board [][]bool
		i     int
		j     int
	}
	tests := []struct {
		name string
		args args
		want int
	}{
		{
			"test1",
			args{
				[][]bool{
					{true, false, false},
					{false, false, false},
					{false, false, false}},
				0,
				0},
			0,
		},
		{
			"test2",
			args{
				[][]bool{
					{true, false, false},
					{false, true, false},
					{false, false, false}},
				1,
				1},
			1,
		}, {
			"test3",
			args{
				[][]bool{
					{true, false, false},
					{false, true, false},
					{false, false, false}},
				2,
				2},
			1,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := countAlive(tt.args.board, tt.args.i, tt.args.j); got != tt.want {
				t.Errorf("countAlive() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_next(t *testing.T) {
	type args struct {
		board [][]bool
	}
	tests := []struct {
		name string
		args args
		want [][]bool
	}{
		{
			"test1",
			args{
				[][]bool{
					{false, false, false, false, false},
					{false, false, true, false, false},
					{false, false, true, false, false},
					{false, false, true, false, false},
					{false, false, false, false, false}},
			},
			[][]bool{
				{false, false, false, false, false},
				{false, false, false, false, false},
				{false, true, true, true, false},
				{false, false, false, false, false},
				{false, false, false, false, false}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := next(tt.args.board); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("next() = %v, want %v", got, tt.want)
			}
		})
	}
}
