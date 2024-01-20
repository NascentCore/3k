package donggang

import (
	"reflect"
	"testing"
)

func TestCountNeighbors(t *testing.T) {
	type Point struct {
		x, y int
	}
	type args struct {
		targetPoint       Point
		initialLivePoints []Point
		expected          int
	}
	tests := []struct {
		name string
		args args
	}{
		{
			name: "middle point ",
			args: args{
				targetPoint: Point{1, 1},
				initialLivePoints: []Point{
					{0, 0},
					{0, 1},
					{0, 2},
					{1, 0},
					{1, 1},
					{1, 2},
					{2, 0},
					{2, 1},
					{2, 2},
				},
				expected: 8,
			},
		},
		{
			name: "diagonal point",
			args: args{
				targetPoint: Point{0, 0},
				initialLivePoints: []Point{
					{0, 0},
					{0, 1},
					{0, 2},
					{1, 0},
					{1, 1},
					{1, 2},
					{2, 0},
					{2, 1},
					{2, 2},
				},
				expected: 3,
			},
		},
		{
			name: "Left, right, up and down point",
			args: args{
				targetPoint: Point{0, 1},
				initialLivePoints: []Point{
					{0, 0},
					{0, 1},
					{0, 2},
					{1, 0},
					{1, 1},
					{1, 2},
					{2, 0},
					{2, 1},
					{2, 2},
				},
				expected: 5,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var grid Grid

			for _, p := range tt.args.initialLivePoints {
				grid.Set(p.x, p.y, true)
			}
			negihbors := grid.CountNeighbors(tt.args.targetPoint.x, tt.args.targetPoint.y)

			if !reflect.DeepEqual(tt.args.expected, negihbors) {
				t.Fatalf("expected %v got %v", tt.args.expected, negihbors)
			}
		})
	}
}

func TestNextGeneration(t *testing.T) {
	type Point struct {
		x, y int
	}
	type args struct {
		targetPoint       Point
		initialLivePoints []Point
		expected          []Point
	}
	tests := []struct {
		name string
		args args
	}{
		{
			name: "case-1",
			args: args{
				targetPoint: Point{1, 1},
				initialLivePoints: []Point{
					{1, 1},
				},
				expected: []Point{},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			//nolint
			var grid Grid

			for _, p := range tt.args.initialLivePoints {
				grid.Set(p.x, p.y, true)
			}
			NextGenerationGrid := grid.NextGeneration()

			var expectedGrid Grid
			for _, p := range tt.args.expected {
				grid.Set(p.x, p.y, true)
			}

			if !reflect.DeepEqual(expectedGrid, NextGenerationGrid) {
				t.Fatalf("expected %v got %v", NextGenerationGrid, expectedGrid)
			}
		})
	}
}
