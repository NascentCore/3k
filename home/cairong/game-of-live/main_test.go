package main

import (
	"reflect"
	"testing"
)

func TestCountAliveNeighbors(t *testing.T) {
	grid := [][]string{
		{"X", "O", "X"},
		{"O", "X", "O"},
		{"X", "O", "X"},
	}

	tests := []struct {
		i, j     int
		expected int
	}{
		{0, 0, 2},
		{1, 1, 4},
		{2, 2, 2},
	}

	for _, test := range tests {
		result := countAliveNeighbors(grid, test.i, test.j)
		if result != test.expected {
			t.Errorf("countAliveNeighbors(%d, %d) = %d; expected %d", test.i, test.j, result, test.expected)
		}
	}
}

func TestUpdateGrid(t *testing.T) {
	tests := []struct {
		initialGrid [][]string
		expected    [][]string
	}{
		{
			initialGrid: [][]string{
				{"X", "O", "X"},
				{"X", "O", "X"},
				{"X", "O", "X"},
			},
			expected: [][]string{
				{"X", "X", "X"},
				{"O", "O", "O"},
				{"X", "X", "X"},
			},
		},
		{
			initialGrid: [][]string{
				{"X", "X", "X"},
				{"X", "O", "X"},
				{"X", "O", "O"},
			},
			expected: [][]string{
				{"X", "X", "X"},
				{"X", "O", "O"},
				{"X", "O", "O"},
			},
		},
	}

	for _, test := range tests {
		result := updateGrid(test.initialGrid)
		if !reflect.DeepEqual(result, test.expected) {
			t.Errorf("updateGrid(%v) = %v; expected %v", test.initialGrid, result, test.expected)
		}
	}
}
