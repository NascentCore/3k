package main

import (
	"reflect"
	"testing"
)

func TestNext(t *testing.T) {
	tests := []struct {
		name      string
		initial   *Board
		steps     int
		wantBoard *Board
	}{
		{
			name: "Blinker Pattern",
			initial: func() *Board {
				b := NewBoard(5, 5)
				b.Cells[2][1] = true
				b.Cells[2][2] = true
				b.Cells[2][3] = true
				return b
			}(),
			steps: 1,
			wantBoard: func() *Board {
				b := NewBoard(5, 5)
				b.Cells[1][2] = true
				b.Cells[2][2] = true
				b.Cells[3][2] = true
				return b
			}(),
		},
		// Add more test cases as needed
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for i := 0; i < tt.steps; i++ {
				Next(tt.initial)
			}
			if !reflect.DeepEqual(tt.initial.Cells, tt.wantBoard.Cells) {
				t.Errorf("After Next() board = %v, want %v", tt.initial, tt.wantBoard)
			}
		})
	}
}

func Test_countLiveNeighbors(t *testing.T) {
	tests := []struct {
		name  string
		board *Board
		x     int
		y     int
		want  int
	}{
		{
			name: "Middle Cell with No Neighbors",
			board: func() *Board {
				b := NewBoard(3, 3)
				// All cells are dead
				return b
			}(),
			x:    1,
			y:    1,
			want: 0,
		},
		{
			name: "Middle Cell with Two Neighbors",
			board: func() *Board {
				b := NewBoard(3, 3)
				b.Cells[0][1] = true // Live cell
				b.Cells[1][0] = true // Live cell
				// Other cells are dead
				return b
			}(),
			x:    1,
			y:    1,
			want: 2,
		},
		{
			name: "Edge Cell with One Neighbor",
			board: func() *Board {
				b := NewBoard(3, 3)
				b.Cells[0][1] = true // Live cell
				// Other cells are dead
				return b
			}(),
			x:    0,
			y:    0,
			want: 1,
		},
		// Add more test cases as needed
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := countLiveNeighbors(tt.board, tt.x, tt.y); got != tt.want {
				t.Errorf("countLiveNeighbors() = %v, want %v", got, tt.want)
			}
		})
	}
}
