package main

// Board represents the game board
type Board struct {
	Width  int
	Height int
	Cells  [][]bool
	Temp   [][]bool // Temporary state for calculations
}

// NewBoard creates a new Board of given width and height
func NewBoard(width, height int) *Board {
	cells := make([][]bool, height)
	temp := make([][]bool, height)
	for i := range cells {
		cells[i] = make([]bool, width)
		temp[i] = make([]bool, width)
	}

	return &Board{
		Width:  width,
		Height: height,
		Cells:  cells,
		Temp:   temp,
	}
}

func Next(board *Board) {}

func countLiveNeighbors(board *Board, x, y int) int {
	return 0
}
