package main

import (
	"fmt"
	"os"
	"runtime"
	"strconv"
	"strings"
	"time"
)

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

// Next updates the board to its next state
func Next(board *Board) {
	for y := 0; y < board.Height; y++ {
		for x := 0; x < board.Width; x++ {
			liveNeighbors := countLiveNeighbors(board, x, y)
			if board.Cells[y][x] {
				// Any live cell with two or three live neighbors survives.
				board.Temp[y][x] = liveNeighbors == 2 || liveNeighbors == 3
			} else {
				// Any dead cell with three live neighbors becomes a live cell.
				board.Temp[y][x] = liveNeighbors == 3
			}
		}
	}

	// Swap cells with temp for the next iteration
	board.Cells, board.Temp = board.Temp, board.Cells
}

// countLiveNeighbors counts the live neighbors of a cell
func countLiveNeighbors(board *Board, x, y int) int {
	count := 0
	for dy := -1; dy <= 1; dy++ {
		for dx := -1; dx <= 1; dx++ {
			if dx == 0 && dy == 0 {
				continue // Skip the cell itself
			}

			nx, ny := x+dx, y+dy
			if nx >= 0 && nx < board.Width && ny >= 0 && ny < board.Height && board.Cells[ny][nx] {
				count++
			}
		}
	}
	return count
}

// printBoard prints the board to the console
func printBoard(board *Board) {
	for y := 0; y < board.Height; y++ {
		for x := 0; x < board.Width; x++ {
			if board.Cells[y][x] {
				fmt.Print("O")
			} else {
				fmt.Print(".")
			}
		}
		fmt.Println()
	}
}

func clearScreen() {
	if runtime.GOOS == "windows" {
		// Windows-specific screen clearing can be complex in Go without flashing a command prompt window
		fmt.Println("\033[H\033[2J")
	} else {
		// Unix-like OS (Linux, macOS)
		fmt.Print("\033[H\033[2J")
	}
}

func main() {
	if len(os.Args) < 5 {
		fmt.Println("Usage: go run gameoflife.go [width] [height] [steps] [live cell positions...]")
		fmt.Println("Example: go run gameoflife.go 5 5 10 1,2 2,2 3,2")
		return
	}

	width, err := strconv.Atoi(os.Args[1])
	if err != nil {
		fmt.Println("Invalid width")
		return
	}

	height, err := strconv.Atoi(os.Args[2])
	if err != nil {
		fmt.Println("Invalid height")
		return
	}

	steps, err := strconv.Atoi(os.Args[3])
	if err != nil {
		fmt.Println("Invalid steps")
		return
	}

	board := NewBoard(width, height)
	// Initialize live cells based on input
	for _, pos := range os.Args[4:] {
		coords := strings.Split(pos, ",")
		if len(coords) != 2 {
			fmt.Printf("Invalid coordinate format: %s\n", pos)
			return
		}
		x, err := strconv.Atoi(coords[0])
		if err != nil {
			fmt.Printf("Invalid X coordinate: %s\n", coords[0])
			return
		}
		y, err := strconv.Atoi(coords[1])
		if err != nil {
			fmt.Printf("Invalid Y coordinate: %s\n", coords[1])
			return
		}
		if x < 0 || x >= width || y < 0 || y >= height {
			fmt.Printf("Coordinate out of bounds: %s\n", pos)
			return
		}
		board.Cells[y][x] = true
	}

	fmt.Println("Initial Board:")
	printBoard(board)

	for i := 0; i < steps; i++ {
		clearScreen()
		fmt.Printf("Step %d:\n", i+1)
		printBoard(board)
		time.Sleep(500 * time.Millisecond) // Delay for half a second
		Next(board)
	}
}
