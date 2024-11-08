package main

import (
	"fmt"
	"math/rand"
	"time"
)

const (
	width  = 100
	height = 100
)

var (
	board    [height][width]bool
	newBoard [height][width]bool
)

func initialize() {
	for i := 0; i < height; i++ {
		for j := 0; j < width; j++ {
			board[i][j] = rand.Intn(2) == 1
		}
	}
}

func printBoard() {
	for i := 0; i < height; i++ {
		for j := 0; j < width; j++ {
			if board[i][j] {
				fmt.Print("██")
			} else {
				fmt.Print("  ")
			}
		}
		fmt.Println()
	}
}

func countNeighbors(x, y int) int {
	count := 0
	for i := -1; i <= 1; i++ {
		for j := -1; j <= 1; j++ {
			if (i != 0 || j != 0) && (x+i >= 0) && (x+i < height) && (y+j >= 0) && (y+j < width) {
				if board[x+i][y+j] {
					count += 1
				}
			}
		}
	}
	return count
}

func updateBoard() {
	for i := 0; i < height; i++ {
		for j := 0; j < width; j++ {
			neighbors := countNeighbors(i, j)
			if board[i][j] {
				if neighbors == 2 || neighbors == 3 {
					newBoard[i][j] = true
				} else {
					newBoard[i][j] = false
				}
			} else {
				if neighbors == 3 {
					newBoard[i][j] = true
				} else {
					newBoard[i][j] = false
				}
			}
		}
	}
	// Swap boards
	board, newBoard = newBoard, board
}

func main() {
	initialize()

	for {
		printBoard()
		fmt.Println()
		time.Sleep(500 * time.Millisecond)
		updateBoard()
	}
}
