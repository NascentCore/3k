package main

import "fmt"

func printBoard(board [][]bool) {
	for _, line := range board {
		for _, b := range line {
			if b {
				fmt.Print("O ")
			} else {
				fmt.Print("X ")
			}
		}
		fmt.Print("\n")
	}
}

func main() {
	board := [][]bool{
		{false, false, false, false, false},
		{false, false, true, false, false},
		{false, false, true, false, false},
		{false, false, true, false, false},
		{false, false, false, false, false}}

	for i := 0; i < 10; i++ {
		fmt.Printf("-- %d ----------\n", i)
		board = next(board)
		printBoard(board)
	}

}
