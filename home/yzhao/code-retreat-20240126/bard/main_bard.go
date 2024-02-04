package gencode

import (
    "fmt"
)

type CellState int

const (
    CellAlive CellState = 1
    CellDead CellState = 0
)

type Point struct {
    x, y int
}

func next(board [][]CellState) [][]CellState {
    newBoard := make([][]CellState, len(board))
    for i := range board {
        newBoard[i] = make([]CellState, len(board[0]))
    }
    for i := range board {
        for j := range board[i] {
            neighbors := countNeighbors(board, Point{i, j})
            newState := board[i][j]
            switch neighbors {
            case 2:
                newState = CellAlive
            case 3:
                newState = CellAlive
            case 4, 5, 6, 7, 8:
                newState = CellDead
            }
            newBoard[i][j] = newState
        }
    }
    return newBoard
}

func countNeighbors(board [][]CellState, point Point) int {
    neighbors := 0
    for dx := -1; dx <= 1; dx++ {
        for dy := -1; dy <= 1; dy++ {
            neighbor := Point{point.x + dx, point.y + dy}
            if neighbor.x >= 0 && neighbor.x < len(board) && neighbor.y >= 0 && neighbor.y < len(board[0]) {
                if board[neighbor.x][neighbor.y] == CellAlive {
                    neighbors++
                }
            }
        }
    }
    return neighbors
}

func main() {
    board := [][]CellState{
        {CellAlive, CellAlive, CellAlive},
        {CellAlive, CellDead, CellAlive},
        {CellAlive, CellAlive, CellAlive},
    }
    for {
        board = next(board)
        fmt.Println(board)
    }
}

