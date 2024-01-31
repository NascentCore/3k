package main

import (
	"fmt"
)

// 定义格子的坐标结构体
type Point struct {
	X int
	Y int
}

// 定义棋盘
type Board map[Point]bool

// 计算邻居活细胞数
func countAliveNeighbors(board Board, p Point) int {
	aliveNeighbors := 0
	for x := -1; x <= 1; x++ {
		for y := -1; y <= 1; y++ {
			if x == 0 && y == 0 {
				continue // 跳过当前格子
			}
			neighbor := Point{X: p.X + x, Y: p.Y + y}
			if board[neighbor] {
				aliveNeighbors++
			}
		}
	}
	return aliveNeighbors
}

// 实现 next 函数
func next(board Board) Board {
	newBoard := Board{}

	for p, isAlive := range board {
		// 计算当前格的活细胞数
		neighbors := countAliveNeighbors(board, p)

		// 应用康威生命游戏的规则
		// 如果当前格是活的且周围有2个或3个活细胞，则下一轮仍然存活
		// 如果当前格是死的但周围恰好有3个活细胞，则下一轮成为活细胞
		if (isAlive && neighbors == 2) || (isAlive && neighbors == 3) ||
			(!isAlive && neighbors == 3) {
			newBoard[p] = true
		} else {
			newBoard[p] = false
		}
	}

	return newBoard
}

func main() {
	// 初始化一个 5x5 的棋盘
	board := Board{}
	for x := 0; x < 5; x++ {
		for y := 0; y < 5; y++ {
			if x == 1 && y == 1 || x == 1 && y == 2 || x == 2 && y == 1 {
				board[Point{X: x, Y: y}] = true
			}
		}
	}

	// 打印初始棋盘
	fmt.Println("Initial board:")
	for p, isAlive := range board {
		if isAlive {
			fmt.Print("X ")
		} else {
			fmt.Print(". ")
		}
	}
	fmt.Println()

	// 计算下一轮棋盘状态
	newBoard := next(board)

	// 打印下一轮棋盘
	fmt.Println("Next board:")
	for p, isAlive := range newBoard {
		if isAlive {
			fmt.Print("X ")
		} else {
			fmt.Print(". ")
		}
	}
	fmt.Println()
}

