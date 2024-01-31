package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"
	"time"
)

const (
	width  = 100
	height = 100
)

// 定义棋盘结构
type Board [width][height]bool

// 检查邻居数量
func countAliveNeighbors(board Board, x, y int) int {
	aliveNeighbors := 0
	for i := -1; i <= 1; i++ {
		for j := -1; j <= 1; j++ {
			if i == 0 && j == 0 {
				continue
			}
			nx, ny := x+i, y+j
			if nx >= 0 && nx < width && ny >= 0 && ny < height {
				aliveNeighbors += board[nx][ny]
			}
		}
	}
	return aliveNeighbors
}

// 更新棋盘状态
func updateBoard(board Board) Board {
	newBoard := Board{}
	for x := 0; x < width; x++ {
		for y := 0; y < height; y++ {
			aliveNeighbors := countAliveNeighbors(board, x, y)
			// 应用康威生命游戏的规则
			if board[x][y] {
				if aliveNeighbors < 2 || aliveNeighbors > 3 {
					newBoard[x][y] = false
				} else {
					newBoard[x][y] = true
				}
			} else {
				if aliveNeighbors == 3 {
					newBoard[x][y] = true
				}
			}
		}
	}
	return newBoard
}

// 打印棋盘
func printBoard(board Board) {
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			if board[x][y] {
				fmt.Print("X ")
			} else {
				fmt.Print(". ")
			}
		}
		fmt.Println()
	}
}

func main() {
	var board Board
	// 初始化棋盘
	for x := 0; x < width; x++ {
		for y := 0; y < height; y++ {
			board[x][y] = false
		}
	}
	// 随机设置一些活细胞
	for i := 0; i < 100; i++ {
		x := rand.Intn(width)
		y := rand.Intn(height)
		board[x][y] = true
	}

	// 游戏主循环
	for {
		printBoard(board)
		time.Sleep(time.Second / 5) // 每5帧更新一次
		board = updateBoard(board)
	}
}

