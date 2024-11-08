package main

import (
	"fmt"
)

const (
	alive = "O" // 活细胞
	dead  = "X" // 死细胞
)

var directions = [][]int{
	{-1, -1}, {-1, 0}, {-1, 1},
	{0, -1}, {0, 1},
	{1, -1}, {1, 0}, {1, 1},
}

// 计算指定位置 (i, j) 处活细胞的数量
func countAliveNeighbors(grid [][]string, i, j int) int {
	rows, cols := len(grid), len(grid[0])
	countAlive := 0

	for _, dir := range directions {
		newI, newJ := i+dir[0], j+dir[1]
		if newI >= 0 && newI < rows && newJ >= 0 && newJ < cols && grid[newI][newJ] == alive {
			countAlive++
		}
	}

	return countAlive
}

// 根据规则更新整个网格的状态
func updateGrid(grid [][]string) [][]string {
	rows, cols := len(grid), len(grid[0])
	newGrid := make([][]string, rows)

	// 初始化新网格
	for i := range grid {
		newGrid[i] = make([]string, cols)
	}

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			countAlive := countAliveNeighbors(grid, i, j)

			if grid[i][j] == alive {
				// 活细胞规则
				if countAlive < 2 || countAlive > 3 {
					newGrid[i][j] = dead
				} else {
					newGrid[i][j] = alive
				}
			} else {
				// 死细胞规则
				if countAlive == 3 {
					newGrid[i][j] = alive
				} else {
					newGrid[i][j] = dead
				}
			}
		}
	}

	return newGrid
}

func printGrid(grid [][]string) {
	for _, row := range grid {
		fmt.Println(row)
	}
	fmt.Println()
}

func main() {
	// 初始化初始网格
	initialGrid := [][]string{
		{"X", "X", "X", "X", "X"},
		{"X", "X", "X", "X", "X"},
		{"X", "O", "O", "O", "X"},
		{"X", "X", "X", "X", "X"},
		{"X", "X", "X", "X", "X"},
	}

	printGrid(initialGrid)

	for round := 1; round <= 10; round++ {
		initialGrid = updateGrid(initialGrid)
		fmt.Printf("第 %d 轮：\n", round)
		printGrid(initialGrid)
	}
}
