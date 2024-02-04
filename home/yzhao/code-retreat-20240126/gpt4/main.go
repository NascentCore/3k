package main

import (
    "fmt"
		"math"
)

// Point 定义格子的坐标
type Point struct {
    X, Y int
}

// Board 定义棋盘类型
type Board map[Point]bool

// getNeighbors 返回一个细胞周围的所有邻居坐标
func getNeighbors(p Point) []Point {
    return []Point{
        {p.X - 1, p.Y - 1}, {p.X, p.Y - 1}, {p.X + 1, p.Y - 1},
        {p.X - 1, p.Y},                     {p.X + 1, p.Y},
        {p.X - 1, p.Y + 1}, {p.X, p.Y + 1}, {p.X + 1, p.Y + 1},
    }
}

// countAliveNeighbors 计算活着的邻居数量
func countAliveNeighbors(b Board, p Point) int {
    count := 0
    for _, neighbor := range getNeighbors(p) {
        if b[neighbor] {
            count++
        }
    }
    return count
}

// getAllPoints 获取棋盘上所有细胞及其邻居的坐标
func getAllPoints(b Board) []Point {
    pointsMap := make(map[Point]bool)
    for p := range b {
        if !pointsMap[p] {
            pointsMap[p] = true
        }
        for _, neighbor := range getNeighbors(p) {
            if !pointsMap[neighbor] {
                pointsMap[neighbor] = true
            }
        }
    }

    var points []Point
    for p := range pointsMap {
        points = append(points, p)
    }
    return points
}

// next 计算下一状态的棋盘
func next(currentBoard Board) Board {
    newBoard := make(Board)
    for _, p := range getAllPoints(currentBoard) {
        alive := currentBoard[p]
        aliveNeighbors := countAliveNeighbors(currentBoard, p)

        // 应用生命游戏的规则
        if alive && (aliveNeighbors == 2 || aliveNeighbors == 3) {
            newBoard[p] = true
        } else if !alive && aliveNeighbors == 3 {
            newBoard[p] = true
        }
    }
    return newBoard
}

// next 计算下一状态的棋盘
/*
func next(currentBoard Board) Board {
    newBoard := make(Board)
    for p, alive := range currentBoard {
        aliveNeighbors := countAliveNeighbors(currentBoard, p)

        // 应用生命游戏的规则
        if alive && (aliveNeighbors == 2 || aliveNeighbors == 3) {
            newBoard[p] = true
        } else if !alive && aliveNeighbors == 3 {
            newBoard[p] = true
        }
    }
    return newBoard
}
*/

// findBounds 找到棋盘的边界
func findBounds(b Board) (int, int, int, int) {
    minX, maxX, minY, maxY := math.MaxInt64, math.MinInt64, math.MaxInt64, math.MinInt64
    for p := range b {
        if p.X < minX {
            minX = p.X
        }
        if p.X > maxX {
            maxX = p.X
        }
        if p.Y < minY {
            minY = p.Y
        }
        if p.Y > maxY {
            maxY = p.Y
        }
    }
    return minX, maxX, minY, maxY
}

// printBoard 打印棋盘的状态
func printBoard(b Board) {
    minX, maxX, minY, maxY := findBounds(b)
    for y := minY; y <= maxY; y++ {
        for x := minX; x <= maxX; x++ {
            if b[Point{x, y}] {
                fmt.Print("O")
            } else {
                fmt.Print("X")
            }
        }
        fmt.Println()
    }
}

func main() {
    // 初始化棋盘（您可以根据需要添加或修改）
		board := Board{
        {X: 0, Y: 0}: true, // 第一个活细胞
        {X: 1, Y: 0}: true, // 第二个活细胞
        {X: 2, Y: 0}: true, // 第三个活细胞
    }
		printBoard(board)

    // 打印下一状态的棋盘
    nextBoard := next(board)
		printBoard(nextBoard)
}

