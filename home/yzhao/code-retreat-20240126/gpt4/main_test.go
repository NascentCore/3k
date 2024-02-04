package gpt4

import (
    "reflect"
    "testing"
)

// 测试 getNeighbors 函数
func TestGetNeighbors(t *testing.T) {
    p := Point{X: 1, Y: 1}
    expectedNeighbors := []Point{
        {0, 0}, {1, 0}, {2, 0},
        {0, 1},        {2, 1},
        {0, 2}, {1, 2}, {2, 2},
    }

    neighbors := getNeighbors(p)
    if !reflect.DeepEqual(neighbors, expectedNeighbors) {
        t.Errorf("getNeighbors(%v) = %v, want %v", p, neighbors, expectedNeighbors)
    }
}

// 测试 countAliveNeighbors 函数
func TestCountAliveNeighbors(t *testing.T) {
    board := Board{
        {0, 0}: true, {1, 0}: true, // 活细胞
        {2, 2}: true, // 活细胞
    }
    p := Point{X: 1, Y: 1}

    expectedCount := 3
    count := countAliveNeighbors(board, p)
    if count != expectedCount {
        t.Errorf("countAliveNeighbors(%v, %v) = %d, want %d", board, p, count, expectedCount)
    }
}

// 测试 next 函数
func TestNext(t *testing.T) {
    initialBoard := Board{
        {0, 1}: true, {1, 1}: true, {2, 1}: true, // 初始活细胞
    }

    expectedBoard := Board{
        {1, 0}: true, {1, 1}: true, {1, 2}: true, // 预期的下一个状态
    }

    nextBoard := next(initialBoard)
		printBoard(initialBoard)
		printBoard(expectedBoard)
    if !reflect.DeepEqual(nextBoard, expectedBoard) {
        t.Errorf("next(%v) = %v, want %v", initialBoard, nextBoard, expectedBoard)
    }
}

// 测试其他函数...

