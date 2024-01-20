package main

import (
	"fmt"
	"strconv"
	"strings"
)

type point struct {
	x int
	y int
}

func toString(x, y int) string {
	return fmt.Sprintf("%s,%s", strconv.Itoa(x), strconv.Itoa(y))
}

func toPoint(str string) point {
	strs := strings.Split(str, ",")
	x, _ := strconv.Atoi(strs[0])
	y, _ := strconv.Atoi(strs[1])
	return point {x, y}
}

func countAlive(board map[string]bool, x, y int) int {
	count := 0
	// return the number of alive neighbors of x, y
	for xOffset := range []int{-1, 0, 1} {
		for yOffset := range []int{-1, 0, 1} {
			x1 := x + xOffset
			y1 := y + yOffset
			if exist, ok := board[toString(x1, y1)]; ok && exist {
				count += 1
			}
		}
	}
	return count
}

func isAlive(gameBoard map[string]bool, p point) bool {
	exist, ok := gameBoard[toString(p.x, p.y)]
	return ok && exist
}

func next(gameBoard map[string]bool) map[string]bool {
	pointsToCheck := make([]point, 0, 100)
	for pointStr, _ := range gameBoard {
		p := toPoint(pointStr)
		for xOffset := range []int{-1, 0, 1} {
			for yOffset := range []int{-1, 0, 1} {
				x1 := p.x + xOffset
				y1 := p.y + yOffset
				pointsToCheck = append(pointsToCheck, point{x1, y1})
			}
		}
	}

	// TODO: Add dedup

	newBoard := make(map[string]bool)

	for _, point := range pointsToCheck {
		fmt.Println("examining point: %v", point)
		aliveCount := countAlive(gameBoard, point.x, point.y)
		if isAlive(gameBoard, point) && aliveCount == 2 || aliveCount == 3 {
			newBoard[toString(point.x, point.y)] = true
		}
		if !isAlive(gameBoard, point) && aliveCount == 3 {
			newBoard[toString(point.x, point.y)] = true
		}
	}

	return newBoard
}

func main() {
	b := make(map[string]bool)
	b["0,0"] = true
	b["0,1"] = true
	b["1,0"] = true
	fmt.Println(b)
	next := next(b)
	fmt.Println(next)
	test := make(map[point]bool)
	test[point{0, 0}] = true
	test[point{0, 0}] = false
	fmt.Println(test)
}
