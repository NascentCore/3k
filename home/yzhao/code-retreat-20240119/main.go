package main

import (
	"fmt"
	"strconv"
	"strings"
	"time"
)

type point struct {
	x int
	y int
}

func pointToString(p point) string {
	return fmt.Sprintf("%s,%s", strconv.Itoa(p.x), strconv.Itoa(p.y))
}

func toPoint(str string) point {
	strs := strings.Split(str, ",")
	x, _ := strconv.Atoi(strs[0])
	y, _ := strconv.Atoi(strs[1])
	return point {x, y}
}

func getNeighbors(p point) []point {
	points := make([]point, 0, 8)
	for _, xOffset := range []int{-1, 0, 1} {
		for _, yOffset := range []int{-1, 0, 1} {
			if xOffset == 0 && yOffset == 0 {
				continue
			}
			points = append(points, point{p.x + xOffset, p.y + yOffset})
		}
	}
	return points
}

func countAlive(board map[string]bool, p point) int {
	count := 0
	neighbors := getNeighbors(p)
	for _, n := range neighbors {
		if _, ok := board[pointToString(n)]; ok {
			count += 1
		}
	}
	return count
}

func isAlive(gameBoard map[string]bool, p point) bool {
	exist, ok := gameBoard[pointToString(p)]
	return ok && exist
}

func next(gameBoard map[string]bool) map[string]bool {
	pointsToCheck := make([]point, 0, 100)
	for pointStr, _ := range gameBoard {
		point := toPoint(pointStr)
		pointsToCheck = append(pointsToCheck, getNeighbors(point)...)
	}

	// TODO: Add dedup

	newBoard := make(map[string]bool)

	for _, point := range pointsToCheck {
		aliveCount := countAlive(gameBoard, point)
		if isAlive(gameBoard, point) && aliveCount == 2 || aliveCount == 3 {
			newBoard[pointToString(point)] = true
		}
		if !isAlive(gameBoard, point) && aliveCount == 3 {
			newBoard[pointToString(point)] = true
		}
	}

	return newBoard
}

func main() {
	b := make(map[string]bool)
	b["0,0"] = true
	b["0,1"] = true
	b["0,2"] = true

	fmt.Println(b)

	for len(b) > 0 {
		b = next(b)
		fmt.Println(b)
		time.Sleep(1 * time.Second)
	}
}
