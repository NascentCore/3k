package main

func countAlive(board [][]bool, i, j int) int {
	fMax := func(a, b int) int {
		if a > b {
			return a
		}
		return b
	}
	fMin := func(a, b int) int {
		if a < b {
			return a
		}
		return b
	}
	res := 0
	for ii := fMax(0, i-1); ii < fMin(len(board), i+2); ii++ {
		for jj := fMax(0, j-1); jj < fMin(len(board[i]), j+2); jj++ {
			if board[ii][jj] {
				res += 1
			}
		}
	}
	if board[i][j] {
		res -= 1
	}
	return res
}

func next(board [][]bool) [][]bool {
	newBoard := [][]bool{}
	for i := 0; i < len(board); i++ {
		newBoard = append(newBoard, make([]bool, len(board[i])))
		for j := 0; j < len(board[i]); j++ {
			alives := countAlive(board, i, j)
			//1. 对一个 O 格子，如果其周围邻居有 < 2 个 O 格子，下一轮这个格子变为 X（死掉）
			//2. 对一个 O 格子，如果其周围邻居有 2/3 个 O 格子，下一轮这个格子仍为 O（活着）
			//3. 对一个 O 格子，如果其周围邻居有 > 3 个 O 格子，下一轮这个格子变为 X（死掉）
			if board[i][j] {
				if alives == 2 || alives == 3 {
					newBoard[i][j] = true
				}
			}
			//4. 对一个 X 格子，如果其周围邻居有 3 个 O 格子，下一轮这个格子变为 O（活着）
			if !board[i][j] && alives == 3 {
				newBoard[i][j] = true
			}
		}
	}
	return newBoard
}
