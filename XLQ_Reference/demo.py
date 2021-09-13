class cell:
	
	id = 0										# 0：空，1：起点，2：终点，5，中间路程，9，墙
	father = None
	sta_cost = 9999
	avi_cost = 9999
	cost = 9999

	def __init__(self,id = 0,sta_cost = 9999,avi_cost = 9999):
		self.id = id
		self.sta_cost = sta_cost
		self.avi_cost = avi_cost
		self.cost = self.sta_cost + self.avi_cost

	def id_change(self,id_new):
		self.id = id_new

	def change_avi_cost(self,current_cost):
		if current_cost > self.avi_cost:
			return False
		self.avi_cost = current_cost
		self.cost = self.sta_cost + self.avi_cost
		return True

def board_print(board):
	for i in range(5):
		for j in range(7):
			print(board[i][j].id, end = '  ')
		print('\n')

def board_init(start,end,walls):
	new_board = []
	for i in range(5):
		board_row = []
		for j in range(7):
			board_row.append(cell(id = 0,sta_cost = 10*(abs(i-end[0])+abs(j-end[1]))))
		new_board.append(board_row)
	new_board[start[0]][start[1]].id = 1
	new_board[start[0]][start[1]].avi_cost = 0
	new_board[end[0]][end[1]].id = 2
	for wall in walls:
		new_board[wall[0]][wall[1]].id = 9
	return new_board

def available(x,y,closed_list):
	if x<0 or x>4:
		return False
	if y<0 or y>6:
		return False
	if board[x][y].id == 9:
		return False
	if [x,y] in closed_list:
		return False
	return True

def find(current,x,y,open_list):
	global board
	[m, n] = current
	avi_cost = 14 + board[m][n].avi_cost
	if [x,y] not in open_list:
		board[x][y].father = current
		open_list.append([x,y])
		board[x][y].change_avi_cost(avi_cost)
	else:
		if board[x][y].change_avi_cost(avi_cost):
			board[x][y].father = current
	return open_list

def sort_by_cost(open_list):
	[x,y] = open_list
	return board[x][y].cost

def a_star(start,end,walls):
	global board
	[x, y] = start
	open_list = []
	closed_list = []
	current = start
	while 1:
		if current == end:
			break
		[x, y] = current
		closed_list.append(current)
		if available(x-1,y-1,closed_list):
			open_list = find(current,x-1,y-1,open_list)
		if available(x,y-1,closed_list):
			open_list = find(current,x,y-1,open_list)
		if available(x+1,y-1,closed_list):
			open_list = find(current,x+1,y-1,open_list)
		if available(x-1,y+1,closed_list):
			open_list = find(current,x-1,y+1,open_list)
		if available(x,y+1,closed_list):
			open_list = find(current,x,y+1,open_list)
		if available(x+1,y+1,closed_list):
			open_list = find(current,x+1,y+1,open_list)
		if available(x-1,y,closed_list):
			open_list = find(current,x-1,y,open_list)
		if available(x+1,y,closed_list):
			open_list = find(current,x+1,y,open_list)
		open_list.sort(key = sort_by_cost)
		current = open_list.pop(0)
		print(1)
	
	[x, y] = current
	[x, y] = board[x][y].father
	while board[x][y].father:
		board[x][y].id = 5
		[x, y] = board[x][y].father
		print(2)

start = [2,0]
end = [2,5]
walls = [[1,3],[2,3],[3,3]]
board = board_init(start,end,walls)
board_print(board)

if __name__ == "__main__":
	a_star(start,end,walls)
	board_print(board)