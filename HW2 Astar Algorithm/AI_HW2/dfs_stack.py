import csv
edgeFile = 'edges.csv'


def dfs(start, end):
    ''' 
    Read the edges.csv file and put the node data in a dict() called graph.
    Use another dict() called visited to record the nodes' parents, if it is not visited, the value is -1.
    And do the dfs, then find the final path by looking the parents of the nodes in the visited dict(), 
    calculate the total distance at the same time.
    '''
    # Begin your code (Part 2)
    file = open(edgeFile, 'r')
    csvFile = csv.reader(file)
    data = []
    header = []
    header = next(csvFile)
    for row in csvFile:
        data.append(row)
    file.close()
    graph = dict()
    visited = dict()
    for i in range(len(data)):
        data[i][0], data[i][1], data[i][2], data[i][3] = int(data[i][0]), int(data[i][1]), float(data[i][2]), float(data[i][3])
        visited[data[i][0]] = -1
        visited[data[i][1]] = -1
        if data[i][0] not in graph:
            graph[data[i][0]] = dict()
        if data[i][1] not in graph:
            graph[data[i][1]] = dict()
        graph[data[i][0]][data[i][1]] = (data[i][2], data[i][3])
    visited[start] = 0
    stack = []
    stack.append(start)
    dist = 0
    num_visited = 1
    flag = True
    while (len(stack) > 0 and flag):
        current = stack.pop()
        for neighbor in graph[current]:
            if visited[neighbor] == -1: 
                visited[neighbor] = current
                if neighbor == end:
                    flag = False
                    break                      
                stack.append(neighbor)
                num_visited += 1
    path = []
    current = end
    if not flag:
        while visited[current] != 0:
            path.append(visited[current])
            dist += graph[visited[current]][current][0]
            current = visited[current]
    path.append(current)
    path.reverse()
    return path, dist, num_visited
    # End your code (Part 2)


if __name__ == '__main__':
    testcases = ((2270143902, 1079387396), (426882161, 1737223506), (1718165260, 8513026827))
    for testcase in testcases:
        path, dist, num_visited = dfs(testcase[0], testcase[1])
        print(f'The number of path nodes: {len(path)}')
        print(f'Total distance of path: {dist}')
        print(f'The number of visited nodes: {num_visited}')
