import csv
edgeFile = 'edges.csv'


def ucs_time(start, end):
    # Begin your code (Part 3)
    file = open(edgeFile, 'r')
    csvFile = csv.reader(file)
    data = []
    header = []
    header = next(csvFile)
    for row in csvFile:
        data.append(row)
    graph = dict()
    visited = dict()
    cost = dict()
    for i in range(len(data)):
        data[i][0], data[i][1], data[i][2], data[i][3] = int(data[i][0]), int(data[i][1]), float(data[i][2]), float(data[i][3])
        visited[data[i][0]] = -1
        visited[data[i][1]] = -1
        if data[i][0] not in graph:
            graph[data[i][0]] = dict()
            cost[data[i][0]] = 10 ** 8
        if data[i][1] not in graph:
            graph[data[i][1]] = dict()
            cost[data[i][1]] = 10 ** 8
        graph[data[i][0]][data[i][1]] = (data[i][2], data[i][3], data[i][2] * 3.6 / data[i][3])
    visited[start] = 0
    cost[start] = 0
    
    queue = []
    queue.append(start)
    time = 0
    num_visited = 1
    while (len(queue) > 0):
        queue = sorted(queue, key=lambda x: cost[x])
        current = queue.pop(0)
        num_visited += 1
        if current == end:
            break
        for neighbor in graph[current]:
            temp = cost[current] + graph[current][neighbor][2]
            if visited[neighbor] == -1:
                visited[neighbor] = current
                cost[neighbor] = temp
                queue.append(neighbor)
            elif temp < cost[neighbor]:
                visited[neighbor] = current
                cost[neighbor] = temp
                queue.append(neighbor)


    path = []
    current = end
    while visited[current] != 0:
        path.append(visited[current])
        time += graph[visited[current]][current][2]
        current = visited[current]
    path.append(current)
    path.reverse()
    return path, time, num_visited
    # End your code (Part 3)


if __name__ == '__main__':
    testcases = ((2270143902, 1079387396), (426882161, 1737223506), (1718165260, 8513026827))
    for testcase in testcases:
        path, time, num_visited = ucs_time(testcase[0], testcase[1])
        print(f'The number of path nodes: {len(path)}')
        print(f'Total second of path: {time}')
        print(f'The number of visited nodes: {num_visited}')