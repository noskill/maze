class Node(object):
    def __init__(self):
        self.data = None
        self.list = []

    def append(self, data):
        self.list.append(data)

    def index(self, data):
        return self.list.index(data)

    def __contains__(self, data):
        return data in self.list

    def pop(self, index):
        return self.list.pop(index)

class Map():
    def __init__(self, xdim, ydim):
        self.dimentions = xdim, ydim
        self.graph = []
        for i in range(self.dimentions[0]):
            self.graph.append([Node() for x in range(self.dimentions[1])])
 
    def remove_edge(self, (x1, y1), (x2, y2)):
       self.graph[x1][y1].pop(self.graph[x1][y1].index((x2,y2)))
       self.graph[x2][y2].pop(self.graph[x2][y2].index((x1,y1)))

    def add_edge(self, (x1, y1), (x2, y2)):
        if not self.is_connected((x1, y1), (x2, y2)):
            self.graph[x1][y1].append((x2, y2))
            self.graph[x2][y2].append((x1, y1))

    def is_connected(self, (x1, y1), (x2, y2)):
       return (x2, y2) in self.graph[x1][y1]

    def __getitem__(self, (x1, y1)):
       return self.graph[x1][y1].data

    def __setitem__(self, (x1, y1), data):
       self.graph[x1][y1].data = data
   
    def __str__(self):
        result = ['' for i in range(self.dimentions[0])]
        for i in range(self.dimentions[0]):
            for j in range(self.dimentions[1]):
                result[i] += '| {} '.format(self.graph[i][j].data)
            result[i] += '|'
            result[i] += '\n' + (' - ' * self.dimentions[1]) + '\n'
        return ''.join(result)
