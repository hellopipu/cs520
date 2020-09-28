class Queue(object):
    '''
    queque structure
    '''
    def __init__(self):
        self.queque = []
    def add(self,data):
        self.queque.append(data)
    def pop(self):
        return self.queque.pop(0)
    def getpeek(self):
        return self.queque[0]
    def __len__(self):
        return len(self.queque)
class Prior_Queue(Queue):
    '''
    queque structure
    '''
    def add(self,data):
        '''
        add in order of 'prior'
        :param data:
        :return:
        '''
        length = len(self.queque)
        if length==0:
            self.queque.append(data)
        else:
            for i in range(length):
                if data['prior']<self.queque['prior']:
                    self.queque.insert(i,data)
                    break
                else:
                    if i==length-1:
                        self.queque.append(data)
                    else:
                        continue

queue = Prior_Queue()
start_point=[0,0]
dim=5
queue.add({'pos':start_point,'dist':0,'heur':2*(dim-1), 'prior':2*(dim-1)})
print(queue.queque)