import gc


class Node:

    def __init__(self, data=None, next_=None, pre_=None):
        self.data = data
        self.next_ = next_
        self.pre_ = pre_

    def __repr__(self):
        return str(self.data)


class DoublyLinkedList:

    def __init__(self, head=None):
        self.head = head
        self.length = 0

    def is_empty(self):
        return self.length == 0

    def append(self, data):
        if isinstance(data, Node):
            item = data
        else:
            item = Node(data)

        if not self.head:
            item.pre_ = item
            item.next_ = item
            self.head = item

        else:

            node = self.head.pre_
            item.pre_ = node
            item.next_ = node.next_
            self.head.pre_ = item
            node.next_ = item

        self.length += 1

    def delete(self, index):
        if self.length == 0:
            print("this chain is empty.")
            return

        if index < 0 or index >= self.length:
            print('error: out of index')
            return

        node = self.head

        if index == 0:
            self.head = node.next_

        else:
            counter = 0
            while counter < index:
                node = node.next_
                counter += 1

        node.next_.pre_ = node.pre_
        node.pre_.next_ = node.next_

        self.length -= 1
        del node
        gc.collect()

    def update(self, index, data):
        if self.length == 0 or index < 0 or index >= self.length:
            print('error: out of index')
            return
        counter = 0
        node = self.head
        while counter < index:
            node = node.next_

        node.data = data

    def get_item(self, index):
        if self.length == 0 or index < 0 or index >= self.length:
            print("error: out of index")
            return
        counter = 0
        node = self.head
        while counter < index:
            node = node.next_
            counter += 1

        return node

    def get_index(self, data):
        if self.length == 0:
            print("this chain  is empty")
            return

        node = self.head

        for i in range(self.length):
            if node == data:
                return True, i
            else:
                node = node.next_
        print("not found")
        return False, None

    def insert(self, index, data, front_insertion=False):
        if self.length == 0:
            print("this chain is empty")
            return

        if index < 0 or index >= self.length:
            print("error: out of index")
            return

        if isinstance(data, Node):
            item = data
        else:
            item = Node(data)

        counter = 0
        if front_insertion:
            counter += 1
        node = self.head
        while counter < index:
            node = node.next_
            counter += 1
        item.pre_ = node
        item.next_ = node.next_
        node.next_.pre_ = item
        node.next_ = item
        self.length += 1

    def clear(self):
        self.head = None
        self.length = 0
