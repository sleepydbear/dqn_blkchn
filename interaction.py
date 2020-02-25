from collections import OrderedDict


class Interaction():

    def __init__(self, sender, message, metadata, timestamp, signature):
        self.sender = sender
        self.message = message
        self.metadata = metadata
        self.signature = signature
        self.timestamp = timestamp

    def to_ordered_dict(self):
        return OrderedDict([('sender', self.sender), ('message', self.message), ('metadata', self.metadata),('timestamp', self.timestamp)])



    def __repr__(self):
        return str(self.__dict__)