import telegram
import json
import os

class Bot:

    def __init__(self, name, chat_id = None):
        home = os.path.expanduser('~')
        if os.name == 'nt':
            file_directory = home + '\\.telegramBots\\{}.json'
        else:
            file_directory = home + '/.telegramBots/{}.json'
        file_directory = file_directory.format(name)
        json_data=open(file_directory, 'r').read()
        data = json.loads(json_data)

        self.token = data['token']
        self.chat_id = chat_id if chat_id else data['chat_id']
        self.bot = telegram.Bot(token=self.token)


    def sendMessage(self, message = [], imgPath = None, filePath = None):
        if isinstance(message, basestring):
            message = [message]

        for m in message:
            self.bot.sendMessage(self.chat_id, text=m)

        if imgPath:
            self.bot.send_photo(self.chat_id, photo=open(os.path.expanduser(imgPath), 'rb'))

        if filePath:
            self.bot.send_document(self.chat_id, document=open(os.path.expanduser(filePath), 'rb'))
