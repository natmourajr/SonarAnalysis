import telegram
import json
import os
from mainConfig import CONFIG


class Bot:
    def __init__(self, name, chat_id=None):
        home = CONFIG['homePath']
        file_directory = os.path.join(home, '.telegramBots', '{}.json'.format(name))
        json_data = open(file_directory, 'r').read()
        data = json.loads(json_data)

        self.token = data['token']
        self.chat_id = chat_id if chat_id else data['chat_id']
        self.bot = telegram.Bot(token=self.token)

    def sendMessage(self, message=[], imgPath=None, filePath=None):
        message = [message]

        for m in message:
            self.bot.sendMessage(self.chat_id, text=m)

        if imgPath:
            self.bot.send_photo(self.chat_id, photo=open(os.path.expanduser(imgPath), 'rb'))

        if filePath:
            self.bot.send_document(self.chat_id, document=open(os.path.expanduser(filePath), 'rb'))
