from datetime import datetime

# Logger Class
class Logger:
    # def __init__(self):
        # pass

    def log(self, file_obj, message):
        """
        Used to Log messages/codes as per requirement in the particular file passed in file_obj
        :param file_obj: File to log messages in.
        :param message: Message to log
        """
        # obtaining current datetime in local machine
        now = datetime.now()
        # using the above instance get date and time
        date = now.date()
        time = now.strftime("%H:%M:%S") # time in string format
        file_obj.write(str(date) + '/' + time + '\t\t' + message +'\n')