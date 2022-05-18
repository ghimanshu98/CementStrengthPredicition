import sqlite3
import os
from Logger.logger import Logger
import csv
import pandas as pd

class DbOperations:
    def __init__(self, dbOps_log_file_path, db_path, tableName):
        
        # logger object and log file location
        self.log_agent = Logger()
        self.dbOps_log_file_path = dbOps_log_file_path
        
        # database
        self.database_path = db_path

        # path of dir containing files ready for db ingestion
        self.db_ready_file_path = 'Db_Ingestion_Ready_Files'

        # path to write files after fetching from DB
        self.final_training_file_path = 'Final_Training_CSV_File'

        # table name
        self.table_name = tableName

    def setUpConnection(self):
        try:
            conn = sqlite3.connect(self.database_path)

            log_file = open(self.dbOps_log_file_path, 'a+')
            # logging after successful connection
            if conn != None:
                message = "DbConnection to "+self.database_path+" established successfully."
                self.log_agent.log(log_file, message)
                log_file.close()
                return conn
            else:
                raise Exception('Error while opening connection to database {}'.format(self.database_path))
        except Exception as e:
            log_file = open(self.dbOps_log_file_path, 'a+')
            # logging after successful connection
            if conn != None:
                message = "DbConnection to "+self.database_path+" is un-successfull. "+ str(e)
                self.log_agent.log(log_file, message)
                log_file.close()

    def createTableInDb(self, column_names):
        """
        :Description: Used to create table in passed database.
        :param columns: a dictionary of columns
        """
        conn = self.setUpConnection()
        if conn != None:
            try:
                log_file = open(self.dbOps_log_file_path, 'a+')
                
                cursor = conn.cursor() # obtaining the cursor from connection
                
                # logging the change
                message = "Cursor obtained successfully for db "+self.database_path + " Moving to Create Table"
                self.log_agent.log(log_file, message)

                # Checking if table exists or not
                statement = "SELECT count(name) FROM sqlite_master WHERE type = 'table' AND name = '"+self.table_name+"'"
                # cursor.execute("select count(name) from sqlite_master where type = 'table' and name = {}".format(self.table_name))
                cursor.execute(statement)

                if cursor.fetchone()[0] == 1: # if true then table exists
                    # closing the db conn
                    conn.close()
                    message = "Table {} already exists.".format(self.table_name)
                    self.log_agent.log(log_file, message)
                else:
                    for key in column_names.keys():
                        type = column_names[key]

                        # in try block we check if table already exists, if yes then and alter table and add columns in it
                        try:
                            statement = 'alter table {table_name} add column "{column_name}" {datatype}'.format(table_name = self.table_name, column_name = key, datatype = type)

                            conn.execute(statement)
                        except:
                            # in except block we are creating table if table do not exist
                            statement = 'create table {table_name} ("{column_name}" {data_type})'.format(table_name = self.table_name, column_name= key, data_type = type)

                            conn.execute(statement)

                    message = "Table {} created successfully".format(self.table_name)
                    self.log_agent.log(log_file, message)
                    log_file.close()
                    conn.close()

            except Exception as e:
                log_file = open(self.dbOps_log_file_path, 'a+')
                message = "Error ocurred while creating Table "+str(e)
                self.log_agent.log(log_file, message)
                log_file.close()                

    def insertFilesInDB(self):
        try:
            log_file = open(self.dbOps_log_file_path, 'a+')
            self.log_agent.log(log_file, "Strating csv files insertion to db.")
            
            filenames = os.listdir(self.db_ready_file_path)

            # open connection to db
            conn = self.setUpConnection()

            for file in filenames:
                count = 1
                try:
                    with open(self.db_ready_file_path+'/'+file, 'r') as f:   # opening the file
                        next(f)  # outputs the row in file (here we are ignoring the header using next)
                        csv_reader_obj = csv.reader(f, delimiter='\n')  # creating a csv reader object
                        for row in enumerate(csv_reader_obj): # gives a number to row
                            for value in row[1]: # (0, [row values])
                                statement = 'insert into "{table_name}" values ({values})'.format(table_name = self.table_name, values = (value))
                                count = row[0]
                                conn.execute(statement)
                        conn.commit()
                        message = "{} records loaded successfully in db {} from file {}".format(count, self.database_path, file)
                        self.log_agent.log(log_file,message)
                        log_file.close()
                except Exception as e:
                    conn.rollback()
                    message = "Error in loading file to db "+str(e)
                    self.log_agent.log(log_file, message)
                    log_file.close()
                    conn.close()

        except Exception as e:
            log_file = open(self.dbOps_log_file_path, 'a+')
            message = "Error in loading file to db "+str(e)
            self.log_agent.log(log_file, message)
            log_file.close()



    def getRowsFromDb(self):
        try:
            log_file = open(self.dbOps_log_file_path, 'a+')
            conn = self.setUpConnection()
            if conn != None:
                message = "Starting record extraction from database"
                self.log_agent.log(log_file, message)

                # obtaining header
                col_names = self.getTableFields()

                # obtain the records from table
                if col_names != None:
                    cursor = conn.cursor()
                    statement = 'select * from {}'.format(self.table_name)
                    cursor.execute(statement)
                    records = cursor.fetchall()
                    conn.close()

                    if len(records) != 0:
                        df = pd.DataFrame(records, columns= col_names)
                        df.to_csv(self.final_training_file_path+"/ready_to_train_file.csv", index= False)
                        message = "{}  records retrieved from table {} in {}".format(len(records), self.table_name, self.database_path)
                        self.log_agent.log(log_file, message)
                        log_file.close()
                    else:
                        raise Exception('Obtained 0 records from table {} in {}'.format(self.table_name, self.database_path))
                else:
                    raise Exception('Obtained None type in col_names ')
        except Exception as e:
            message = "Error occurred while fetching records from table. "+str(e)
            self.log_agent.log(log_file, message)
            log_file.close()


                
    def getTableFields(self):
        try:
            log_file = open(self.dbOps_log_file_path, 'a+')
            conn = self.setUpConnection()
            if conn != None:                
                # obtaining cursor
                cursor = conn.cursor()
                statement = 'PRAGMA table_info("{}")'.format(self.table_name)
                cursor.execute(statement)
                col_name = []
                for info in cursor.fetchall():  # cursor.fetchall - gives a list of tuple
                    col_name.append(info[1])  # column name is at position 1 in each tuple
                conn.close()
                if len(col_name) != 0:
                    message = "Following column names {} obtained successfully from table {} in {}".format(str(col_name), self.table_name, self.database_path)  
                    self.log_agent.log(log_file, message)
                    log_file.close()
                    return col_name
                else:
                    raise Exception('No cols found in table {} in '.format(self.table_name, self.database_path))
        except  Exception as e:
            message = "Error occurred while obtaining column names "+str(e)
            self.log_agent.log(log_file, message)
            log_file.close()
            return None

        
        
                




                






                    





            

