import pandas as pd
import numpy as np
import random
import mysql.connector
from mysql.connector import Error


class MySQL:
    def __init__(self, host_name, user_name, user_password, databasename):
        self.databasevariables = {
            "addweightstable": """
             CREATE TABLE weightsvalues(
             RowID varchar(5),
             ColID varchar(5),
             Val varchar(5),
             Neuron int 
             );
             """,
                               "addbiastable": """
         CREATE TABLE biasvalues(
         Neuron int,
         Bias int 
         );
         """
        }
        self.serverconnection = self.create_server_connection(host_name, user_name, user_password)
        self.dbconnection = self.create_db_connection(host_name, user_name, user_password, databasename)

    def create_server_connection(self, host_name, user_name, user_password):
        connection = None
        try:
            connection = mysql.connector.connect(
                host=host_name,
                user=user_name,
                passwd=user_password
            )
            print("MySQL Database connection successful")
        except Error as err:
            print(f"Error: '{err}'")

        return connection

    def create_database(self, connection):
        cursor = connection.cursor()
        try:
            cursor.execute("CREATE DATABASE weights")
            print("Database created successfully")
        except Error as err:
            print(f"Error: '{err}'")

    def create_db_connection(self, host_name, user_name, user_password, db_name):
        connection = None
        try:
            connection = mysql.connector.connect(
                host=host_name,
                user=user_name,
                passwd=user_password,
                database=db_name
            )
            print("MySQL Database connection successful")
        except Error as err:
            print(f"Error: '{err}'")

        return connection

    def execute_query(self, query):
        cursor = self.dbconnection.cursor()
        try:
            cursor.execute(query)
            self.dbconnection.commit()
            #print("Query successful")
        except Error as err:
            print(f"Error: '{err}'")

    def insert_data(self, row, col, val, group):
        temp = """
                    INSERT INTO weightsvalues(RowID, ColID, Val, Neuron)
                    VALUES(""" + str(row) + ", " + str(col) + ", " + str(val) + ", " + str(group) + ")"
        return temp
    def insert_bias(self, neuron, bias):
        temp = """
                    INSERT INTO biasvalues(Neuron, Bias)
                    VALUES(""" + str(neuron) + ", " + str(bias) + ")"
        self.execute_query(temp)
    def read_query(self, query):
        cursor = self.dbconnection.cursor()
        result = None
        try:
            cursor.execute(query)
            result = cursor.fetchall()
            return result
        except Error as err:
            print(f"Error: '{err}'")

    def getData(self, row, col, neuron):
        data = """ SELECT val from weightsvalues WHERE rowid =""" + str(row) + """ AND colid=""" + str(col) + """ AND neuron=""" + str(neuron)
        return float(self.read_query(data)[0][0])
    def getBias(self, neuron):
        data = """ SELECT bias from biasvalues WHERE neuron=""" + str(neuron)
        return float(self.read_query(data)[0][0])

    def updateElement(self, row, col, neuron, newnum):
        data = """UPDATE weightsvalues SET val = """ + str(newnum) + """  WHERE rowid =""" + str(row) + """ AND colid=""" + str(col) + """ AND neuron=""" + str(neuron)
        self.execute_query(data)
    def updateBias(self, neuron, newnum):
        data = """UPDATE biasvalues SET Bias = """ + str(newnum) + """  WHERE neuron=""" + str(neuron)
        self.execute_query(data)