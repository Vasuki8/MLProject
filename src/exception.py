import sys
import logging
import time

class Custom_Exception(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(self,error_message)
        self.error_detail = error_detail
        self.error_message = error_message
        self.test =  self.error_message_detail() 

    def error_message_detail(self):
        _,_,exc_tb = self.error_detail.exc_info()
        self.file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        error_message = f"Error occured in python script name [{self.file_name}] line number [{line_number}] error message [{self.error_message}]"
        return error_message  
    
    def __str__(self):
        return self.test



