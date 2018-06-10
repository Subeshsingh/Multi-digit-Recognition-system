import os
print(os.path.join('input/','abc.txt'))
file_name_input=abc.txt
print(os.path.join(os.path.dirname(os.path.realpath(__file__)),file_name_input))