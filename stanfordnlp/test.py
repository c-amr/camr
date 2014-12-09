import os


print os.path.relpath(__file__)
print os.path.exists('data.py')

print open('data.py','r').read()
print open('data.py','r').readlines()
