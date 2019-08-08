import os
import json

def get_key(dict, value):
    for k, v in dict.items():
        #val_len = len(v)
        if value in v:
            return k
def write_txt(path, source_name, context_name, context_len, context):
    outfile = open(os.path.join(path, source_name), 'a', encoding='utf-8')
    outfile.write(context_name + '\n')
    outfile.write(context_len + '\n')
    for i in context:
        note = str(i) + "\n"
        outfile.write(note)
    outfile.close()
'''
d = { 'Adam': [[1, 3], [2, 4]], 'Lisa': [[1, 2, 4], [1, 3, 5]], 'Bart': [[5, 9], [1, 2]], 'Paul': [[1, 5], [2, 4]] }
a = [[[1, 3], [2, 4]], [[1, 2, 4], [1, 3, 5]], [[1, 5], [2, 4]]]
Test_path = 'D:\Code\Deep learning\Test'
Pred_path = 'D:\Code'
folder_name = 'Deep learning'
file_name = 'Test'
path = os.path.join(Pred_path, folder_name+'\\'+file_name)
context_len = '10'
context_name = 'WWXXZZ.jpg' + '\\' + 'dsaf'
#a = context_name.index('.jpg')
a = context_name.replace('.jpg', '.txt')

write_txt(path, "2.txt", context_name, context_len, a)
'''

def read_json(path, source_name):
    with open(os.path.join(path, source_name), 'r', encoding='utf-8') as sourcefile:
        source = json.load(sourcefile)
    return source
Test_path = 'D:\Code\Deep learning\Test'
id_dict = read_json(Test_path, 'id_dict.json')
print((id_dict["0"]))
