import chardet

# 自动检测编码
def read_file(filename):
    with open(filename, 'r',encoding="gbk") as f:
        raw_data = f.read()
    detected = chardet.detect(raw_data)
    encoding = detected['encoding']
    with open("呐喊_utf-8", 'w',encoding="utf-8") as f1:
        f1.write(raw_data)
file_content = read_file('呐喊.txt')
