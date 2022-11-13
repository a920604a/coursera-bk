import re

with open("regex_sum_1550557.txt") as file:
    text = file.read()
    number_final = re.findall("[0-9]+", text)
total = sum(list(map(lambda x: int(x), number_final)))
print(total)
