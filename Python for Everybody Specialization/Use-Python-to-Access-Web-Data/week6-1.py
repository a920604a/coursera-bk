import json
import urllib.request, urllib.parse, urllib.error

url = "http://py4e-data.dr-chuck.net/comments_1550562.json"

data = urllib.request.urlopen(url).read()


info = json.loads(data)
total = 0
for item in info["comments"]:
      total+=item['count']
print(total)

