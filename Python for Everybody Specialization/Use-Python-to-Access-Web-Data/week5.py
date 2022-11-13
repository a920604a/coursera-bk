import urllib.request, urllib.parse, urllib.error
import xml.etree.ElementTree as ET
import ssl

url = "http://py4e-data.dr-chuck.net/comments_1550561.xml"


# Ignore SSL certificate errors
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


xml = urllib.request.urlopen(url).read()
tree = ET.fromstring(xml)
counts = tree.findall(".//count")
total = 0
for count in counts:
    total += int(count.text)
print(total)
