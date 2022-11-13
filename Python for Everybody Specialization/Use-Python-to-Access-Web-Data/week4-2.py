from urllib.request import urlopen
from bs4 import BeautifulSoup
import ssl

# Ignore SSL certificate errors
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

# url = input('Enter - ')
url = "http://py4e-data.dr-chuck.net/known_by_Kamil.html"
# url = 'http://py4e-data.dr-chuck.net/known_by_Fikret.html'
# count = int(input('Enter count: '))
count = 7
# position = int(input('Enter position: '))
position = 18 - 1
# Retrieve all of the anchor tags


def get_html(url):
    html = urlopen(url).read()
    soup = BeautifulSoup(html, "html.parser")
    return soup


urllist = []
for _ in range(count):
    taglist = list()

    for tag in get_html(url)("a"):  # Needed to update your variable to new url html
        taglist.append(tag)
    url = taglist[position].get(
        "href", None
    )  # You grabbed url but never updated your tags variable.

    print("Retrieving: ", url)
    urllist.append(url)


print(urllist[-1])