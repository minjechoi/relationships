import urllib.request

print("Downloading the pretrained model")
url='https://www.dropbox.com/s/xuukpqqtuxqzhjj/full-model.pth?dl=1'
u = urllib.request.urlopen(url)
data = u.read()
u.close()

with open('data/full-model.pth', "wb") as f:
    f.write(data)
