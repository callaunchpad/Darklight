import os
import subprocess

with open("./dataset/Sony_val_list.txt") as f:
    content = f.readlines()

content = [x.strip() for x in content]
shorts = ["./dataset" + x.split(" ")[0][1:] for x in content]
longs = ["./dataset" + x.split(" ")[1][1:] for x in content]
print(longs)

for long in longs:
    #command = "mv " + short + " ./dataset/Sony_val/short/"
    subprocess.run(["mv", long, "./dataset/Sony_val/long/"])
