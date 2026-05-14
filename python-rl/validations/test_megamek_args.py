import subprocess
import os

cwd = "/home/stuart_hatzioannou/repos/megamek/megamek"
cmd = [
    "build/install/MegaMek/bin/megamek",
    "-rlexport",
    "-autogen",
    "-randomMap",
    "-p1meks", "3075/Pariah Prime.mtf",
    "-p2meks", "Rec Guides ilClan/Vol 33/Gyrfalcon 5.mtf"
]

# We modify the script locally to print arguments just before Java is called? No, we can just look at PrincessOfflineTrainer logs!
