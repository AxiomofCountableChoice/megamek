import subprocess
import os
import argparse

cwd = "/home/stuart_hatzioannou/repos/megamek/megamek"
cmd = [
    "build/install/MegaMek/bin/megamek",
    "-rlexport",
    "-autogen"
]

parser = argparse.ArgumentParser()
parser.add_argument("--scenario", type=str, default="", help="Path to .mms scenario file")
parser.add_argument("--options", type=str, nargs="*", default=[], help="List of game options like -VICTORY_USE_KILL_COUNT=true")
args = parser.parse_args()

if args.scenario:
    cmd.extend(["-scenario", args.scenario])
else:
    cmd.extend([
        "-randomMap",
        "-p1meks", "3075/Pariah Prime.mtf",
        "-p2meks", "Rec Guides ilClan/Vol 33/Gyrfalcon 5.mtf"
    ])

for opt in args.options:
    cmd.append(opt)

proc = subprocess.Popen(cmd, cwd=cwd)
print(f"Started MegaMek with PID {proc.pid}")
