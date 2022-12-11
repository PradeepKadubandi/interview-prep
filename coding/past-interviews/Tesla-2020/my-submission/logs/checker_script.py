import os
import subprocess

with open('all_outputs.txt') as f:
    lines = f.readlines()
    output = []
    for l in lines:
        if len(l) > 0:
            idx = l.find(':')
            ip = "\"{}\"".format(l[idx+1:-1])
            res = subprocess.run('./checker_osx {}'.format(ip), shell=True, universal_newlines=True, capture_output=True)
            output.append(l)
            output.append(str(res.stdout))
            output.append('\n')

with open('compare_output.txt', 'w') as w:
    w.writelines(output)
        


