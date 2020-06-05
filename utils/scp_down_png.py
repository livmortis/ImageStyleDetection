import os
import sys
import getopt


argv = sys.argv[1:]
print(sys.argv)
optarg, o = getopt.getopt(argv, "i:p")
for opt, arg in optarg:
    if opt == "-i":
        img = arg

os.system('scp xizy@124.207.72.10:~/imgStyleClass/code/util/%s ../../data/testsym/ori/%s'%(arg, arg))