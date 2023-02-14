import numpy as np
import argparse
import os
parser = argparse.ArgumentParser()
# Experiment
parser.add_argument("--path", default="./")
args = parser.parse_args()
 
for root, dirs, files in os.walk(args.path, topdown=False):
	for name in files:
		ret = []
		if 'log' in name:
			with open(os.path.join(root, name),'r') as f:
				for line in f:
					if 'evaluation/Average Returns' in line:
						data = line.split()
						ret.append(float(data[2]))

		np.save(os.path.join(root, 'ret.npy'), ret)

