def linear_algebra():
	import numpy as np 
	import pandas as pd import os
	for dirname, _, filenames in os.walk('/kaggle/input'):
	    for filename in filenames:
	        print(os.path.join(dirname, filename))# Any results you write to the current directory are saved as output.
