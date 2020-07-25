import xlrd
import numpy as np
from tune_hyper import mlib_data

rootdir = '/home/leondong/proj/motion_study/data/rawdata'
resultFile = f'{rootdir}/results.xlsx'

N = 62
workbook = xlrd.open_workbook(resultFile)
sheet = workbook.sheet_by_index(0)

columns = [0,29,311,151,259,302,433]

data = []
for r in range(N):
	row = [sheet.cell(r,c).value for c in columns]
	data.append(row)

N_col = len(columns)
for j in range(N_col-4,N_col):
	tmp = np.array([data[i][j] for i in range(1,N)])
	thresh = mlib_data.get_thresh(tmp)
	tmp = tmp >= thresh
	tmp = tmp.astype(int)
	data[0].append('{} thresh'.format(data[0][j]))
	for i,elem in enumerate(tmp):
		data[i+1].append(elem)

data = np.array(data)
fh = open('table.csv','w')
for row in data:
	print(','.join(row) ,file=fh)
fh.close()