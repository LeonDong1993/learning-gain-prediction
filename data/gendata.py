# coding: utf-8
import xlrd
import numpy as np
import datetime
from glob import glob
from pyLeon import utils
from pdb import set_trace

rootdir = '/home/lab/data/HailiangDong/motion_study/data/rawdata'
resultFile = f'{rootdir}/results.xlsx'

def convert_timestr(timestr):
	postfix = timestr[-2:]
	if postfix in ["AM", "PM"]:
		obj = datetime.datetime.strptime(timestr,"%m/%d/%Y %I:%M:%S %p")
	else:
		obj = datetime.datetime.strptime(timestr,"%Y-%m-%d %H:%M:%S.%f")
	return obj

def load_user_motion(uid,mov_type,columns = 'selected'):
	motion_dir = '{}/mov_{}'.format(rootdir,mov_type)
	candidates = glob('{}/{}_*.csv'.format(motion_dir,uid))
	if len(candidates) < 1:
		print('User {} does not have motion data'.format(uid))
		return None
	assert(len(candidates) == 1)
	movfile = candidates[0]
	return read_motion_file(movfile,columns)

def read_motion_file(fpath,columns):
	fh = open(fpath)
	content = fh.readlines()
	fh.close()
	ret = []
	for line in content[1:]: # remove header
		info = line.strip().split(',')
		ret.append(info)
	if columns == 'selected':
		data = np.array(ret)
		# parse the time
		strtime = data[:,0]
		timearr = np.array([convert_timestr(elem) for elem in strtime])
		timearr = timearr.reshape(-1,1)
		# remove the time column
		selector = feature_selector[1:]
		data = data[:,selector]
		data = data.astype(float)
		# eliminate invalid row data
		valid = 0
		EPS = 1e-20
		while 1:
			row = data[valid,:]
			if np.all(np.abs(row)> EPS):
				break
			valid += 1
		if valid != 0:
			print(f'Eliminated {valid} rows in {fpath}')
		data = data[valid:,:]
		timearr = timearr[valid:,:]
		data = np.hstack([timearr,data]) # dtype is object
	else:
		ret.insert(0, content[0].strip().split(',')) # resume header
		data = np.array(ret)
		# selector = feature_selector
		# data = data[:,selector]
	return data

def old_main():
	global feature_selector
	# feature_selector = tuple(range(15)) + tuple(range(16,23)) # all
	feature_selector = tuple(range(11)) + tuple(range(16,19)) # head and hand position
	# feature_selector = tuple(range(8)) # only the head
	motion_type = 'learn'
	score_col = 259 # 259-learn_score, 151-learn_srate, 302-prac_score, 433-prac_srate
	score_format = int # score format function
	out_file = 'learn_score_{}.pkl'.format(len(feature_selector))
	print('Output filename is {}'.format(out_file))

	N = 62 # number of rows
	workbook = xlrd.open_workbook(resultFile)
	sheet = workbook.sheet_by_index(0)

	# retrieve the uid and score from excel
	uids = [sheet.cell(i,0).value for i in range(1,N)]
	scores = [score_format(sheet.cell(i,score_col).value) for i in range(1,N)]

	dataset = []
	for i,x in enumerate(uids):
		utils.progress(i+1,N)
		movdata = load_user_motion(x,motion_type)
		if movdata is not None:
			dataset.append( (x,movdata,scores[i]) )

	utils.pickle_dump(dataset,out_file)
	# set_trace()

def main(args):
	score_col = 259 # 259-learn_score, 151-learn_srate, 302-prac_score, 433-prac_srate
	score_format = int # score format function
	# ==============================================
	rootdir = args[0]
	mov_type = args[1]
	out_file = args[2]

	resultFile = f'{rootdir}/results.xlsx'
	N = 62 # number of rows
	workbook = xlrd.open_workbook(resultFile)
	sheet = workbook.sheet_by_index(0)
	uids = [sheet.cell(i,0).value for i in range(1,N)]
	scores = [score_format(sheet.cell(i,score_col).value) for i in range(1,N)]

	selector =  tuple(range(14)) + tuple(range(15,22)) # head pos + orientation, lhand pos + orientation, skip (lbutton), rhand pos + orientation

	dataset = []
	for i,u in enumerate(uids):
		utils.progress(i+1,N)
		motion_dir = '{}/mov_{}'.format(rootdir,mov_type)
		# find the filename of that user
		candidates = glob( '{}/{}_*.csv'.format(motion_dir,u) )
		if len(candidates) < 1:
			print('User {} does not have motion data'.format(uid))
		else:
			assert(len(candidates) == 1)
			mov = utils.read_text(candidates[0],header = True)
			data = np.array(mov)
			data = data[:,selector]
			data = data.astype(float)

			# invalid data remove
			valid = 0
			EPS = 1e-20
			while 1:
				row = data[valid,:]
				if np.all(np.abs(row)> EPS):
					break
				valid += 1
			if valid != 0:
				print('Eliminated {} rows in {}'.format(valid,candidates[0]))
			data = data[valid:,:]


			dataset.append( (u,data,scores[i]) )

	utils.pickle_dump(dataset,out_file)
	# set_trace()

if __name__ == '__main__':
	import sys
	main(sys.argv[1:])
	# old_main()
