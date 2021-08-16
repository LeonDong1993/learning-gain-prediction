# coding:utf-8
import sys,os,pickle,time,warnings
from pdb import set_trace

def read_text(fpath, splitter = ',', header = False, encoding = 'utf-8'):
	fh = open(fpath, 'r', encoding = encoding)
	content = fh.readlines()
	fh.close()

	for i in range(len(content)):
		line = content[i]
		content[i] = line.strip().split(splitter)

	if header:
		content = content[1:]

	return content

############## AUX ###############
def _merge_item(x,y):
	nx = x
	ny = y
	if not isinstance(x,tuple):
		nx = tuple([x])
	if not isinstance(y,tuple):
		ny = tuple([y])
	return nx+ny


############## SHELL ##################
def cd(directoty):
	os.chdir(directoty)
	return

def bash(command,verbose = True):
	if verbose:
		print(command)
	ret = os.system(command)
	if (ret != 0):
		print('Command [%s] execute failed' % command)
		sys.exit(1)

def makedir(path):
	if (not os.path.exists(path)):
		print('Making directory %s' % path)
		os.makedirs(path)
	else:
		print('Directory already exist!')


############# UTIL ##################
def reqver(a,b,flag):
	assert(flag in ['eq','le','ge'])
	pyver = sys.version_info
	if flag == 'eq':
		assert(pyver >= (a,b-1))
		assert(pyver <= (a,b+1))
	if flag == 'ge':
		assert(pyver >= (a,b-1))
	if flag == 'le':
		assert(pyver <= (a,b+1))
	return

def allin(x,y):
	ret = sum(map(lambda v:v in y,x)) == len(x)
	return ret

def notin(x,y):
	ret = list(filter(lambda v:v not in y,x))
	return ret

def pickle_dump(obj,fpath):
	pickle.dump( obj, open( fpath, "wb" ) , protocol = 2)
	return True

def pickle_load(fpath):
	try:
		return pickle.load( open( fpath, "rb" ) )
	except Exception as e:
		print("ERROR: {}".format(e))
		sys.exit(0)

def progress(cur,total,msg='Running',fd = 2, pid = False):
	assert( fd in [1,2])
	sysout = [sys.stdout, sys.stderr]
	out = sysout[fd-1]

	if pid:
		myid = os.getpid()
		msg = 'Process-{} {}'.format(myid,msg)

	percent = 100*cur/float(total)
	if percent < 100:
		end_char = '\r'
	else:
		end_char = '\n'
	out.write('[{}:{}/{} {:.2f}%] {}'.format(msg,cur,total,percent,end_char))
	out.flush()

def crossprod(X,Y):
	for x in X:
		for y in Y:
			yield _merge_item(x,y)

# The elements are always in the tuple form (iterable)
# even if the input only has one array
def product(args):
	D = args[0]
	if len(args) == 1:
		D = list(map(lambda x:(x,),D))
	else:
		for item in args[1:]:
			D = crossprod(D,item)
	return D

def halfprod(X):
	ret=[] ; N = len(X)
	for i in range(N):
		for j in range(i+1,N):
			ret.append( _merge_item(X[i],X[j]) )
	return ret

def diffprod(X):
	ret=[] ; N = len(X)
	for i in range(N):
		for j in range(N):
			if i!=j:
				ret.append( _merge_item(X[i],X[j]) )
	return ret

def user_warn(msg):
	warnings.warn(msg)
	# set_trace()