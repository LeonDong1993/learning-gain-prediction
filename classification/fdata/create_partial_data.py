from pyLeon import utils
from pdb import set_trace

def main(args):
    infile = args[0]
    outfile = args[1]
    drops = eval('[{}]'.format(args[2]))

    data = utils.pickle_load(infile)
    for k, v  in data.items():
        if k == 'info':
            continue

        (X,Y,T,S) = v
        (l,s,f) = k
        assert(X.shape[1] == f*6)
        
        selector = []
        for i in range(6):
            if i not in drops:
                selector += list(range(i*f,(i+1)*f))

        selector = tuple(selector)
        # set_trace()
        X = X[:,selector]
        data[k] = (X,Y,T,S)

    utils.pickle_dump(data,outfile)


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
