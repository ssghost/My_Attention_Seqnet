import getopt, sys
from seqnet import *

def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'c:m:', ['corpora=','maxlen='])
    except getopt.GetoptError as err:
        print(err) 
        sys.exit()

    corpora = ''
    maxlen = [0,0]

    for o, a in opts:
        if o in ('-c', '--corpora') and type(a)==str:
            corpora = a
        elif o in ('-m', '--maxlen') and type(a)==list[int]:
            maxlen = a
        else:
            assert False, 'unhandled option'

    if corpora != '':    
        seqnet().read(corpora = corpora, maxlen = maxlen)
    else:
        print('Corpora Is Needed.')
        sys.exit()
    
    seqnet().compile_model()
    seqnet().train()
    
if __name__ == "__main__":
    main()