import getopt, sys
from seqnet import Seqnet

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
            maxlen = list(a)
        else:
            assert False, 'unhandled option'

    seq = Seqnet()
    if corpora != '':    
        seq.read(corpora = corpora, maxlen = maxlen)
    else:
        print('Corpora Is Needed.')
        sys.exit()
    
    seq.compile_model()
    seq.callback()
    seq.train()
    
if __name__ == "__main__":
    main()
