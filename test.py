import getopt, sys
from seqnet import Seqnet

def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], 't:o:l:', ['text=','outpath=','modelpath='])
    except getopt.GetoptError as err:
        print(err) 
        sys.exit()

    tpath, opath, loadpath = '','',''
    
    for o, a in opts:
        if o in ('-t', '--text') and type(a)==str:
            tpath = a
        elif o in ('-o', '--outpath') and type(a)==str:
            opath = a 
        elif o in ('-l', '--modelpath') and type(a)==str:
            loadpath = a
        else:
            assert False, 'unhandled option'    
    
    seq = Seqnet()
    if loadpath != None:
        seq.load_model(loadpath)
        
    seq.test(tpath=tpath,opath=opath)
    
if __name__ == "__main__":
    main()
