import getopt, sys
from seqnet import *

def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], 't:o:l:', ['text=','outdir=','loadmodel='])
    except getopt.GetoptError as err:
        print(err) 
        sys.exit()

    tpath, opath, loadpath = '','',''
    
    for o, a in opts:
        if o in ('-t', '--text') and type(a)==str:
            tpath = a
        elif o in ('-o', '--outdir') and type(a)==str:
            opath = a 
        elif o in ('-l', '--loadmodel') and type(a)==str:
            loadpath = a
        else:
            assert False, 'unhandled option'    
    
    if loadpath != None:
        seqnet().load_model(loadpath)
        
    seqnet().test(tpath=tpath,opath=opath)
    
if __name__ == "__main__":
    main()