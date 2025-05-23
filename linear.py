import torch
import numpy as np
import time


def write_linear_params(filename, ln):
    with open(filename, "w") as f:
        f.write("linear\n")
        wparams = ln.state_dict()['weight']
        f.write(f"{wparams.shape[0]} {wparams.shape[1]}\n")
        data = wparams.numpy()
        for i in range(len(data)):
            for w in range(len(data[i])):
                f.write(f"{data[i][w]} ")
        
        if ln.bias is not None:
            bparams = ln.state_dict()['bias']
            data = bparams.numpy()
            for i in range(len(data)):
                f.write(f"{data[i]} ")

if __name__ == '__main__':
    print('Creating linear layer...')
    nin = 100
    nout = 1000
    ln = torch.nn.Linear(nin, nout)
    torch.save({"linear_state_dict": ln.state_dict()}, "linear.ckpt")
    torch.manual_seed(42)
     
    # ln.load_state_dict(torch.load('linear.ckpt')['linear_state_dict'])
    
    # save params to file
    write_linear_params("linear_params.txt", ln)   

    with torch.no_grad():    
        x = torch.ones(1, nin)
        start = time.monotonic_ns()
        out = ln(x)
        elapsed = time.monotonic_ns() - start
        numpy_out = out.numpy()[0]
        with open("linear_out.txt", "w") as outfile:
            outfile.write(f"{len(numpy_out)}\n")
            for i in range(len(numpy_out)):
                outfile.write(f"{numpy_out[i] }")

        print(out)
        print(out.sum())
        print(f"time ns: {elapsed}")
    

