import torch
import numpy as np
import time


def write_conv2d_params(filename, conv):
    with open(filename, "w") as f:
        f.write("conv2d\n")
        wparams = conv.state_dict()['weight']
        f.write(f"{conv.in_channels} ")
        f.write(f"{conv.out_channels} ")
        f.write(f"{conv.kernel_size} ")
        f.write(f"{conv.stride} ")
        f.write(f"{conv.padding} ")
        f.write(f"{conv.dilation} ")
        f.write(f"{conv.groups} ")
        f.write(f"{'1' if conv.bias is not None else '0'} ")
        f.write(f"{conv.padding_mode}\n")

        data = wparams.flatten().numpy()
        for i in range(len(data)):
            f.write(f"{data[i]} ")
        
        if conv.bias is not None:
            bparams = conv.state_dict()['bias']
            data = bparams.flatten().numpy()
            for i in range(len(data)):
                f.write(f"{data[i]} ")

def write_tensor(filename, n):
    with open(filename, "w") as f:
        for i in range(len(n.shape)):
            f.write(f"{n.shape[i]} ")
        f.write(f"\n")
        data = n.flatten().numpy()
        for i in range(len(data)):
            f.write(f"{data[i]} ")
        
if __name__ == '__main__':
    print('Creating conv layer...')
    
    conv2d = torch.nn.Conv2d(
        in_channels=4,
        out_channels=8,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros'
    )

    torch.save({"state_dict": conv2d.state_dict()}, "conv2d.ckpt")
    
    print(conv2d.state_dict())
    print(conv2d.state_dict()['weight'].flatten().numpy())
    #conv2d.load_state_dict(torch.load('conv2d.ckpt')['state_dict'])
    
    # save params to file
    write_conv2d_params("conv_params.txt", conv2d)   

    with torch.no_grad():    
        torch.manual_seed(45)
        x = torch.randn(3, 4, 16, 16)
        print(x)
        print(x[0][0][3][4])
        print(x[0][0][4][3])
        write_tensor('conv_input.txt', x)
        out = conv2d(x)
        #print(out)
        #print(out.shape)
        write_tensor('conv_output.txt', out)
