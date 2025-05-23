With pytorch generate a neural network and saves it's parameters:\
```python linear.py```

Then load these parameters generate an 'executable' that will perform the linear layer operation:\
```gcc linear.c -o linear && ./linear```

Build the generated executable function (might take a bit of time):\
```./build_linear_exec.sh```

Run the executable providing an output file to compare:\
```./execute_nn linear_out.txt```
