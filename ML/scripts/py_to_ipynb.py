import IPython.nbformat.current as nbf
py_name = 'py_to_ipynb'
nb_name = 'py_to_ipynb'
# notebooks\multi_label.ipynb
nb = nbf.read(open(f'{py_name}.py', 'r'), 'py')
nbf.write(nb, open(f'notebooks/{nb_name}.ipynb', 'w'), 'ipynb')