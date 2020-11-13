import os


def Settings( **kwargs):
   conda_path = os.environ['CONDA_PREFIX']
   return {
        'interpreter_path': conda_path+'/bin/python'
    }
