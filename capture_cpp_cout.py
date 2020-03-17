import os
import select
import sys

def capture_cpp_cout(f):
    """
    Executes the function f, without parameters, returns a tuple of (return of f, output written to cout)
    Solution found online: https://stackoverflow.com/questions/9488560/capturing-print-output-from-shared-library-called-from-python-with-ctypes-module
    """
    sys.stdout.write(' \b')
    pipe_out, pipe_in = os.pipe()
    stdout = os.dup(1)
    os.dup2(pipe_in, 1)
    def more_data():
            r, _, _ = select.select([pipe_out], [], [], 0)
            return bool(r)
    def read_pipe():
            out = ''
            while more_data():
                    out += str(os.read(pipe_out, 1024))
            return out

    ret = f()
    ret_txt = read_pipe()
    os.dup2(stdout, 1)
    os.close(pipe_out)
    os.close(pipe_in)
    os.close(stdout)
    return ret, ret_txt
