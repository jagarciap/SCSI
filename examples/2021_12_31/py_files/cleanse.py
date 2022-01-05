import os
from subprocess import check_output
import sys

def cleanse():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cwd = os.path.split(os.getcwd())[0]
    cwd = cwd+'/results/'
    stdout = check_output('ls' +' {}'.format(cwd), shell=True)
    #files = list(filter(lambda x: '0-0-0-0_ts' in x, files)) 

    files = stdout.decode().split(sep='\n')
    files = list(filter(lambda x: x.partition('_')[0] == '0' and 'ts' in x.partition('_')[2], files)) 
    long_path = [cwd+file_ for file_ in files]
    for i in range(len(long_path)):
        if i%10 != 0:
            print("Eliminated: ", long_path[i])
            stdout_rm = check_output('rm' +' {}'.format(long_path[i]), shell=True)
            print(stdout_rm)

    files = stdout.decode().split(sep='\n')
    files = list(filter(lambda x: x.partition('_')[0] == '0-0' and 'ts' in x.partition('_')[2], files)) 
    long_path = [cwd+file_ for file_ in files]
    for i in range(len(long_path)):
        if i%10 != 0:
            print("Eliminated: ", long_path[i])
            stdout_rm = check_output('rm' +' {}'.format(long_path[i]), shell=True)
            print(stdout_rm)

    files = stdout.decode().split(sep='\n')
    files = list(filter(lambda x: x.partition('_')[0] == '0-0-0' and 'ts' in x.partition('_')[2], files)) 
    long_path = [cwd+file_ for file_ in files]
    for i in range(len(long_path)):
        if i%10 != 0:
            print("Eliminated: ", long_path[i])
            stdout_rm = check_output('rm' +' {}'.format(long_path[i]), shell=True)
            print(stdout_rm)
    
    files = stdout.decode().split(sep='\n')
    files = list(filter(lambda x: x.partition('_')[0] == '0-0-0-0' and 'ts' in x.partition('_')[2], files)) 
    long_path = [cwd+file_ for file_ in files]
    for i in range(len(long_path)):
        if i%10 != 0:
            print("Eliminated: ", long_path[i])
            stdout_rm = check_output('rm' +' {}'.format(long_path[i]), shell=True)
            print(stdout_rm)

cleanse()
