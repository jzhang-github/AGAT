INCAR_TAG = '''
SYSTEM
ISTART
ICHARG
INIWAV
ENCUT
ENAUG
PREC
IALGO
NELM
NELMIN
NELMDL
EDIFF
NBANDS
GGA
VOSKOWN
LREAL
WEIMIN
EDIFFG
NSW
IBRION
ISIF
POTIM
IOPT
ISYM
SIGMA
ISMEAR
ISPIN
MAGMOM
LWAVE
LCHARG
RWIGS
NPAR
LORBIT
LDAU
LDAUTYPE
LDAUL
LDAUU
LDAUJ
LDAUPRINT
LMAXMIX
LASPH
IDIPOL
LDIPOL
LAECHG
LADDGRID
NGX
NGY
NGZ
NGXF
NGYF
NGZF
ICHAIN
IMAGES
SPRING
LCLIMB
DdR
DRotMax
DFNMin
DFNMax
NFREE
LUSE_VDW
Zab_vdW
AGGAC
AMIX
AMIX_MAG
BMIX
BMIX_MAG
ALGO
KPAR
NCORE
NEDOS
IVDW
LELF
MDALGO
'''.split()

def modify_INCAR(key='NSW', value='300', s=''):
    if not key in INCAR_TAG:
        print('Input key not avaliable, please check.')
        return 1

    new_incar, discover_code = [], False
    with open('INCAR', 'r') as f:
        for line in f:
            str_list = line.split()
            if len(str_list) == 0:
                new_incar.append('\n')
            elif str_list[0] == key:
                str_list[2] = value
                new_incar.append(f'  {str_list[0]} = {str_list[2]}\n')
                discover_code = True
            else:
                new_incar.append(line)

    if s:
        new_incar.append(f'\n{s}\n')

    if not discover_code:
        new_incar.append(f'  {key} = {value}\n')

    with open('INCAR', 'w') as f:
        for line in new_incar:
            f.write(line)
    return 0