import constants as c
import mesh
import pic
import field
from Boundaries.inner_2D_rectangular import Inner_2D_Rectangular
from Boundaries.outer_1D_rectangular import Outer_1D_Rectangular
from Boundaries.outer_2D_rectangular import Outer_2D_Rectangular

#------------------------------------------------------------------------
# Functions
#------------------------------------------------------------------------

def mesh_setup(filelines):
    line = filelines.pop(0)
    while line[0] == '#':
        line = filelines.pop(0)
    assert line == '{', "Error reading the file. A \"{\" was expected. "+line+" was printed instead"
    line = filelines.pop(0)
    new_dic = {}
    #NOTE: I think it works right
    while line != '}':
        key, val = map(lambda x: x.strip(' \t'), line.split(sep = '='))
        if key == 'mesh':
            new_dic[key] = eval('mesh.'+val)
            assert filelines.pop(0).strip(' \t') == '{', "Error reading the file. A \'{\' was expected"
            new_dic['mesh_args'] = {}
            line = filelines.pop(0).strip(' \t')
            while line != '}':
                key, val = map(lambda x: x.strip( ' \t'), line.split(sep = '='))
                new_dic['mesh_args'][key] = eval(val)
                line = filelines.pop(0).strip(' \t')
        elif 'boundary' in key:
            new_dic[key] = eval(val)
            assert filelines.pop(0).strip(' \t') == '{', "Error reading the file. A \'{\' was expected"
            new_dic[key+'_args'] = {}
            line = filelines.pop(0).strip(' \t')
            while line != '}':
                key_in, val = map(lambda x: x.strip( ' \t'), line.split(sep = '='))
                new_dic[key+'_args'][key_in] = eval(val)
                line = filelines.pop(0).strip(' \t')
        elif key == 'pic':
            new_dic[key] = eval('pic.'+val)
            assert filelines.pop(0).strip(' \t') == '{', "Error reading the file. A \'{\' was expected"
            new_dic['pic_args'] = {}
            line = filelines.pop(0).strip(' \t')
            while line != '}':
                key, val = map(lambda x: x.strip( ' \t'), line.split(sep = '='))
                new_dic['pic_args'][key] = eval(val)
                line = filelines.pop(0).strip(' \t')
        elif key == 'field':
            new_dic[key] = eval('field.'+val)
            assert filelines.pop(0).strip(' \t') == '{', "Error reading the file. A \'{\' was expected"
            new_dic['field_args'] = {}
            line = filelines.pop(0).strip(' \t')
            while line != '}':
                key, val = map(lambda x: x.strip( ' \t'), line.split(sep = '='))
                new_dic['field_args'][key] = eval(val)
                line = filelines.pop(0).strip(' \t')
        line = filelines.pop(0).strip(' \t')

    ## Creation and linking of objects
    # List to be returned
    obj_list = []
    del_ind = None
    if len(filelines) > 1:
        obj_list = mesh_setup(filelines)

    #Linking parent-child
    for i in range(len(obj_list)):
        if new_dic['mesh_args']['id'] in obj_list[i][0].id:
            del_ind = i
            new_dic['mesh_args']['children'].append(obj_list[i][0])
            new_dic['pic_args']['children'].append(obj_list[i][1])
            new_dic['field_args']['children'].append(obj_list[i][2])

    #Deleting linked objects
    obj_list = obj_list[del_ind+1:] if del_ind != None else obj_list

    #Creating the new objects
    for i, boundary  in enumerate(new_dic['mesh_args']['boundaries']):
        new_dic['mesh_args']['boundaries'][i] = new_dic[boundary](*new_dic[boundary+'_args'].values())
        
    mesh = new_dic['mesh'](*new_dic['mesh_args'].values())
    new_dic['pic_args']['mesh'] = mesh
    pic = new_dic['pic'](*new_dic['pic_args'].values())
    new_dic['field_args']['pic'] = pic
    field = new_dic['field'](*new_dic['field_args'].values())

    #Finalizing the method
    obj_list.insert(0, (mesh, pic, field))
    return obj_list


def mesh_file_reader(filename):
    #Reading the file
    fileloc = "../domains/"+filename
    data = open(fileloc, 'r')

    #Calling the set up function
    returns = mesh_setup(data.read().split('\n'))
    assert len(returns) == 1, "There was a problem with the tree structure created"

    #Finishing
    data.close()
    return returns[0]
