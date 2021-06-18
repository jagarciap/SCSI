from mesh import Mesh_2D_rm_sat as m
wtf = self.children[0].potential
oki = m.vtkOrdering(self.children[0].pic.mesh, wtf)
m.saveVTK(self.children[0].pic.mesh, "whatever", {'potential': oki})
