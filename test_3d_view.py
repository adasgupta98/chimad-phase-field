import fipy as fp

r = nx
cellSize = dx
radius = r - 0.5*cellSize

substring1 = '''
radius = {0};
cellSize = {1};
'''.format(r, cellSize)

mesh = fp.Gmsh2DIn3DSpace(substring1 + '''

// create inner 1/8 shell
Point(1) = {0, 0, 0, cellSize};
Point(2) = {-radius, 0, 0, cellSize};
Point(3) = {0, radius, 0, cellSize};
Point(4) = {0, 0, radius, cellSize};
Circle(1) = {2, 1, 3};
Circle(2) = {4, 1, 2};
Circle(3) = {4, 1, 3};
Line Loop(1) = {1, -3, 2};
Ruled Surface(1) = {1};

// create remaining 7/8 inner shells
t1[] = Rotate {{0,0,1},{0,0,0},Pi/2} {Duplicata{Surface{1};}};
t2[] = Rotate {{0,0,1},{0,0,0},Pi} {Duplicata{Surface{1};}};
t3[] = Rotate {{0,0,1},{0,0,0},Pi*3/2} {Duplicata{Surface{1};}};
t4[] = Rotate {{0,1,0},{0,0,0},-Pi/2} {Duplicata{Surface{1};}};
t5[] = Rotate {{0,0,1},{0,0,0},Pi/2} {Duplicata{Surface{t4[0]};}};
t6[] = Rotate {{0,0,1},{0,0,0},Pi} {Duplicata{Surface{t4[0]};}};
t7[] = Rotate {{0,0,1},{0,0,0},Pi*3/2} {Duplicata{Surface{t4[0]};}};

// create entire inner and outer shell
Surface Loop(100)={1, t1[0],t2[0],t3[0],t7[0],t4[0],t5[0],t6[0]};
''', order=2.0).extrude(extrudeFunc=lambda r: 1.1*r)

var = fp.CellVariable(mesh=mesh, value=mesh.x * mesh.y * mesh.z)
print mesh

view = fp.Viewer(var)
view.plot('viewer.png')

raw_input('stopped')
