import bpy
import bmesh
import math
from mathutils import Vector
from mathutils.interpolate import poly_3d_calc
from bpy.types import Scene, Mesh, MeshPolygon, Image

eps = 0.005 # 0.01

def getUVCoord(mesh:Mesh, face:MeshPolygon, point:Vector, image:Image):
    """ returns UV coordinate of target point in source mesh image texture
    mesh  -- mesh data from source object
    face  -- face object from mesh
    point -- coordinate of target point on source mesh
    image -- image texture for source mesh
    """
    # get active uv layer data
    uv_layer = mesh.uv_layers.active
    assert uv_layer is not None # ensures mesh has a uv map
    uv = uv_layer.data
    uv_lay = bm.loops.layers.uv.active
    # get 3D coordinates of face's vertices
    lco = [v.co for v in face.verts]
    # get uv coordinates of face's vertices
    luv = [loop[uv_lay].uv for loop in face.loops]
    # calculate barycentric weights for point
    lwts = poly_3d_calc(lco, point)
    # multiply barycentric weights by uv coordinates
    uv_loc = sum((p*w for p,w in zip(luv,lwts)), Vector((0,0)))
    # ensure uv_loc is in range(0,1)
    # TODO: possibly approach this differently? currently, uv verts that are outside the image are wrapped to the other side
    uv_loc = Vector((uv_loc[0] % 1, uv_loc[1] % 1))
    # convert uv_loc in range(0,1) to uv coordinate
#    image_size_x, image_size_y = image.size
#    x_co = round(uv_loc.x * (image_size_x - 1))
#    y_co = round(uv_loc.y * (image_size_y - 1))
    uv_coord = (uv_loc.x, uv_loc.y)# (x_co, y_co)

    # return resulting uv coordinate
    return Vector(uv_coord)


# reference: https://svn.blender.org/svnroot/bf-extensions/trunk/py/scripts/addons/uv_bake_texture_to_vcols.py
def getPixel(uv_pixels, img, uv_coord):
    """ get RGBA value for specified coordinate in UV image
    pixels    -- list of pixel data from UV texture image
    uv_coord  -- UV coordinate of desired pixel value
    """
    image_size_x, image_size_y = img.size
    uv_coord.x = round(uv_coord.x * (image_size_x - 1))
    uv_coord.y = round(uv_coord.y * (image_size_y - 1))
    pixelNumber = (img.size[0] * int(uv_coord.y)) + int(uv_coord.x)
    r = uv_pixels[pixelNumber*4 + 0]
    g = uv_pixels[pixelNumber*4 + 1]
    b = uv_pixels[pixelNumber*4 + 2]
    a = uv_pixels[pixelNumber*4 + 3]
    return (r, g, b, a)

def getAlpha(uv_pixels, img, uv_coord):
    """ get RGBA value for specified coordinate in UV image
    pixels    -- list of pixel data from UV texture image
    uv_coord  -- UV coordinate of desired pixel value
    """
    image_size_x, image_size_y = img.size
    uv_coord_x = round(uv_coord.x * (image_size_x - 1))
    uv_coord_y = round(uv_coord.y * (image_size_y - 1))
    pixelNumber = (img.size[0] * int(uv_coord_y)) + int(uv_coord_x)
    a = uv_pixels[pixelNumber*4 + 3]
    return a

scene = bpy.context.scene
obj = bpy.context.object
me = obj.data
image = me.materials[0].node_tree.nodes["Image Texture"].image
image_size_x, image_size_y = image.size
uv_pixels = image.pixels[:]
bm = bmesh.new()   # create an empty BMesh
bm.from_mesh(me)   # fill it in from a Mesh
#bm = bmesh.from_edit_mesh(me)

import sys
packages_path = "C:\\Users\\linkk\\AppData\\Roaming\\Python\\Python39\\site-packages"
sys.path.insert(0, packages_path )

import numpy as np
import cv2

alpha_map = np.zeros((image_size_x, image_size_y), dtype="uint8")
for x in range(image_size_x):
    for y in range(image_size_y):
        pixelNumber = (image.size[0] * int(y)) + int(x)
        a = uv_pixels[pixelNumber*4 + 3]
        alpha_map[x][y] = 255 if a > 0.5 else 0

#cv2.imshow("alpha", alpha_map)
#cv2.waitKey()

tri_map = np.zeros((image_size_x, image_size_y), dtype="uint8")
for face in bm.faces:
    uv_lay = bm.loops.layers.uv.active
    luv2 = [np.array((loop[uv_lay].uv[0] * image_size_x, loop[uv_lay].uv[1] * image_size_y)) for loop in face.loops]
    cv2.drawContours(tri_map, [np.array([np.flip(p).astype(int) for p in luv2])], 0, 255, -1)
#cv2.imshow("tri", tri_map)
#cv2.waitKey()

alpha_map = np.bitwise_and(alpha_map, tri_map)
#cv2.imshow("combined alpha", alpha_map)
#cv2.waitKey()

alpha_map = alpha_map.transpose()
contours, hierarchy = cv2.findContours(alpha_map, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)


# https://pyimagesearch.com/2021/10/06/opencv-contour-approximation/
approx_contours = []
import imutils
cnts = imutils.grab_contours((contours, hierarchy))
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, eps * peri, True)
    # draw the approximated contour on the image
    output = image.copy()
    approx_contours.append(approx)
    #    cv2.drawContours(output, [approx], -1, (0, 255, 0), 3)
contours = approx_contours

#result = np.ones((image_size_x, image_size_y), np.uint8) * 255

## fill spaces between contours by setting thickness to -1
#cv2.drawContours(result, contours, -1, 0, 3)
##cv2.drawContours(result_borders, contours, -1, 255, 1)

## xor the filled result and the borders to recreate the original image
##result = result_fill ^ result_borders

##cv2.imwrite('contours.png', result)
#cv2.imshow("result", result)
#cv2.waitKey()
#print(contours)

#contour_vertices = set()

#for contour in contours:
#        for point_array in contour:
#            point = point_array[0]
#            point_x, point_y = point[0], point[1]
#            contour_vertices.add((point_x, point_y))

#print(contour_vertices)



def barycentric(x1, y1, x2, y2, x3, y3, x, y):
    a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / (
        (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    )
    b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / (
        (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    )
    c = 1.0 - a - b
    return a, b, c

#tri_img_size = 500
#tri_img = np.zeros((tri_img_size, tri_img_size, 3), dtype="uint8")
#for face in bm.faces:
#    uv_lay = bm.loops.layers.uv.active
#    luv2 = [np.array(loop[uv_lay].uv) for loop in face.loops]
#    cv2.drawContours(tri_img, [np.array([(p*tri_img_size).astype(int) for p in luv2])], 0, (255,255,255), 2)
#for contour in contours:
#    for point_array in contour:
#        point_pixels = point_array[0]
#        point = np.array([point_pixels[0] / image_size_x, point_pixels[1] / image_size_y])
#        if point[0] > 1 or point[1] > 1:
#            print(point_pixels, image_size_x, image_size_y)
#            raise Exception("hi")
#        cv2.circle(tri_img, (point*tri_img_size).astype(int), 1, (0, 255, 0), 2)
#        
#cv2.imshow("tri", tri_img)
#cv2.waitKey()

uv_lay = bm.loops.layers.uv.active

def find_face_bary(point):
    face_barys = []        
    for face in bm.faces:
        # get 3D coordinates of face's vertices
        lco = [np.array(v.co) for v in face.verts]
        if len(lco) != 3:
            raise Exception("more than 3 vertexes on this triangle")
        # get uv coordinates of face's vertices
        luv = [np.array(loop[uv_lay].uv) for loop in face.loops]
        
        # calculate barycentric coordinates of uv point
        bary = barycentric(luv[0][0], luv[0][1], luv[1][0], luv[1][1], luv[2][0], luv[2][1], point[0], point[1])
        
        # dist
        index_max = np.argmax(np.array(bary))
        closest_bary = bary[index_max]
        if closest_bary <= 1 and min(bary) >= 0:
            dist = 0
        else:
            dist = np.linalg.norm(luv[index_max] - point)
        
        face_barys.append([face, lco, luv, bary, dist])
        
    face, lco, luv, bary, dist = min(face_barys, key=lambda x : x[4])

#    print(point, dist, luv, bary)

    if dist > 0:
        
        new_bary = np.array(bary)
        index_max = np.argmax(np.array(bary))
        for i in range(3):
            if bary[i] < 0:
                new_bary[index_max] += bary[i]
                new_bary[i] = 0
        bary_remaining = new_bary[index_max] - 1
        if bary_remaining > 0:
            new_bary[index_max] = 1
            new_bary[index_max-1] += bary_remaining/2
            new_bary[index_max-2] += bary_remaining/2
        
        new_point = np.matmul(np.array(luv).transpose(), (new_bary))
#        print(new_point)
#        
        point, bary = new_point, new_bary
        
#        tri_img_size = 500
#        tri_img = np.zeros((tri_img_size, tri_img_size, 3), dtype="uint8")
#        for face in bm.faces:
#            luv2 = [np.array(loop[uv_lay].uv) for loop in face.loops]
#            cv2.drawContours(tri_img, [np.array([(p*tri_img_size).astype(int) for p in luv2])], 0, (255,255,255), 2)
#        cv2.drawContours(tri_img, [np.array([(p*tri_img_size).astype(int) for p in luv])], 0, (0,255,255), 2)
#        cv2.circle(tri_img, (point*tri_img_size).astype(int), 1, (255, 0, 0), 2)
#        cv2.circle(tri_img, (new_point*tri_img_size).astype(int), 1, (0, 0, 255), 2)
#        #cv2.imshow("tri", tri_img)
        #cv2.waitKey()
        
    return point, face, bary

contours_with_info = []
for contour in contours:
    contour_with_info = []
    contours_with_info.append(contour_with_info)
    for point_array in contour:
        point_pixels = point_array[0]
        point = np.array([point_pixels[0] / image_size_x, point_pixels[1] / image_size_y])

        bary_info = find_face_bary(point)
        
        contour_with_info.append(bary_info)

extra_interval = 0.001
def isininterval(x, a, b):
    return a-extra_interval <= x <= b+extra_interval or b-extra_interval <= x <= a+extra_interval

def isinline(p, line):
    return isininterval(p[0], line[0][0], line[1][0]) and isininterval(p[1], line[0][1], line[1][1])

def line_intersection(line1, line2):
    # https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       return None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    
    if not isinline((x, y), line1) or not isinline((x, y), line2):
        return None
    
    return x, y

skip_close = 0.03

for edge in bm.edges:
    edge_verts = [edge.verts[0], edge.verts[1]]
    edge_uvs = [np.array((v.link_loops[0][uv_lay].uv[0], v.link_loops[0][uv_lay].uv[1])) for v in edge_verts]
    
    for contour in contours_with_info:
        new_points = []
        for index, bary_info in enumerate(contour):
            point1 = contour[index-1][0]
            point2 = contour[index][0]
            
            intersect = line_intersection(edge_uvs, [point1, point2])
            
            if intersect != None:
#                print(edge_uvs)
#                print([point1, point2])
#                tri_img_size = 500
#                tri_img = np.zeros((tri_img_size, tri_img_size, 3), dtype="uint8")
#                for face in bm.faces:
#                    luv2 = [np.array(loop[uv_lay].uv) for loop in face.loops]
#                    cv2.drawContours(tri_img, [np.array([(p*tri_img_size).astype(int) for p in luv2])], 0, (255,255,255), 2)
#                cv2.line(tri_img, (edge_uvs[0]*tri_img_size).astype(int), (edge_uvs[1]*tri_img_size).astype(int), (0,255,255), 2)
#                cv2.line(tri_img, (point1*tri_img_size).astype(int), (point2*tri_img_size).astype(int).astype(int), (0,255,255), 2)
#                cv2.circle(tri_img, (np.array(intersect)*tri_img_size).astype(int), 1, (0, 0, 255), 2)
#                cv2.imshow("tri", tri_img)
#                cv2.waitKey()

                intersect = np.array(intersect)

                if np.linalg.norm(point1 - intersect) < skip_close or np.linalg.norm(point2 - intersect) < skip_close:
                    continue

                bary_info = find_face_bary(intersect)
                new_points.append((index, bary_info))
        
        for index, bary_info in reversed(new_points):
            contour.insert(index, bary_info)

#tri_img_size = 500
#tri_img = np.zeros((tri_img_size, tri_img_size, 3), dtype="uint8")
#for face in bm.faces:
#    luv2 = [np.array(loop[uv_lay].uv) for loop in face.loops]
#    cv2.drawContours(tri_img, [np.array([(p*tri_img_size).astype(int) for p in luv2])], 0, (255,255,255), 2)

#for contour in contours_with_info:
#    for point, face, bary in contour:
#        cv2.circle(tri_img, (np.array(point)*tri_img_size).astype(int), 1, (0, 0, 255), 2)

#cv2.imshow("tri", tri_img)
#cv2.waitKey()

shape_names = []
for shape_name, shape_lay in bm.verts.layers.shape.items():
    shape_names.append(shape_name)

deform_names = []
for deform_name, shape_lay in bm.verts.layers.deform.items():
    deform_names.append(deform_name)
vertex_group_names = tuple(vertex_group.name for vertex_group in obj.vertex_groups)

import triangle as tr

triangulated_points = []
triangulated_triangles = []

for contour in contours_with_info:
    tr_poly = {"vertices": [], "segments": []}
    for index, bary_info in enumerate(contour):
        tr_poly["vertices"].append(list(bary_info[0]))
        tr_poly["segments"].append([(index - 1) % len(contour), index])
    triangulated = tr.triangulate(tr_poly, opts="p")
    
#    tri_img_size = 500
#    tri_img = np.zeros((tri_img_size, tri_img_size, 3), dtype="uint8")
#    allverts = triangulated["vertices"]
#    print(triangulated.keys())
#    for triangle in triangulated["triangles"]:
#        triverts = [allverts[triangle[0]], allverts[triangle[1]], allverts[triangle[2]]]
#        cv2.drawContours(tri_img, [np.array([(np.array(p)*tri_img_size).astype(int) for p in triverts])], 0, (255, 0, 0), 2)
#    cv2.imshow("tri", tri_img)
#    cv2.waitKey()

    indexconvert = {}
    for index, vert in enumerate(triangulated["vertices"]):
        point, face, bary = find_face_bary(vert)
        
        uv = point
        
        face_vert_pos = np.array([np.array((v.co[0], v.co[1], v.co[2])) for v in face.verts])
        pos = np.matmul(face_vert_pos.transpose(), bary)
        
        shapedata = []
        for shape_name in shape_names:
            for shape_name2, shape_lay in bm.verts.layers.shape.items():
                if shape_name == shape_name2:
                    face_shape_pos = np.array([np.array((v[shape_lay][0], v[shape_lay][1], v[shape_lay][2])) for v in face.verts])
                    shape_pos = np.matmul(face_shape_pos.transpose(), bary)
                    shapedata.append(shape_pos)
        
        deformdata = []
        for deform_name in deform_names:
            for deform_name2, deform_lay in bm.verts.layers.deform.items():
                if deform_name == deform_name2:
                    vertex_group_weights = []
                    for vertex_group_index, vertex_group_name in enumerate(vertex_group_names):
                        face_weights = np.array([([v[deform_lay][vertex_group_index]] if (vertex_group_index in v[deform_lay]) else 0) for v in face.verts])
                        weight = np.matmul(face_weights.transpose(), bary)
                        print(vertex_group_name, weight)
                        vertex_group_weights.append(weight)
                    deformdata.append(vertex_group_weights)
            
        point_data = [pos, uv, shapedata, deformdata]
        triangulated_points.append(point_data)
        indexconvert[index] = len(triangulated_points)-1
    
    for triangle in triangulated["triangles"]:
        newtriangle = [indexconvert[i] for i in triangle]
        triangulated_triangles.append(newtriangle)

bm.faces.ensure_lookup_table()
exampleface = bm.faces[0]

bm2 = bmesh.new()   # create an empty BMesh
print(bm2.verts.layers.deform.items())

for deform_name in deform_names:
    bm2.verts.layers.deform.new(deform_name)

createdverts = []
for pos, uv, shapedata, deformdata in triangulated_points:
    vert = bm2.verts.new(pos.tolist())
    createdverts.append(vert)
    for deform_index, deform_name in enumerate(deform_names):
        for deform_name2, deform_lay in bm2.verts.layers.deform.items():
            if deform_name == deform_name2:
                for vertex_group_index, vertex_group_name in enumerate(vertex_group_names):
                    weight = deformdata[deform_index][vertex_group_index]
                    if weight != 0:
                        vert[deform_lay][vertex_group_index] = weight

uv_lay = bm2.loops.layers.uv.new()

for triangle in triangulated_triangles:
    verts = [createdverts[i] for i in triangle]
    vertsdata = [triangulated_points[i] for i in triangle]
    face = bm2.faces.new(verts, exampleface)
    for i, vertdata in enumerate(vertsdata):
        uv = vertdata[1]
        loop = face.loops[i]
        loop[uv_lay].uv = uv.tolist()

bm2.to_mesh(me)

obj.shape_key_clear()

shape_keys = {}

for shape_name in shape_names:
    shape_key = obj.shape_key_add(name=shape_name)
    shape_keys[shape_name] = shape_key

for index, vertdata in enumerate(triangulated_points):
    shapedata = vertdata[2]
    for shapeindex, shape_name in enumerate(shape_names):
        shape_keys[shape_name].data[index].co = shapedata[shapeindex]


#print(image_size_x, image_size_y)

"""

shape_key_vals = {}
shape_names = []
for shape_name, shape_lay in bm.verts.layers.shape.items():
    shape_names.append(shape_name)

for vert in bm.verts:
    index = vert.index
    shape_key_vals[index] = {}
    for shape_name, shape_lay in bm.verts.layers.shape.items():
        shape_key_vals[index][shape_name] = vert[shape_lay]

obj.shape_key_clear()

me = obj.data
bm = bmesh.new()   # create an empty BMesh
bm.from_mesh(me)   # fill it in from a Mesh

#for x in bm.verts.layers.shape.items():
#    print(x)
#shape_lay = bm.verts.layers.shape["Fcl_EYE_Close"]

def loop_uv(loop, uv_lay):
    uv_coord = loop[uv_lay].uv
    uv_coord = Vector((uv_coord[0] % 1, uv_coord[1] % 1))
    return uv_coord

def lin_vectors2_2(l, v1, v2):
    return Vector((v1.x * l + v2.x * (1-l), v1.y * l + v2.y * (1-l)))

def lin_vectors3_2(l, v1, v2):
    return Vector((v1.x * l + v2.x * (1-l), v1.y * l + v2.y * (1-l), v1.z * l + v2.z * (1-l)))

#faces_to_remove = []
test = []

uv_layer = me.uv_layers.active
assert uv_layer is not None # ensures mesh has a uv map
uv = uv_layer.data
uv_lay = bm.loops.layers.uv.active

#for face in bm.faces:
##    remove_face = False
#    
    # get active uv layer data
#    for i, loop in enumerate(face.loops):
#        v = loop.vert
#        uv = loop_uv(loop, uv_lay)
#        
#        loop2 = face.loops[i-1]
#        v2 = loop2.vert
#        uv2 = loop_uv(loop2, uv_lay)
#        

def vert_uv(v, uv_lay):
    for loop in v.link_loops:
        return loop_uv(loop, uv_lay)
    print("No loops")
    return None

edge_vertices = []
for edge in bm.edges:
    v1, v2 = edge.verts
    
    uv1 = vert_uv(v1, uv_lay)
    uv2 = vert_uv(v2, uv_lay)
    
    if uv1 == None or uv2 == None:
        print("skipping edge, no loops")
        continue
    
    for edgetraverseint in range(0, 10):
        edgetraverse1 = edgetraverseint / 10
        edgetraverse2 = (edgetraverseint + 1) / 10
        euv1 = lin_vectors2_2(edgetraverse1, uv1, uv2)
        euv2 = lin_vectors2_2(edgetraverse2, uv1, uv2)
        a1 = getAlpha(uv_pixels, image, euv1) > 0.5
        a2 = getAlpha(uv_pixels, image, euv2) > 0.5
        
        if a1 != a2:
            edgetraversemid = edgetraverse1 + 0.05
            midpos = lin_vectors3_2(edgetraversemid, v1.co, v2.co)
            print(midpos, edge)
            edge_vertices.append([midpos, edge, v1 if a1 else v2, edgetraversemid if a1 else 1-edgetraversemid])

edge_vertices_pairs = []

for face in bm.faces:
    face_edge_vertices = []
    for edge in face.edges:
        for mid, midedge, vinside, vinsidedist in edge_vertices:
            if midedge == edge:
#                print(mid, midedge, vinside, edge)
                face_edge_vertices.append([mid, midedge, vinside, vinsidedist])
    
    if len(face_edge_vertices) % 2 != 0:
        print("error: not an even number")
        break
    
    remaining_edge_vertices = face_edge_vertices[:]
    
    face_edge_vertices_pairs = []
    
    while len(remaining_edge_vertices) > 0:
        point_to_match = remaining_edge_vertices.pop()
        
        current_edge, current_vert, current_dist = point_to_match[1], point_to_match[2], point_to_match[3]
        inside_verts = []
        while True:
            found_point = None
            for possible_point in remaining_edge_vertices:
                if possible_point[1] == current_edge:
                    if possible_point[2] != current_vert:
                        if 1-possible_point[3] < current_dist:
                            if found_point == None or possible_point[3] > found_point[3]:
                                found_point = possible_point
            if found_point:
                break
            inside_verts.append(current_vert)
            current_edge = [e for e in face.edges if current_vert in e.verts and e != current_edge][0]
            current_vert = [v for v in current_edge.verts if v != current_vert][0]
            current_dist = 1
        
        remaining_edge_vertices.remove(found_point)
        
        face_edge_vertices_pairs.append([point_to_match, found_point])
        print(point_to_match[0], found_point[0])

    edge_vertices_pairs.append([face, face_edge_vertices_pairs])
#        alpha = getAlpha(uv_pixels, image, uv_coord)
#        print(uv_coord, alpha)
#        if alpha < 0.5:
#            remove_face = True
#    
#    if remove_face:
#        faces_to_remove.append(face)
#    
#for face in test:
#    bm.faces.remove(face)


#bpy.ops.ed.undo_push()

edge_vertices_to_vert = []

connected = {}

for edge_vert in edge_vertices:
    vert = bm.verts.new(edge_vert[0])
    bm.verts. index_update()
    index = vert.index
    edge_vertices_to_vert.append([edge_vert, vert])
    connected[vert] = []
    
    shape_key_vals[index] = {}

    for shape_name in shape_names:
        on_edge = edge_vert[1]
        in_vert = edge_vert[2]
        out_vert = [v for v in on_edge.verts if v != in_vert][0]
        in_vert_dist = edge_vert[3]
        vert_shape_pos = lin_vectors3_2(in_vert_dist, shape_key_vals[in_vert.index][shape_name], shape_key_vals[out_vert.index][shape_name])
        shape_key_vals[index][shape_name] = vert_shape_pos

for face, face_edge_vertices_pairs in edge_vertices_pairs:
    for from_edge_vert, to_edge_vert in face_edge_vertices_pairs:
        v1 = [v for p, v in edge_vertices_to_vert if p == from_edge_vert][0]
        v2 = [v for p, v in edge_vertices_to_vert if p == to_edge_vert][0]
        if v2 in connected[v1]:
            print("double connected")
        else:
            connected[v1].append(v2)
            connected[v2].append(v1)
#                if from_edge_vert[1] == to_edge_vert[1]:
#                    midvert = bm.verts.new(lin_vectors3_2(0.5, from_edge_vert[0], to_edge_vert_[0]))
#                    bm.edges.new((v1, midvert))
#                    bm.edges.new((midvert, v2))
#                else:
#                    bm.edges.new((v1, v2))
            bm.edges.new((v1, v2))

bm.to_mesh(me)

shape_keys = {}

for shape_name in shape_names:
    shape_key = obj.shape_key_add(name=shape_name)
    shape_keys[shape_name] = shape_key

#me = obj.data
#bm = bmesh.new()   # create an empty BMesh
#bm.from_mesh(me)   # fill it in from a Mesh

#for shape_name, shape_lay in bm.verts.layers.shape.items():
for shape_name in shape_names:
#    for index, vert in enumerate(bm.verts):
#        vert[shape_lay] = shape_key_vals[index][shape_name]
    for index in shape_key_vals:
        shape_keys[shape_name].data[index].co = shape_key_vals[index][shape_name]

#bm.to_mesh(me)




bm.free()  # free and prevent further access

"""