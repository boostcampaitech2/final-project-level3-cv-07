import math, random

def generatePolygon(center_X,
                    center_Y, 
                    aveRadius, 
                    numVerts,
                    rand_plus):
    rand_minus=-rand_plus
    
    angleSteps = (2*math.pi / numVerts)
    points = []
    angle = random.uniform(0, 2*math.pi)
    for i in range(numVerts) :
        r_i = aveRadius+random.randint(rand_minus,rand_plus)
        x = center_X + math.cos(angle)*r_i
        y = center_Y + math.sin(angle)*r_i
        points.append((int(x),int(y)))

        angle -=angleSteps
    return points

def sort_polygonpoint(poly,control_point=(0,468)):
    min_dist=50000
    min_idx=0
    for idx,p in enumerate(poly):
        temp_dist=math.sqrt(math.fsum([(px - qx) ** 2.0 
                     for px, qx in zip(p, control_point)]))
        if min_dist > temp_dist:
            min_dist = temp_dist
            min_idx = idx
    return poly[min_idx:]+poly[0:min_idx]

def get_random_polygon():
    X=random.randint(250,700)
    Y=random.randint(100,300)
    R=random.randint(50,100)
    N_P = random.randint(2,5)
    N_P *= 2 
    randomP=random.randint(10,30)
    verts = generatePolygon(center_X=X,
                            center_Y=Y,
                            aveRadius=R,
                            numVerts=N_P,
                            rand_plus=randomP)
    return change_zero_ground(sort_polygonpoint(verts))

def change_zero_ground(lst):
    return [[x,468-y] for x,y in lst]


def get_random_point(min_xy=100,max_image=1000):
    rand_x=random.randint(0,max_image-min_xy)
    rand_y=random.randint(0,max_image-min_xy)
    rand_w=random.randint(min_xy,max_image-rand_x)
    rand_h=random.randint(min_xy,max_image-rand_y)
    return [rand_x,rand_y,rand_w,rand_h]
