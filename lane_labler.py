import numpy as np
from scipy import interpolate
from PIL import ImageTk, Image
import Tkinter as tk
import glob
import cv2
import numpy as np

class Controller():
    """
    Class that contains all the lanes, UI buttons and controls the movement and drawing of the lanes
    """
    
def compute_bezier_points(anchor_points, n):
    """
    returns n bezier points that fir a curve from the anchor points.
    These points can be used to draw a line
    """
    x = [p[0] for p in anchor_points]
    y = [p[1] for p in anchor_points]
    tck,u = interpolate.splprep( [x,y], k = 2)
    xnew,ynew = interpolate.splev( np.linspace( 0, 1, n ), tck,der = 0)
    return [c for p in zip(xnew, ynew) for c in p]

class Lane():
    unselected_colors = ['#c80000','#00c800','#0000c8']
    selected_colors =  ['#ff0000','#00ff00','#0000ff']
    unselected_anchor_size = 10
    selected_anchor_size = 15
    num_bezier_points = 41
    line_color = '#3d3d3d'
    line_width = 3
    deactivated_color = '#00FF33'    
    
    def __init__(self, canvas, activated, anchor_points=[[300,100], [300,300], [300,500]]):
        """
        initialize the lane
        """
        self.anchor_points = anchor_points
        self.line_id = None
        self.anchor_ids = []
        self.canvas = None

        self.selected_index = None
        self.activated = True

        self.canvas = canvas
        self.activated = activated
        line_fill = self.get_line_fill()
        bezier_points = compute_bezier_points(self.anchor_points, self.num_bezier_points)
        self.line_id = self.canvas.create_line(bezier_points, fill=line_fill, smooth=True, width=self.line_width, splinesteps=2)
        print self.canvas.coords(self.line_id)
        for index in range(len(self.anchor_points)):
            anchor_fill = self.get_anchor_fill(index)
            anchor_point = self.anchor_points[index]
            anchor_size = self.get_anchor_size(index)
            anchor_coordinates = self.square_centered_at_point(anchor_point, anchor_size)
            anchor_id = self.canvas.create_rectangle(anchor_coordinates, fill=anchor_fill)
            self.anchor_ids.append(anchor_id)      

    def update_lane_color(self):
        """
        Updates the components of the lane with the appropriate color.
        """
        line_fill = self.get_line_fill()
        self.line_id = self.canvas.create_line(self.anchor_points, fill=line_fill)
        for index in range(len(anchor_points)):
            anchor_fill = self.get_anchor_fill(index)            
            anchor_id = self.anchor_ids[index]
            self.canvas.itemconfig(anchor_id, fill=fill)
            
    def activate(self):
        """
        activates the line, by painting it appropriately
        """
        self.activated = True
        self.update_lane_color()
            
    def deactivate(self):
        """
        deactivates the line, by painting it appropriately
        """
        self.activated = False
        self.update_lane_color()
        
        
    def update_spline(self):
        """
        Updates the spline based on the anchor point locations
        """
        bezier_points = compute_bezier_points(self.anchor_points, self.num_bezier_points)
        
        self.canvas.coords(self.line_id, *bezier_points)
        
    def get_anchor_size(self, index):
        if not self.activated:
            return 2

        size = self.selected_anchor_size if self.selected_index == index else self.unselected_anchor_size
        return size

    def move_anchors(self, anchor_points):
        for index in range(len(anchor_points)):
            anchor_point = anchor_points[index]
            self.move_anchor(index, anchor_point)

    def move_anchor(self, index, anchor_point):
        """
        Moves the anchor to the selected location.  Updates the anchor and spline appropriately.
        """
        self.anchor_points[index] = anchor_point
        anchor_id = self.anchor_ids[index]
        # size of the anchor is based on whether it is selected
        size = self.get_anchor_size(index)
        # generate coordinates of anchor rectangle based on anchor point and size
        anchor_coordinates = self.square_centered_at_point(anchor_point, size)
        # move the anchor
        self.canvas.coords(anchor_id, *anchor_coordinates)
        # move the spline with the new anchor location
        self.update_spline()
        
    
    def square_centered_at_point(self, point, size):
        """
        given a point, creates a square if size centered around point.
        
        point is an array of size 2 looking like [row, col]
        """
        return [point[0] - (size/2),point[1] - (size/2),point[0] + (size/2),point[1] + (size/2)]
    
    def update_anchor(self, anchor_id, anchor_point, fill, size):
        """
        Updates an anchor's size and fill color
        """
        self.canvas.itemconfig(anchor_id, fill=fill)
        anchor_coordinates = self.square_centered_at_point(anchor_point, size)
        self.canvas.coords(anchor_id, *anchor_coordinates)
        
    def get_anchor_fill(self, index):
        if not self.activated:
            return self.deactivated_color
        
        fill = self.selected_colors[index] if self.selected_index == index else self.unselected_colors[index]
        return fill
    
    def get_line_fill(self):
        return self.line_color if self.activated else self.deactivated_color
    
    def deselect_all_anchors(self):
        for index in range(len(self.anchor_points)):
            self.deselect_anchor(index)

    def deselect_anchor(self, index):
        """
        Deselects the anchor at index.  Results in a smaller anchor and darker color.
        """
        anchor_point = self.anchor_points[index]
        anchor_id = self.anchor_ids[index]
        fill = self.get_anchor_fill(index)
        self.update_anchor(anchor_id, anchor_point, fill, self.unselected_anchor_size)
        self.selected_index = None
        
    def select_anchor(self, index):
        """
        Visually updates the selected anchor to be brighter and bigger.  Deselects any previously selected anchor.
        """
        self.deselect_all_anchors()

        if index is None:
            self.selected_index = None
            return 

        self.selected_index = index
        anchor_point = self.anchor_points[index]
        anchor_id = self.anchor_ids[index]
        fill = self.get_anchor_fill(index)
        self.update_anchor(anchor_id, anchor_point, fill, self.selected_anchor_size)
        self.canvas.tag_raise(anchor_id)

    def is_point_in_box(self, box, point):
        if point[0]< box[0]:
            return False
        if point[1] < box[1]:
            return False
        if point[0]> box[2]:
            return False
        if point[1] > box[3]:
            return False
        return True

    def get_clicked_anchor(self, point):
        """
        returns the index of an anchor underneath this point, returns None if no anchor is underneath the point
        """
        for index in range(len(self.anchor_points)):
            anchor_point = self.anchor_points[index]
            size = self.get_anchor_size(index)
            anchor_box = self.square_centered_at_point(anchor_point, size)
            if self.is_point_in_box(anchor_box, point):
                return index
        return None

    def __str__(self):
        s = '[ '
        for index in range(len(self.anchor_points)):
            anchor_point = self.anchor_points[index]
            s += "Anchor-" + str(index) + ':' + str(anchor_point) + ' '
        s +=']'
        return s

       
class ImagePlayer():
    def __init__(self, master, canvas, images, fps, load_image_fn, on_new_frame_fn=None):
        """
        load_image_fn(image_file) - function that takes in a filename and outputs an ImageTK object
        """
        self.master = master
        self.canvas = canvas
        self.is_playing = False
        self.fps = fps
        self.images = images
        self.frame = 0
        self.load_image = load_image_fn
        self.image = self.load_image(images[0])
        self.image_id = self.canvas.create_image(int(canvas['width'])/2,int(canvas['height'])/2,image=self.image)
        self.canvas.tag_lower(self.image_id)
        # on_new_frame_fn(image_file, frame)
        self.on_new_frame = on_new_frame_fn


    def compute_frame_delay(self):
        return 1000/self.fps

    def go_to_frame(self, frame):
        self.frame = frame
    def stop(self):
        """
        stops the video
        """
        self.is_playing = False

    def play(self):
        """
        plays the video
        """
        if self.frame >= len(self.images):
            self.is_playing = False
            print "Video is done playing.  call player.reset() to rewind to the beggining."
            return
        self.is_playing = True
        self.display_next_frame()

    def reset(self):
        self.go_to_frame(0)

    def display_next_frame(self):
        if not self.is_playing:
            return

        next_image = self.images[self.frame]
        self.image = self.load_image(next_image)
        self.canvas.itemconfig(self.image_id, image=self.image)
        self.canvas.tag_lower(self.image_id)
        if self.on_new_frame:
            self.on_new_frame(next_image, self.frame)
        self.frame+=1
        frame_delay = self.compute_frame_delay()
        self.master.after(frame_delay, self.display_next_frame)

    def set_speed(self, fps):
        self.fps = fps

def get_inverse_perspective_transform(): 
    return get_perspective_transform(inverse=True)

def get_perspective_transform(inverse=False):        
    p1 = (232, 380) # top left
    p2 = (360, 380) # top right
    p3 = (206, 414) # bottom left
    p4 = (383, 414) # bottom right

    pts1 = np.float32([list(p1),list(p2),list(p3),list(p4)])
    output_width = 100
    output_height = 35
    dx=250
    dy=585
    pts2 = np.float32([[dx,dy],[dx+output_width,dy],[dx,dy+output_height],[dx+output_width,dy+output_height]])
    if inverse:
        M = cv2.getPerspectiveTransform(pts2,pts1)
    else:
        M = cv2.getPerspectiveTransform(pts1,pts2)
    return M

def transform_anchor_point(point, M):
    return transform_anchor_points([point], M)[0]

def transform_anchor_points(points, M):
    points = np.array(points, dtype=np.float32)
    transformed = cv2.perspectiveTransform(points[None, :, :], M)
    return transformed.tolist()[0]

TRANSFORMATION_SIZE = 680
TRANSFORMATION_MATRIX = get_perspective_transform()
INVERSE_TRANSFORMATION_MATRIX = get_inverse_perspective_transform()

def load_topdown_image(image_filename):
    M = TRANSFORMATION_MATRIX
    img = cv2.imread(image_filename)
    img = cv2.cvtColor(img, cv2.cv.CV_BGR2RGB)
    dst = cv2.warpPerspective(img,M,(TRANSFORMATION_SIZE,TRANSFORMATION_SIZE))
    dst = cv2.cvtColor(dst, cv2.cv.CV_BGR2RGB)
    dst_img = Image.fromarray(dst)
    dst_img = ImageTk.PhotoImage(dst_img)
    return dst_img
        

def load_front_image(image_filename):
    pil_image = Image.open(image_filename)
    img = ImageTk.PhotoImage(pil_image)
    return img   

def get_front_lane_points(anchor_points, M, n=20):
    bezier_points = compute_bezier_points(anchor_points, n)
    bezier_points = [ bezier_points[i:i+2] for i in range(0,len(bezier_points),2)]
    transformed_anchor_points = transform_anchor_points(bezier_points, M)
    return transformed_anchor_points
                    

def run():
    master = tk.Tk()


    topdown_canvas = tk.Canvas(master, width=TRANSFORMATION_SIZE , height=TRANSFORMATION_SIZE)
    front_canvas = tk.Canvas(master, width=640 , height=480)
    front_canvas.pack(side='right', expand=False, fill='none')
    topdown_canvas.pack()
    # topdown_canvas.grid(row=0, column=0)
    # front_canvas.grid(row=0, column=1, sticky=tk.NW)
    #topdown_canvas.place(rely=1.0, relx=1.0, x=0, y=0, anchor='nw')
    #front_canvas.place(rely=1.0, relx=1.0, x=0, y=0, anchor='ne')

    global is_shift_pressed
    is_shift_pressed = False

    topdown_lanes = []
    front_lanes = []
    SHIFT_L = 'Shift_L'


    images = sorted(glob.glob('images/*.jpg'))

    def next_frame(image, frame):
        print image + " frame: " + str(frame)

    topdown_player = ImagePlayer(master, topdown_canvas, images, 5, load_topdown_image, next_frame)
    front_player = ImagePlayer(master, front_canvas, images, 5, load_front_image)


    def mouse_press(event):
        print "mouse press"
        lane_selected = False
        for lane in topdown_lanes:
            if lane_selected:
                lane.deselect_all_anchors()
                continue
            index = lane.get_clicked_anchor([event.x, event.y])
            lane.select_anchor(index)            
            if index is not None:
                lane_selected = True

    def key_press(event):  
        global is_shift_pressed 
        if event.keysym ==  SHIFT_L:
            is_shift_pressed = True
        if event.keysym == 'n':
            anchor_points = [[event.x, 100], [event.x,300], [event.x, 500]]
            topdown_lane = Lane(topdown_canvas, True, anchor_points=anchor_points)
            topdown_lanes.append(topdown_lane)

            front_anchor_points = get_front_lane_points(anchor_points, INVERSE_TRANSFORMATION_MATRIX)
            front_lane = Lane(front_canvas, False, anchor_points=front_anchor_points)
            front_lanes.append(front_lane)

    def move_selected_anchor(event):
        global is_shift_pressed
        if is_shift_pressed:
            for index in range(len(topdown_lanes)):
                topdown_lane = topdown_lanes[index]
                if topdown_lane.selected_index is not None: 
                    anchor_point =  [event.x, event.y]
                    topdown_lane.move_anchor(topdown_lane.selected_index, anchor_point)
                    anchor_points = topdown_lane.anchor_points
                    front_anchor_points = get_front_lane_points(anchor_points, INVERSE_TRANSFORMATION_MATRIX)
                    front_lane = front_lanes[index]
                    front_lane.move_anchors(front_anchor_points)

    def key_release(event):
        global is_shift_pressed
        if event.keysym == SHIFT_L:
            is_shift_pressed = False

    def stop_start(event):
        if topdown_player.is_playing:
            topdown_player.stop()
            front_player.stop()
        else:
            topdown_player.play()
            front_player.play()

    topdown_canvas.focus_set()
    topdown_canvas.bind('<Button-1>', mouse_press)
    topdown_canvas.bind('<Motion>', move_selected_anchor)
    topdown_canvas.bind('<Key>', key_press)
    topdown_canvas.bind('<KeyRelease>', key_release)
    topdown_canvas.bind('<space>', stop_start)

    tk.mainloop()
run()