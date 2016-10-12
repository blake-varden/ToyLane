import numpy as np
from scipy import interpolate
from PIL import ImageTk, Image
import Tkinter as tk
import glob
import cv2
import numpy as np
import time
import pickle
import copy
import argparse
import shutil
import re
parser = argparse.ArgumentParser()
parser.add_argument("--images", help="images_folder")
parser.add_argument("--recording", help="recoring location")
parser.add_argument("--frame", default=0, type=int, help="recoring location")
parser.add_argument("--load_recording", action='store_true', help="if set this will load from the recorind at the specified location")
args = parser.parse_args()


IS_DEBUG = False

def debug(s):
    if IS_DEBUG:
        print s
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

class Recording(object):
    DELETED = 'deleted'
    def __init__(self, recording_file=None, image_files=None, front_to_topdown_transformation=None, topdown_to_front_transformation=None, transformation_size=None, y_coordinates=None):
        if recording_file:
            self.load(recording_file)
        elif (image_files is not None and front_to_topdown_transformation is not None and topdown_to_front_transformation is not None and transformation_size):
            self.image_files = image_files
            self.lanes = {}
            self.front_to_topdown_transformation = front_to_topdown_transformation
            self.topdown_to_front_transformation = topdown_to_front_transformation
            self.transformation_size = transformation_size
            self.y_coordinates = y_coordinates
            # empty lane data for each image
            for image_file in image_files:
                self.lanes[image_file] = {}
        else:
            raise Exception("A recording must be created with either a recording file or an orded list of image files and transformation parameters.")
        

    def load(self, input_file):
        """
        Loads a recording from the file"
        """
        debug("Loading recording from " + input_file)
        with open(input_file) as f:
            recording = pickle.load(f)
        self.image_files = recording['image_files']
        self.lanes = recording['lanes']
        self.front_to_topdown_transformation = recording['front_to_topdown_transformation']
        self.topdown_to_front_transformation = recording['topdown_to_front_transformation']
        self.transformation_size = recording['transformation_size']
        self.y_coordinates = recording['y_coordinates']           

    def save(self, output_file, backup=False):
        debug("Saving recording to " + output_file)
        output = {
            'image_files' : self.image_files,
            'lanes' : self.lanes,
            'front_to_topdown_transformation' : self.front_to_topdown_transformation,
            'topdown_to_front_transformation' : self.topdown_to_front_transformation,
            'transformation_size' : self.transformation_size,
            'y_coordinates' : self.y_coordinates
        }
        with open(output_file, 'w') as f:
            pickle.dump(output, f)
        if backup:
            shutil.copyfile(output_file, output_file + '.bkup')


    def propagate_new_lanes(self, image_file):
        """
        Takes the lanes from previous frame that dont exist in current farme and propagates them to this frame
        """
        frame = self.image_files.index(image_file)
        if frame == 0:
            return
        prev_image = self.image_files[frame - 1]
        prev_lanes = self.lanes[prev_image]
        cur_lanes = self.lanes[image_file]
        for lane_id, lane_data in prev_lanes.iteritems():
            if lane_id not in cur_lanes and lane_data['lane_status'] != Recording.DELETED:
                cur_lanes[lane_id] = copy.deepcopy(lane_data)


    def create_lane(self, image_file, lane_id, anchor_points, lane_type=None, lane_status=None):
        lane_data = {
        'anchor_points' : list(anchor_points),
        'lane_type' : lane_type,
        'lane_status' : lane_status
        }
        self.lanes[image_file][lane_id] = lane_data

    def update_lane_anchor(self, image_file, lane_id, anchor_id, anchor_point, lane_type=None, lane_status=None):  
        lane_data = self.lanes[image_file][lane_id]
        lane_data['anchor_points'][anchor_id] = anchor_point
        lane_data['lane_status'] = lane_status
        if lane_type:
            lane_data['lane_type'] = lane_type
            

    def delete_lane(self, image_file, lane_id):
        frame = self.image_files.index(image_file)        
        self.lanes[image_file][lane_id]['lane_status'] = Recording.DELETED

        for i in range(frame+1, len(self.image_files)):
            frame_image = self.image_files[i]
            if lane_id in self.lanes[frame_image]:
                del self.lanes[frame_image][lane_id]

    def keep_lane(self, image_file, lane_id):
        self.lanes[image_file][lane_id]['lane_status'] = None

    def get_image_lanes(self, image_file):
        return self.lanes[image_file]

class Lane(object):
    unselected_colors = ['#c80000','#00c800','#0000c8']
    selected_colors =  ['#ff0000','#00ff00','#0000ff']
    unselected_anchor_size = 10
    selected_anchor_size = 15
    num_bezier_points = 41
    line_color = '#3d3d3d'
    line_width = 3
    deactivated_color = '#00FF33'    
    
    def __init__(self, canvas, activated, anchor_points=[[300,100], [300,300], [300,500]], lane_id=None):
        """
        initialize the lane
        """
        self.lane_id = lane_id if lane_id is not None else int(time.time()*100000)
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
        debug(str(self.canvas.coords(self.line_id)))
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
        old_anchor_point = self.anchor_points[index]
        # no need to do any updating if the anchor point hasn't changed
        if old_anchor_point == anchor_point:
            return

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

    def delete(self):
        self.canvas.delete(self.line_id)
        for anchor_id in self.anchor_ids:
            self.canvas.delete(anchor_id)

    def __str__(self):
        s = '[ '
        for index in range(len(self.anchor_points)):
            anchor_point = self.anchor_points[index]
            s += "Anchor-" + str(index) + ':' + str(anchor_point) + ' '
        s +=']'
        return s

       
class ImagePlayer(object):
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
        return int(1000/self.fps)

    def go_to_frame(self, frame):
        self.frame = frame
        self.update()

    def get_frame_image(self):
        return self.images[self.frame]   

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
            debug("Video is done playing.  call player.reset() to rewind to the beggining.")
            return
        self.is_playing = True
        self.display_next_frame()

    def reset(self):
        self.go_to_frame(0)

    def update(self):
        image_file = self.images[self.frame]
        self.image = self.load_image(image_file)
        self.canvas.itemconfig(self.image_id, image=self.image)
        self.canvas.tag_lower(self.image_id)


    def display_next_frame(self):
        if not self.is_playing:
            return        
        if self.frame >= len(self.images):
            self.is_playing = False
            debug("Video is done playing.  call player.reset() to rewind to the beggining.")
            return
        
        self.frame+=1
        self.update()
        if self.on_new_frame:
            num_prev_frames = max(1,self.fps/4)
            prev_image_files =self.images[max(0,self.frame-num_prev_frames):self.frame]
            image_file = self.images[self.frame]
            self.on_new_frame(image_file, prev_image_files, self.frame)
        
        frame_delay = self.compute_frame_delay()
        
        self.master.after(frame_delay, self.display_next_frame)


    def set_speed(self, fps):
        self.fps = fps


class Controller(object):

    transformation_size = (640, 480)
    
    """
    Class that contains all the lanes, UI buttons and controls the movement and drawing of the lanes
    """
    SHIFT_L = 'Shift_L'
    def __init__(self, images, recording_file, load_recording=False, fps=15, speed_profile=None, frame=0):
        """
        """
        self.master = tk.Tk()

        self.recording_file = recording_file
        self.load_recording = load_recording
        if load_recording:
            self.recording = Recording(recording_file=recording_file)
            self.transformation_matrix = self.recording.front_to_topdown_transformation
            self.inverse_transformation_matrix = self.recording.topdown_to_front_transformation
            self.y_coordinates = self.recording.y_coordinates        
        else:
            self.transformation_matrix = Controller.get_perspective_transform()
            self.inverse_transformation_matrix = Controller.get_inverse_perspective_transform()  
            self.y_coordinates = [10,380,470]           
            self.recording = Recording(image_files=images, 
                topdown_to_front_transformation=self.inverse_transformation_matrix, 
                front_to_topdown_transformation=self.transformation_matrix, 
                transformation_size=self.transformation_size,
                y_coordinates = self.y_coordinates)
        self.images = images
        self.speed_profile = speed_profile if speed_profile else [fps] * len(images)
        self.topdown_canvas = tk.Canvas(self.master, width=self.transformation_size[0], height=self.transformation_size[1])
        self.front_canvas = tk.Canvas(self.master, width=640 , height=480)
        self.front_canvas.pack(side='right', expand=False, fill='none')
        self.topdown_canvas.pack()  
        self.is_shift_pressed = False
        
        self.topdown_lanes = {}
        self.front_lanes = {}
        self.fps = fps
        self.topdown_player = ImagePlayer(self.master, self.topdown_canvas, images, self.fps, self.load_topdown_image, self.next_frame)
        self.front_player = ImagePlayer(self.master, self.front_canvas, images, self.fps, self.load_front_image)     

        self.topdown_canvas.focus_set()
        self.topdown_canvas.bind('<Button-1>', self.mouse_press)
        self.topdown_canvas.bind('<Motion>', self.mouse_move)
        self.topdown_canvas.bind('<Key>', self.key_press)
        self.topdown_canvas.bind('<KeyRelease>', self.key_release)
        self.topdown_canvas.bind('<space>', self.stop_start)
        self.go_to_frame(frame)

    def go_to_frame(self, frame):
        self.topdown_player.go_to_frame(frame)
        self.front_player.go_to_frame(frame)
        image = self.images[frame]
        self.next_frame(image, None, frame)

    def run(self):
        tk.mainloop() 

    def set_fps(self, fps):
        if fps == self.fps:
            return
        print 'fps : ' + str(fps)
        self.fps = fps
        self.topdown_player.fps = fps
        self.front_player.fps = fps 

    def increase_fps(self):
        self.set_fps(self.fps + 1)

    def decrease_fps(self):
        self.set_fps(self.fps - 1)
        

    def next_frame(self, image, prev_image_files, frame):
        """
        Plays Back the recording.  The recording will be update dby any movement
        """
        # propagate lanes from previous frame to this frame.  If lanes from previous frame
        # exists in this frame or are marked for deletion, they wont be propagated
        print "Frame: " + str(frame) + " " + image
        self.recording.propagate_new_lanes(image)   
        self.set_fps(self.speed_profile[frame])
        # iterate through the lanes for this frame
        # visually update them     
        lanes = self.recording.get_image_lanes(image)
        for lane_id, lane_data in lanes.iteritems():

            # if the lane is deleted this frame, delete it
            lane_deleted = lane_data['lane_status'] == Recording.DELETED
            if lane_deleted and self.is_lane_being_moved(lane_id):
                # lane was recorded to be deleted, but it is being moved so we shouldnt delete it
                self.recording.keep_lane(image, lane_id)
            elif lane_deleted:
                self.delete_lane(lane_id)
                continue

                        
            if lane_id in self.topdown_lanes:
                # if the lane is already being drawn from previous frame, update it

                if self.is_lane_being_moved(lane_id):
                    # if the lane is being moved by the user at this frame change
                    # update the lane recorded in this frame to have the location of the moving anchor
                    selected_lane = self.topdown_lanes[lane_id]
                    selected_index = selected_lane.selected_index
                    selected_anchor = selected_lane.anchor_points[selected_index]                    
                    self.recording.update_lane_anchor(image, lane_id, selected_index, selected_anchor)
                    # two most recent frames have the same anchors to account for slow reaction time in user
                    if prev_image_files is not None:
                        for prev_image in prev_image_files:
                            self.recording.update_lane_anchor(prev_image, lane_id, selected_index, selected_anchor)

                anchor_points = list(lane_data['anchor_points'])                
                self.update_lane(lane_id, anchor_points)
            else:
                # if the lane isn't being drawn already, draw it
                anchor_points = list(lane_data['anchor_points'])
                self.create_lane(lane_id, anchor_points)

        # save recording every 5 frames
        if frame % 300 == 0:
            self.recording.save(self.recording_file, backup=self.load_recording)
            # self.recording.save(self.recording_file + '.bkup')
    
    def is_lane_being_moved(self, lane_id):
        if lane_id not in self.topdown_lanes:
            return False
        topdown_lane = self.topdown_lanes[lane_id]
        is_selected = topdown_lane.selected_index is not None
        # don't update the position if it is being moved
        return self.is_shift_pressed and is_selected


    def update_lane(self, lane_id, anchor_points, record=True):
        image = self.topdown_player.get_frame_image()
        topdown_lane = self.topdown_lanes[lane_id]
        topdown_lane.move_anchors(anchor_points)

        front_lane = self.front_lanes[lane_id]
        front_anchor_points = self.get_front_lane_points(anchor_points)
        front_lane.move_anchors(front_anchor_points) 

    def get_front_lane_points(self, anchor_points, n=20):
        bezier_points = compute_bezier_points(anchor_points, n)
        bezier_points = [ bezier_points[i:i+2] for i in range(0,len(bezier_points),2)]
        transformed_anchor_points = Controller.transform_anchor_points(bezier_points, self.inverse_transformation_matrix)
        return transformed_anchor_points

    def create_lane(self, lane_id, anchor_points):
        image = self.topdown_player.get_frame_image()
        # create a lane using lane_id and anchor_points
        topdown_lane = Lane(self.topdown_canvas, True, anchor_points=anchor_points, lane_id=lane_id)
        lane_id = topdown_lane.lane_id
        self.topdown_lanes[lane_id] = topdown_lane

        # create the corresponding front view lane
        front_anchor_points = self.get_front_lane_points(anchor_points)
        front_lane = Lane(self.front_canvas, False, anchor_points=front_anchor_points, lane_id=lane_id)
        self.front_lanes[lane_id]= front_lane
        return lane_id

    def delete_lane(self, lane_id):
        if lane_id in self.topdown_lanes:
            self.topdown_lanes[lane_id].delete()
            del self.topdown_lanes[lane_id]

        if lane_id in self.front_lanes:
            self.front_lanes[lane_id].delete()
            del self.front_lanes[lane_id]

    def mouse_press(self, event):
        debug("mouse press")
        lane_selected = False
        for lane_id, lane in self.topdown_lanes.iteritems():
            if lane_selected:
                lane.deselect_all_anchors()
                continue
            index = lane.get_clicked_anchor([event.x, event.y])
            lane.select_anchor(index)            
            if index is not None:
                lane_selected = True

    def get_selected_lane_id(self):
        for lane_id, topdown_lane in self.topdown_lanes.iteritems():
            topdown_lane = self.topdown_lanes[lane_id]
            if topdown_lane.selected_index is not None: 
                return lane_id
        return None       

    def adjust_selected_anchor(self, dx=0, dy=0):
        lane_id = self.get_selected_lane_id()
        if lane_id is not None:
            lane = self.topdown_lanes[lane_id]
            anchor_point = list(lane.anchor_points[lane.selected_index])
            anchor_point[0] += dx
            anchor_point[1] += dy
            self.move_selected_anchor(lane_id, anchor_point)


    def key_press(self, event): 
        debug(str(event.keysym))
        if event.keysym == 'Up':
            self.increase_fps()
        if event.keysym == 'Down':
            self.decrease_fps()

        if event.keysym == 'Left':
            self.adjust_selected_anchor(dx=-2)

        if event.keysym == 'Right':
            self.adjust_selected_anchor(dx=2)            

        if event.keysym ==  Controller.SHIFT_L:
            debug("shift pressed")
            self.is_shift_pressed = True
        if event.keysym == 'n':
            debug("creating a lane")
            anchor_points = [[event.x, self.y_coordinates[0]], [event.x, self.y_coordinates[1]], [event.x, self.y_coordinates[2]]]
            # visually create the  lane
            lane_id = self.create_lane(None, anchor_points)
            image = self.topdown_player.get_frame_image()
            # record the lane in this frame
            self.recording.create_lane(image, lane_id, anchor_points)

        if event.keysym == 'd':
            debug("deleting a lane")
            lane_id = self.get_selected_lane_id()
            if lane_id is not None:
                # visually delete the frame                
                self.delete_lane(lane_id)
                # record that this lane is to be deleted
                image = self.topdown_player.get_frame_image()
                self.recording.delete_lane(image, lane_id)

    def mouse_move(self, event):
        #debug("mouse move: shift: " + str(self.is_shift_pressed))
        lane_id = self.get_selected_lane_id()
        if self.is_shift_pressed and lane_id is not None:
            self.move_selected_anchor(lane_id,[event.x, event.y])



    def move_selected_anchor(self, lane_id, anchor_point):
        topdown_lane = self.topdown_lanes[lane_id]
        # update anchor points of the
        anchor_index =  topdown_lane.selected_index
        anchor_points = list(topdown_lane.anchor_points)
        anchor_point[1] = anchor_points[anchor_index][1]
        anchor_points[anchor_index] = anchor_point
        
        self.update_lane(lane_id, anchor_points, record=False)
        
        image = self.topdown_player.get_frame_image()                        
        self.recording.update_lane_anchor(image, lane_id, anchor_index, anchor_point)


    def key_release(self, event):
        if event.keysym == Controller.SHIFT_L:
            debug("shift released")
            self.is_shift_pressed = False

    def stop_start(self, event):
        if self.topdown_player.is_playing:
            self.topdown_player.stop()
            self.front_player.stop()
        else:
            self.topdown_player.play()
            self.front_player.play()

    @staticmethod
    def get_inverse_perspective_transform(): 
        return Controller.get_perspective_transform(inverse=True)

    @staticmethod
    def get_perspective_transform(inverse=False):        
        p1 = (232, 380) # top left
        p2 = (360, 380) # top right
        p3 = (206, 414) # bottom left
        p4 = (383, 414) # bottom right

        pts1 = np.float32([list(p1),list(p2),list(p3),list(p4)])
        output_width = 85
        output_height = 35
        dx=276
        dy=400
        pts2 = np.float32([[dx,dy],[dx+output_width,dy],[dx,dy+output_height],[dx+output_width,dy+output_height]])
        if inverse:
            M = cv2.getPerspectiveTransform(pts2,pts1)
        else:
            M = cv2.getPerspectiveTransform(pts1,pts2)
        return M

    @staticmethod
    def transform_anchor_points(points, M):
        points = np.array(points, dtype=np.float32)
        transformed = cv2.perspectiveTransform(points[None, :, :], M)
        return transformed.tolist()[0]

    def load_topdown_image(self, image_filename):
        img = cv2.imread(image_filename)
        img = cv2.cvtColor(img, cv2.cv.CV_BGR2RGB)
        dst = cv2.warpPerspective(img, self.transformation_matrix, (self.transformation_size[0], self.transformation_size[1]))
        dst = cv2.cvtColor(dst, cv2.cv.CV_BGR2RGB)
        dst_img = Image.fromarray(dst)
        dst_img = ImageTk.PhotoImage(dst_img)
        return dst_img
            
    def load_front_image(self, image_filename):
        pil_image = Image.open(image_filename)
        img = ImageTk.PhotoImage(pil_image)
        return img   

def main(args):
    images = sorted(glob.glob(args.images + '/*.jpg'), key = lambda s :int(re.match('.*?(\d+)\.jpg', s).group(1)))
    speed_profile = np.array([10]* len(images))
    speed_profile[:100] = 6
    speed_profile[230:570] = 20
    speed_profile[571:816] = 7
    speed_profile[817:1414] = 20
    speed_profile = speed_profile.tolist()
    recording_file = args.recording
    c = Controller(images, recording_file, load_recording=args.load_recording, speed_profile=speed_profile, frame=args.frame)    
    try:
        c.run()
    finally:
        print "Saving"
        c.recording.save(c.recording_file, backup=args.load_recording)

if __name__ == '__main__':
    main(args)