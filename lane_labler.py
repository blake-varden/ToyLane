import numpy as np
from scipy import interpolate


class Controller():
    """
    Class that contains all the lanes, UI buttons and controls the movement and drawing of the lanes
    """
    
class Lane():
    unselected_colors = ['#c80000','#00c800','#0000c8']
    selected_colors =  ['#ff0000','#00ff00','#0000ff']
    initial_points = [[100,100], [200,200], [200,300]]
    unselected_anchor_size = 10
    selected_anchor_size = 15
    num_bezier_points = 10
    line_color = '#3d3d3d'
    line_width = 3
    deactivated_color = '#8a8a8a'    
    
    def __init__(self, canvas, activated):
        """
        initialize the lane
        """
        self.anchor_points = self.initial_points
        self.line_id = None
        self.anchor_ids = []
        self.canvas = None

        self.selected_index = None
        self.activated = True

        self.canvas = canvas
        self.activated = activated
        line_fill = self.get_line_fill()
        bezier_points = self.compute_bezier_points(self.num_bezier_points)
        print bezier_points
        self.line_id = self.canvas.create_line(bezier_points, fill=line_fill, smooth=True, width=self.line_width, splinesteps=2)
        print self.canvas.coords(self.line_id)
        for index in range(len(self.anchor_points)):
            anchor_fill = self.get_anchor_fill(index)
            anchor_point = self.anchor_points[index]
            anchor_coordinates = self.square_centered_at_point(anchor_point, self.unselected_anchor_size)
            anchor_id = self.canvas.create_rectangle(anchor_coordinates, fill=anchor_fill)
            self.anchor_ids.append(anchor_id)      
    
    def compute_bezier_points(self, n):
        """
        returns n bezier points that fir a curve from the anchor points.
        These points can be used to draw a line
        """
        x = [p[0] for p in self.anchor_points]
        y = [p[1] for p in self.anchor_points]
        tck,u = interpolate.splprep( [x,y], k = 2)
        xnew,ynew = interpolate.splev( np.linspace( 0, 1, n ), tck,der = 0)
        return [c for p in zip(xnew, ynew) for c in p]

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
        bezier_points = self.compute_bezier_points(self.num_bezier_points)
        
        self.canvas.coords(self.line_id, *bezier_points)
        
    def move_anchor(self, index, anchor_point):
        """
        Moves the anchor to the selected location.  Updates the anchor and spline appropriately.
        """
        self.anchor_points[index] = anchor_point
        anchor_id = self.anchor_ids[index]
        # size of the anchor is based on whether it is selected
        size = self.selected_anchor_size if self.selected_index == index else self.unselected_anchor_size
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
    
    def deselect_anchor(self, index):
        """
        Deselects the anchor at index.  Results in a smaller anchor and darker color.
        """
        anchor_point = anchor_points[index]
        anchor_id = anchor_ids[index]
        fill = self.get_anchor_fill(index)
        self.update_anchor(anchor_id, anchor_point, fill, self.unselected_anchor_size)
        self.selected_index = None
        
    def select_anchor(self, index):
        """
        Visually updates the selected anchor to be brighter and bigger.  Deselects any previously selected anchor.
        """
        if self.selected_index:
            deselect(self.selected_index)
        self.selected_index = index
        anchor_point = self.anchor_points[index]
        anchor_id = self.anchor_ids[index]
        fill = self.get_anchor_fill(index)
        self.update_anchor(anchor_id, anchor_point, fill, self.selected_anchor_size)
       
        
        
    

def create_lane():
    """
    creates a lane segment composed of 3 rectangle anchors connecting 2 splines.
    bottom color is red.
    top color i
    """
