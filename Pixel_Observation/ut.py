
import numpy as np
import gym
def disable_view_window():
    from gym.envs.classic_control import rendering
    org_constructor = rendering.Viewer.__init__

    def constructor(self, *args, **kwargs):
        org_constructor(self, *args, **kwargs)
        self.window.set_visible(visible=False)

    rendering.Viewer.__init__ = constructor
    
def crop_image(img): 
    new_y = int(1/4*img.shape[0])
    v = img.shape[0]
    new_x = int(2/3*img.shape[1]-20)
    img = img[new_y:v,0:new_x]
    img = img/255
    return img
    
    
