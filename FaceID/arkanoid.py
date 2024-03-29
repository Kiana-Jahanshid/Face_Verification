import cv2
import arcade 
import random 
import numpy as np
from frog import Frog 
from objects import Bug , Bee , Worm , Fish 
from ground import Ground
from face_identification import FaceIdentification
from create_facebank import CreateFaceBank
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

class Heart(arcade.Sprite):
    def __init__(self ):
        super().__init__("pics/like.png")
        self.center_x = 0
        self.center_y = 25 
        self.height = 30
        self.width = 30


class Game(arcade.Window):
    def __init__(self , accept_playing):
        super().__init__( width= 560 , height= 600 , title= "Arkanoid" )
        arcade.set_background_color(arcade.color.BLUE_GREEN)
        self.frog = Frog(self)
        self.ground = Ground(self)
        self.bees = arcade.SpriteList()
        self.bugs= arcade.SpriteList()
        self.fishes = arcade.SpriteList()
        self.worms = arcade.SpriteList()
        self.hearts = []
        self.flag = 0
        self.counter = 0
        self.score = 0 
        x = 460
        self.accept_playing = accept_playing


        for i in range(7):
            obj =Bug(35+80*i, 360)
            self.bugs.append(obj)     
            obj =Worm(35+80*i, 450)
            self.worms.append(obj)        
            
        for i in range(6):
            obj =Bee(75+80*i, 390)
            self.bees.append(obj)
            obj =Fish(80+80*i, 475)
            self.fishes.append(obj)            

        for i in range (3) :
            new_heart = Heart()
            self.hearts.append(new_heart)
            self.hearts[i].center_x += x
            x += 35



    def on_draw(self):
        arcade.start_render()
        self.frog.draw()
        self.ground.draw()
        arcade.draw_text(f"Score: {self.score}",20,15,arcade.color.WHITE , 23)
        
        for heart in self.hearts :          
            heart.draw()

        for bees in self.bees:
            bees.draw()
        for bugs in self.bugs:
            bugs.draw()
        for fishes in self.fishes:
            fishes.draw()
        for worms in self.worms:
            worms.draw()

        if len(self.hearts)== 0 :
            self.flag = 1
            self.clear()
            arcade.draw_text("GAME OVER",20,250,arcade.color.RED , 62 , bold=True)
          
        if self.accept_playing == False :
            self.flag = 1
            self.clear()
            arcade.draw_text("You dont have permission ",20,340,arcade.color.RED_DEVIL , 30 , bold=True)
            arcade.draw_text("to play this game",115,270,arcade.color.RED_DEVIL , 30 , bold=True)


    def on_key_press(self , symbol : int , modifiers : int):
        if symbol == arcade.key.RIGHT or symbol == arcade.key.SPACE :
            self.ground.change_x = 1
        elif symbol == arcade.key.LEFT or symbol == arcade.key.L :
            self.ground.change_x = -1
            
    def on_mouse_motion(self , x : int , y:int , dx: int , dy: int):
        if self.ground.width//2 -5< x < self.width - self.ground.width//2 -5 :
            self.ground.center_x = x
 

    def on_update(self, delta_time: float):
        self.frog.move()
        
        for bee in self.bees :
            if arcade.check_for_collision(self.frog , bee) :
                self.bees.remove(bee)
                self.frog.change_y=-1
                self.score +=1  
        for bug in self.bugs :
            if arcade.check_for_collision(self.frog , bug) :
                self.bugs.remove(bug)
                self.frog.change_y=-1
                self.score +=1
        for worm in self.worms :
            if arcade.check_for_collision(self.frog , worm) :
                self.worms.remove(worm)
                self.frog.change_y=-1
                self.score +=1  
        for fish in self.fishes :
            if arcade.check_for_collision(self.frog , fish) :
                self.fishes.remove(fish)
                self.frog.change_y=-1
                self.score +=1      

        if arcade.check_for_collision(self.frog,self.ground):
             self.frog.change_y=1

        if self.frog.center_y == self.height :
            self.frog.change_y = -1 

        if self.frog.center_x < 10 or self.frog.center_x > self.width - 10 :
            self.frog.change_x *= -1

        if self.frog.center_x < 0 or self.frog.center_x >= self.width : 
            self.frog.center_y *= -1
        
        index= 0
        if  len(self.hearts) != 0 :
            for heart in self.hearts :
                if self.frog.center_y == 0 :
                    self.score -= 1
                    del self.frog
                    self.frog = Frog(self)
                    self.hearts.remove(heart)
                    index += 1
            if len(self.hearts) == 0 :
                self.flag = 1



if __name__ == "__main__" :
    
    obj = FaceIdentification()
    app = obj.load_model()
    face_bank = np.load("face_bank.npy",allow_pickle=True)
    cam = cv2.VideoCapture(0)  
    temp =  True

    while temp:
        _, input_image = cam.read()

        results , face_bank = obj.load_facebank(input_image , app)
        accept_playing , temp = obj.identification(input_image , results , face_bank)

        cv2.imshow("Frame", input_image)
        if cv2.waitKey(1):
            break   
    cam.release()
    cv2.destroyAllWindows()

    window = Game(accept_playing)
    arcade.run()        
