import Adafruit_PCA9685
import json

pwm = Adafruit_PCA9685.PCA9685()
pwm.set_pwm_freq(60)

commands_json_file = "commands.json"

with open(commands_json_file) as json_file:
    commands = json.load(json_file)

def convert_direction_to_pwm(direction):
    if not -1 <= direction <= 1 :
        print('Not a valid direction')
        return 0
    return int((commands['right'] - commands['left']) * direction / 2 + commands['straight'])

def convert_speed_to_pwm(speed):
    if not 0 <= speed <= 1 :
        print('Not a valid speed')
        return 0
    if speed == 0 :
        return 300
    else :
        return int(speed * (commands['drive_max'] - commands['drive']) + commands['drive'])

def set_direction_angle(direction, pwm_object=pwm) :
    pwm_object.set_pwm(commands['direction'], 0, convert_direction_to_pwm(direction))
    return

def stop_all(pwm_object=pwm):
    pwm_object.set_all_pwm(0,0)
    return

def set_speed_forward(gas, pwm_object=pwm):
    pwm_object.set_pwm(commands['gas'],0,convert_speed_to_pwm(gas))
    return

def set_speed_backward(gas, pwm_object=pwm):
    return

def brake_car(pwm_object=pwm):
    pwm_object.set_pwm(commands['gas'], 0, commands['stop'])
    return

def reset_car(pwm_object=pwm):
    pwm_object.set_pwm(commands['direction'], 0, commands['straight'])
    pwm_object.set_pwm(commands['gas'], 0, commands['neutral'])
    return


